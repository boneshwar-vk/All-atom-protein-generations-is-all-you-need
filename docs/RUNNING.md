# Running the GRPO binder pipeline

Two supported ways to run the real (non-mock) pipeline. Pick by your hardware.

| | **Linux + A100 (online)** | **MacBook M5 (stage-wise)** |
|---|---|---|
| Mode | `aapg-grpo` online cascade (RFD3→MPNN→fold→Rosetta **per candidate**, early-exit) | each stage a **separate process**, run to completion before the next |
| Why | A100 (40/80 GB) holds every model at once; fastest throughput | 24 GB unified memory can't co-resident all models → isolate per stage so RAM never stacks |
| Scorer | Boltz-2 (open) or Protenix v2 (licensed) | Boltz-2 (MPS); PX2 unverified on Apple Silicon |
| Throughput (8 h) | ~**150–320 complexes / A100** (folding-bound) | ~**30–60 complexes** total (see caveats) |

Both reuse the same adapters, configs, weights, and reward. The CPU/mock health check (`scripts/smoke_test.sh`) works on either machine and needs no GPU/weights — run it first.

> **The honest caveats up front.** RFdiffusion3 (via RosettaCommons/foundry) and Boltz-2 are
> **CUDA-oriented**. On A100 everything is native. On Apple Silicon they run on MPS/CPU at best,
> RFD3 may not run on MPS at all, and the folding stage is the bottleneck + the main OOM risk.
> **Protenix v2 weights are proprietary** and are never downloaded by this repo. **PyRosetta is
> separately licensed.**

---

## Part 1 — Linux + A100 (online GRPO)

### 1.1 Prerequisites
- Linux x86-64, NVIDIA driver + CUDA 12.x, one or more A100 (40 or 80 GB).
- `git`, `uv` (https://docs.astral.sh/uv/), ~50 GB free disk for weights + outputs.

### 1.2 Install
```bash
git clone https://github.com/boneshwar-vk/All-atom-protein-generations-is-all-you-need.git
cd All-atom-protein-generations-is-all-you-need
uv sync --all-packages --extra full          # core + hydra + polars (CPU libs; no heavy deps yet)

# Heavy stacks (into the project venv). Match the torch CUDA build to your driver:
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install boltz                          # Boltz-2 (MIT) — the default structure scorer
uv pip install "rc-foundry[all]"              # RFdiffusion3 (RFD3) via foundry
# PyRosetta (licensed; not on PyPI). The installer helper is the easy path:
uv pip install pyrosetta-installer && \
  uv run python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
# Optional, licensed only: uv pip install protenix   # then drop v2 weights (see 1.3)
```

### 1.3 Weights
```bash
make weights-download        # Boltz-2 -> data/weights/boltz2 ; RFD3 -> data/weights/rfd3 ; verifies ProteinMPNN
# Protenix v2 (optional, licensed): place weights in data/weights/protenix_v2/ and the
# structure_auto scorer auto-switches to it; otherwise the run stays on open Boltz-2.
```
ProteinMPNN weights are vendored in the repo — nothing to download.

### 1.4 The EGFR target
Put your EGFR structure (e.g. ectodomain PDB `1IVO`/`3NJP`, or a single domain) under `data/`,
then edit `configs/experiment/grpo_egfr.yaml`: set `target.structure_path`, `target.chains`,
`target.hotspots` (the epitope residues), and `target.binder_length`. See
`docs/EGFR_smoke_test.md` for how to pick the domain + hotspots.

### 1.5 Run (single A100)
```bash
export AAPG_ROOT="$PWD"
uv run --extra full aapg-grpo +experiment=grpo_egfr \
  steering.mock=false \
  steering.n_backbones=512 steering.seqs_per_backbone=4 \
  steering.group_size=64 steering.n_rounds=1 steering.top_k=64 \
  scorer=structure_auto designer=proteinmpnn \
  +steering.output_path=experiments/outputs/egfr_a100/ranked.parquet
```
Knobs: `n_backbones` × `seqs_per_backbone` = candidates scored; the cost-ordered cascade
early-exits (cheap structure stage gates before Rosetta), so most compute is folding survivors.
Switch backend with `scorer=boltz2` / `scorer=protenix`; designer with `designer=protonmpnn`
(placeholder) / `proteinmpnn`.

### 1.6 Multiple A100s
v1 selection is single-process (the `launcher` group is a stub). To use N GPUs, run **N independent
shards** with different seeds and merge:
```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g uv run --extra full aapg-grpo +experiment=grpo_egfr \
    steering.mock=false +steering.seed=$g steering.n_backbones=512 steering.seqs_per_backbone=4 \
    +steering.output_path=experiments/outputs/egfr_a100/shard_$g.parquet &
done; wait
```
`+steering.seed=$g` (leading `+`) — `seed` is not a preset key in the `steering` group, so a bare
`steering.seed=...` is a Hydra compose error. Then merge the shards:
```python
import glob, polars as pl

pl.concat(
    [pl.read_parquet(f) for f in glob.glob("experiments/outputs/egfr_a100/shard_*.parquet")]
).sort("reward", descending=True).write_parquet("experiments/outputs/egfr_a100/ranked.parquet")
```

### 1.7 Throughput (A100, rough)
Folding dominates. A single ~250–700 aa complex on Boltz-2: ~1–3 min on an A100. RFD3 backbone
~10–60 s, ProteinMPNN seconds, Rosetta interface ~1–2 min (CPU). Folding-bound:
**~150–320 fully-scored complexes per A100 in 8 h**, ×(GPU count) with sharding. Smaller target +
`recycling_steps` low pushes toward the high end.

---

## Part 2 — MacBook M5 (stage-wise, memory-safe)

24 GB unified memory cannot hold RFD3 + ProteinMPNN + Boltz-2 + PyRosetta at once, so do **not**
run the online cascade. Run each stage as its **own process** that finishes and exits (freeing its
model's memory) before the next starts. Persist each stage's output to disk and feed it to the
next.

### 2.1 Prerequisites + install
```bash
uv sync --all-packages --extra full
uv pip install torch                 # arm64 wheel ships MPS support
uv pip install boltz
uv pip install "rc-foundry[all]"     # RFD3 (Stage 1) — see the Apple-Silicon caveat below
uv pip install pyrosetta-installer && \
  uv run python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"  # Apple Silicon build
make weights-download                # Boltz-2 + RFD3 into data/weights/ (ProteinMPNN is vendored)
export PYTORCH_ENABLE_MPS_FALLBACK=1 # let ops MPS doesn't implement fall back to CPU (slower, but runs)
```
**RFD3 on Apple Silicon is the big unknown** — foundry/RFD3 is CUDA-oriented and may not run on
MPS. Test it on **one** backbone first (Stage 1 with `--n 1` below). If it won't run, generate
backbones on a CUDA box (Part 1 install + the `rfd3` stage only) and copy `data/weights/egfr_mac/`'s
`backbones.json` + `backbones/` over; ProteinMPNN/Boltz/Rosetta are fine on Mac. (Boltz-2 weights
also auto-download on first fold, so `make weights-download` is optional for the fold stage.)

### 2.2 Memory-safety rules (keep you under 24 GB)
- **One stage = one process.** Never run two stages at once.
- **Fold one complex at a time** (no batching), `diffusion_samples=1`, `recycling_steps=1`.
- **Keep the target small** — a single EGFR domain (~150–200 aa), not the full ~620 aa ectodomain.
  Folding memory + time scale steeply with total residues; 24 GB is comfortable at ~250 aa total,
  risky past ~500.
- Watch memory in another terminal: `while true; do vm_stat | awk '/free|inactive/'; sleep 30; done`.

### 2.3 Stage-wise run (`aapg-grpo-stage`)
Each stage is a subcommand of the `aapg-grpo-stage` runner, run as its **own process** so the
previous stage's model memory is released before the next loads. Every stage writes a manifest
under `$RUN` and is **resumable** — re-running a stage skips the ids already recorded (append +
dedupe), so an interrupted run (or the fold stage's memory bail-out, below) continues where it
left off. Reuse the repo's adapters, the shared scorer registry and the gated reward; nothing is
reimplemented. Add `--mock` to any stage for a CPU dry run with no GPU/weights.

**One-liner:** `scripts/macbook/run_all.sh` chains all five stages (one `uv run` process each) with
env-var knobs (`RUN`, `N`, `SEQS_PER`, `BACKEND`, `MIN_FREE_GB`, `TARGET_JSON`/`STRUCTURE_PATH`…;
set `MOCK=1` for the dry run). Run `scripts/macbook/run_all.sh --help` for the full list.

```bash
export RUN="experiments/outputs/egfr_mac"; mkdir -p "$RUN"
# Describe the target once (or pass --structure-path/--chains/--hotspots/--binder-length flags):
cat > "$RUN/target.json" <<'JSON'
{"structure_path": "data/egfr_domainIII.pdb", "chains": ["A"], "binder_chains": ["B"],
 "hotspots": ["A465", "A467", "A468"], "binder_length": 80}
JSON
T=(--run "$RUN" --target-json "$RUN/target.json")
```

**Stage 1 — RFD3 (all backbones first):** sample `N` backbones; each binder CA trace is saved to
`$RUN/backbones/<id>.npy` with a row in `$RUN/backbones.json` (skips if `>= N` already exist,
unless `--force`).
```bash
uv run aapg-grpo-stage rfd3 "${T[@]}" --n 60 --seed 0 \
  --checkpoint data/weights/rfd3/base.ckpt --rfd3-command rfd3   # tune --n to your time budget
```
(The exact RFD3 checkpoint filename under `data/weights/rfd3/` is foundry/version-dependent — see
`configs/model/rfd3.yaml` and `foundry list-installed`; point `--checkpoint` at the actual file.)

**Stage 2 — ProteinMPNN (sequences for all backbones):** designs `--seqs-per` sequences on each
backbone not yet in `$RUN/sequences.jsonl`.
```bash
uv run aapg-grpo-stage mpnn "${T[@]}" --seqs-per 4 --designer proteinmpnn --weights-set soluble
```

**Stage 3 — fold + score (one complex at a time):** rebuilds each candidate, folds + scores it
with the structure backend (CIF persisted to `$RUN/structures` for Stage 4), computes scRMSD, and
appends a row to `$RUN/structure.jsonl`. A **memory guard** checks free RAM before each fold; if it
cannot stay above `--min-free-gb` it `gc.collect()`s, clears the torch MPS/CUDA cache, re-checks,
and — still low — exits `75` with progress already checkpointed (just re-run `fold` to resume).
```bash
uv run aapg-grpo-stage fold "${T[@]}" --backend boltz2 \
  --recycling-steps 1 --diffusion-samples 1 --min-free-gb 4 --weights-dir data/weights/boltz2
# --backend structure_auto auto-uses Protenix v2 when its licensed weights are present
# (--protenix-weights-dir), else Boltz-2.
```

**Stage 4 — PyRosetta interface (on the predicted complexes):** scores `rosetta_interface` on the
predicted CIF from Stage 3 (falling back to the backbone only in `--mock`, where the mock ignores
the structure) and appends to `$RUN/rosetta.jsonl`.
```bash
uv run aapg-grpo-stage rosetta "${T[@]}"
```

**Stage 5 — reward + rank:** joins `structure.jsonl` + `rosetta.jsonl` per `cand`, applies
`IpSAEGatedReward` from `configs/reward/ipsae_gated.yaml`, sorts by reward, and writes the ranked
table (format follows the `--output` extension: `.jsonl`/`.json` → JSON Lines, `.parquet` →
Parquet). CPU-cheap.
```bash
uv run aapg-grpo-stage reward "${T[@]}" --output "$RUN/ranked.parquet"
```

### 2.4 Throughput (MacBook M5, rough — I cannot benchmark Apple Silicon)
Folding-bound at ~5–8 min/complex on MPS for a ~250 aa complex → with RFD3/MPNN overhead,
**~30–60 complete generations in 8 h**. Optimistic (fast MPS, lean recycling): ~80–100.
Pessimistic (Boltz falls back to CPU, or full ectodomain): ~10–20 with OOM risk. **Time one
complex end-to-end first**, then set Stage-1 `n` to `~ (480 min − RFD3 time) / minutes-per-fold`.

---

## Resume & outputs (both)
- Online run writes one ranked table (`.parquet` or `.jsonl` by extension) at the path you pass.
- Stage-wise writes per-stage manifests; re-running a stage skips completed ids (append + dedupe).
- Per-candidate scratch dirs are auto-removed; non-winning RFD3 backbones are cleaned per round.
- Nothing under `data/weights/` or `experiments/outputs/` is committed to git.
