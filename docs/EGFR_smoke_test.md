# EGFR binder design — verification & real-run checklist

This is an ordered checklist for verifying that this repository works, and then running a **real**
GRPO mini-binder design against the **EGFR** receptor using the `grpo_egfr` experiment preset
(`configs/experiment/grpo_egfr.yaml`).

It has two tiers:

- **Tier A — CPU / mock smoke tests.** Anyone can run these *now*, on any machine (laptop
  included). They prove the repo *wiring* is correct — config composition, registries, adapters,
  the selection pipeline, and the ranked-table output — using deterministic mock adapters. **No
  GPU, no model weights, no PyRosetta.**
- **Tier B — the real EGFR run.** Requires a CUDA GPU machine, downloaded weights, PyRosetta, and
  a real EGFR structure. This is the actual structure-folding + interface-scoring pipeline.

> **Hard reality check.** The real pipeline (Boltz-2 / Protenix for folding, RFdiffusion3 for
> backbone sampling, PyRosetta for the interface scorer) is **CUDA-oriented**, so a CUDA GPU is the
> only supported, full-speed way to run it (Tier B below). On Apple Silicon (MPS) a reduced,
> **stage-wise** run may be possible but is unverified and much slower — RFD3 in particular may not
> run on MPS at all — see `docs/RUNNING.md` Part 2 for that experimental path and its caveats. Tier
> A (this doc) is the CPU/mock health check and runs on any laptop.

---

## Tier A — CPU / mock smoke tests (run these first, anywhere)

Everything here runs through the mock adapters, so it is fast and needs no heavy dependencies.

### A.0 — One command

```bash
scripts/smoke_test.sh            # full suite (incl. lint/type): 14 checks
scripts/smoke_test.sh --quick    # skip ruff/mypy: 11 checks  -> expect "11 passed, 0 failed"
scripts/smoke_test.sh --egfr-only  # only the EGFR mock pipeline check
```

A healthy clone passes every check. `--quick` reports **11 passed, 0 failed**; the full run adds
the three lint/type checks (14 total). The suite prints `PASS`/`FAIL` per check, continues on
failure, and exits non-zero if anything failed, so one run gives the full picture.

### A.1 — What each check does, and the expected result

The script runs the checks below in order. (`--quick` skips checks 7–9; `--egfr-only` runs only
check 13.)

| # | Check | What it proves | Expected result |
|---|-------|----------------|-----------------|
| 1 | `env: uv available` | `uv` is on `PATH` | PASS |
| 2 | `env: python importable` | the project's Python runs | prints the Python version; PASS |
| 3 | `uv sync --all-packages` | the CPU-only workspace installs with **no** heavy deps (no torch/boltz/pyrosetta) | PASS |
| 4 | `import core + registries …` | importing the packages registers the expected scorers and rewards | lists scorers incl. `boltz2(_mock)`, `protenix_v2(_mock)`, `rosetta_interface(_mock)`, `structure_auto(_mock)`, `structure_head(_mock)` and reward `ipsae_gated`; PASS |
| 5 | `import adapters (mock …)` | the mock RFD3 sampler and ProteinMPNN designer import | prints `rfd3 + proteinmpnn adapters import OK`; PASS |
| 6 | `pytest (full suite, CPU-only)` | the whole unit-test suite is green (includes the advantage / non-finite-reward tests) | all tests pass; PASS |
| 7 | `ruff check` | lint is clean | `All checks passed!`; PASS *(skipped with `--quick`)* |
| 8 | `ruff format --check` | formatting is clean | PASS *(skipped with `--quick`)* |
| 9 | `mypy (first-party src)` | types are clean across first-party packages | `Success`; PASS *(skipped with `--quick`)* |
| 10 | `ProteinMPNN vendored weights present` | the in-repo ProteinMPNN weights are present (no download) | PASS |
| 11 | `weight scripts dry-run (no download)` | the weight-download scripts are sound, with **no network access** | PASS |
| 12 | `mock pipeline (default target)` | the selection pipeline runs end-to-end on the default target via mocks | PASS |
| 13 | `EGFR mock pipeline (ranked table written)` | an EGFR-shaped selection run (binder_length 80, chains, domain-III hotspots) runs via mocks and writes a ranked table to `experiments/outputs/egfr_smoke/ranked.jsonl` | PASS |
| 14 | `EGFR ranked table has all 11 metric columns + rows` | the written table has 32 candidate rows, all 11 metric columns, and at least one selected candidate | prints row count, columns OK, top reward; PASS |

The EGFR mock checks (13–14) use EGFR-shaped inputs (target chain `A`, binder chain `B`,
hotspots `A465,A467,A468`, `binder_length=80`) but, being mocks, exercise **config + plumbing,
not biology** — the mock ignores the (absent) structure.

### A.2 — Verify the `grpo_egfr` preset composes and ranks (mock)

This confirms the new experiment preset specifically composes under Hydra and writes a ranked
table, still entirely on CPU:

```bash
uv run --extra full aapg-grpo +experiment=grpo_egfr steering.mock=true \
  steering.n_backbones=6 steering.seqs_per_backbone=3 \
  steering.group_size=6 steering.top_k=4 \
  +steering.output_path=/tmp/egfr_mock/ranked.parquet
```

**Expected:** completes without error and writes `/tmp/egfr_mock/ranked.parquet` (an 18-row ranked
table — 6 backbones × 3 sequences — with `reward`, `advantage`, `selected`, and the metric
columns). Note the **leading `+`** on `+experiment=grpo_egfr` (the `experiment` group is not in the
root defaults list) and on `+steering.output_path=…` (a key not already present in the group).
`--extra full` provides Hydra + polars; mock mode needs nothing heavier.

---

## Tier B — the real EGFR run (CUDA GPU machine; NOT a Mac)

Do Tier A first and get it fully green. Tier B is the actual pipeline: RFdiffusion3 samples
backbones, ProteinMPNN designs sequences, Boltz-2 (or licensed Protenix v2) folds the complex,
and PyRosetta scores the interface.

### B.0 — Prerequisites

1. **A CUDA GPU.** Boltz-2 / Protenix folding and RFdiffusion3 sampling require an NVIDIA GPU with
   CUDA. A workstation or cluster node with a recent GPU and ample VRAM. **Not a Mac; not CPU.**
2. **Disk.** Boltz-2 weights are ~6 GB; add RFD3 checkpoints and per-run outputs. Have tens of GB
   free. (See the low-disk note in B.5 if space is tight.)
3. **The repo installed with the heavy/tools deps** (the CPU `uv sync` from Tier A does *not*
   pull these — they are behind a never-true marker so the lock stays light):

   ```bash
   uv sync --all-packages --extra tools   # biopython, gemmi, polars, onnxruntime
   # Heavy stacks installed out-of-band at generation time (not in uv.lock):
   pip install boltz                       # default structure backend (MIT; weights auto-download)
   # pip install protenix                  # OPTIONAL licensed v2 alternative (needs licensed weights)
   ```
4. **PyRosetta** — the interface scorer. It is a **separately-licensed package, not weights**, and
   is **not** installed by `uv sync` (on Python ≥ 3.11 its marker is never satisfied). Install it
   out-of-band from the licensed **RosettaCommons** channel (a non-PyPI wheel; requires a
   RosettaCommons license). Without it, the `rosetta_interface` stage cannot run.

### B.1 — Download the open model weights

```bash
make weights-download        # Boltz-2 (~6 GB) + RFdiffusion3 checkpoints; verifies ProteinMPNN
# optional extras:
make weights-download-all    # also Protenix v1 (alt open backend) + ESM-2
make weights                 # just print the weights plan (what goes where), downloads nothing
```

Weights land under `${AAPG_ROOT:-<repo>}/data/weights/<model>/` (git-ignored). Protenix **v2** is
proprietary/licensed — the script prints placement instructions only; drop licensed v2 weights in
`data/weights/protenix_v2/` to have `structure_auto` auto-select it over Boltz-2.

### B.2 — Obtain the EGFR target structure

The preset ships a **placeholder** target — you must supply a real EGFR extracellular-domain
structure and renumber/pick residues before any real run.

1. Download an EGFR ectodomain structure from the RCSB PDB, e.g.:
   - **1IVO** — EGFR extracellular region bound to EGF (receptor chain `A`, EGF chain `C`), or
   - **3NJP** — an EGFR domain-III complex.
2. Strip to the receptor chain(s) you want to design against, and note the **chain IDs** as they
   appear in your cleaned file (1IVO's receptor is chain `A`).
3. Pick the **epitope hotspot residues** on EGFR **domain III** (the EGF-competitive /
   cetuximab-like face, roughly residues 310–480 in mature EGFR numbering). Use chain-prefixed
   residue IDs that **actually exist and are surface-exposed** in your file, e.g. `A465,A467,A468`.
   Confirm each residue in a viewer — these must be real residues in the file you pass.

### B.3 — Edit the preset (or override on the CLI)

`configs/experiment/grpo_egfr.yaml` marks the target block as a placeholder: `structure_path` is
`???` (Hydra mandatory-missing), so a real run **fails loudly** until you set it. Either edit the
file in place or override on the CLI. These keys already exist in the `target` block (so they take
a **plain** override, no `+`); supplying the mandatory-missing `structure_path` is also a plain
assignment:

- `steering.target.structure_path` → absolute path to your cleaned EGFR PDB/mmCIF
- `steering.target.chains` → the EGFR receptor chain IDs in that file
- `steering.target.hotspots` → the domain-III residues you chose
- `steering.target.binder_chains` / `binder_length` → binder chain ID and length (~80)

### B.4 — Run the real GRPO selection

Real mode is the default (`steering.mock=false`). Example:

```bash
uv run --extra full aapg-grpo +experiment=grpo_egfr \
  steering.target.structure_path=/abs/path/to/egfr_domainIII.pdb \
  'steering.target.chains=[A]' \
  'steering.target.hotspots=[A465,A467,A468,A440,A441]' \
  'steering.target.binder_chains=[B]' \
  steering.target.binder_length=80 \
  steering.n_backbones=256 \
  steering.seqs_per_backbone=8 \
  steering.group_size=64 \
  steering.top_k=50 \
  +steering.output_path=experiments/outputs/grpo_egfr_demo/ranked.parquet
```

Notes:
- The cascade is `[structure_auto, rosetta_interface]`: `structure_auto` uses **Boltz-2** by
  default, or **Protenix v2** automatically if licensed v2 weights are in `data/weights/protenix_v2/`.
  To force a backend, set `scorer=boltz2` or `scorer=protenix_v2` (or override
  `steering.cascade`).
- Start modest (`n_backbones` in the hundreds) to confirm the GPU path end-to-end, then scale up.
  The production default in `configs/steering/grpo.yaml` is `n_backbones=4096`.
- Interim hard gates are applied after the (cheap) structure stage, so the expensive Rosetta
  stage runs **only on survivors** (early exit). Relax the soft `iptm` bound toward 0.5 for hard
  targets (see `configs/reward/ipsae_gated.yaml`).

### B.5 — Where outputs land, and a step-by-step / low-disk note

- **Ranked table:** whatever you pass as `+steering.output_path` (Parquet if the path has a
  `.parquet`/no suffix and polars is available, else JSONL). The example above writes
  `experiments/outputs/grpo_egfr_demo/ranked.parquet`. The table has lead columns
  `name, binder_sequence, round, selected, reward, advantage, backbone_path` plus the 11 metric
  columns, one row per scored candidate, ranked by group-relative advantage.
- **Run output dir:** `paths.output_dir` defaults to
  `${paths.root}/experiments/outputs/${run_name}` (here `run_name=grpo_egfr_demo`). Override
  `paths.root` / `run_name` per machine.

**Step-by-step / low-disk.** If disk is tight, you don't have to run the whole cascade at once:

- Run with a **small `n_backbones`** first, inspect the written ranked table and the structure
  outputs, then scale up once the path is proven.
- The cascade **early-exits**: candidates failing an interim hard gate after the cheap structure
  stage never reach the expensive Rosetta stage, so disk and compute are spent only on survivors.
- Each candidate is scored inside an **auto-cleaned per-candidate scratch directory**
  (a `tempfile.TemporaryDirectory(prefix="aapg_cand_")`): the structure predictor writes its CIF
  there and the dir is deleted when the candidate finishes, so transient per-candidate files do
  not accumulate across the run. Only the final ranked table (and any outputs you explicitly keep)
  persist.

---

## Summary

- **Laptop / CI:** `scripts/smoke_test.sh` (Tier A) — all mock, no GPU. `--quick` → 11/11.
- **Real EGFR design:** Tier B — CUDA GPU (not Mac/CPU), `make weights-download`, PyRosetta from
  the licensed channel, a real EGFR structure, then `aapg-grpo +experiment=grpo_egfr …` with the
  placeholder target replaced.
