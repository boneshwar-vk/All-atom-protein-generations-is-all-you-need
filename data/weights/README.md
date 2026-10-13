# data/weights/

Model weights live here, one subdirectory per model. **Nothing in this tree is committed to
git** (it is git-ignored) — weights are large and some are license-restricted, so they are
*fetched/placed on setup*, not version-controlled.

Populate it with:

```bash
make weights            # print the plan (python scripts/download_weights.py)
make weights-download   # fetch the open weights into the exact dirs (scripts/weights/download_all.sh)
make weights-download-all  # also the optional extras (Protenix v1, ESM-2)
```

Per-model scripts live in `scripts/weights/` (each takes `--dry-run`): `boltz2.sh`, `rfd3.sh`,
`proteinmpnn.sh`, `protenix_v1.sh`, `protenix_v2.sh`, `esm2.sh`, and the `download_all.sh`
orchestrator. They place each model in the exact directory the configs read.

| Subdir | Model | License | How it gets here |
|--------|-------|---------|------------------|
| `boltz2/`       | Boltz-2 | MIT (open) | `scripts/weights/boltz2.sh` — HF snapshot `boltz-community/boltz-2` (~6 GB); also used as the `boltz predict --cache` dir |
| `proteinmpnn/`  | ProteinMPNN | MIT (open) | **vendored in the repo** (`packages/proteinmpnn/.../vendor`, 8× `v_48_*.pt`); `proteinmpnn.sh` verifies + symlinks here |
| `rfd3/`         | RFdiffusion3 | BSD-3 (open) | `scripts/weights/rfd3.sh` — `pip install rc-foundry[all]` + `foundry install rfd3 --checkpoint-dir` here |
| `protenix_v1/`  | Protenix v1 | Apache-2.0 (open) | `scripts/weights/protenix_v1.sh` — `pip install protenix`; checkpoint auto-downloads on first `protenix pred` |
| `protenix_v2/`  | **Protenix v2** | **PROPRIETARY** | **licensed users only** — `scripts/weights/protenix_v2.sh` prepares the dir; download yourself and place here |
| `esm2/`         | ESM-2 | MIT (open) | `scripts/weights/esm2.sh` — huggingface `facebook/esm2_*` (optional) |

> **PyRosetta** (the interface scorer) is a separately-licensed *package*, not weights — install it
> via the `aapg-toolkit` `tools` extra on a box with a PyRosetta wheel. The mock pipeline
> (`uv run aapg-grpo steering.mock=true ...`) needs none of these and runs on CPU.

## Protenix v2 (licensed)

Protenix v2 weights are proprietary and **must not be committed or redistributed**. If you are
licensed to use them, download them and drop them in `data/weights/protenix_v2/`. The default
`structure_auto` scorer detects them there and switches from Boltz-2 to Protenix v2 automatically;
otherwise the pipeline stays on open Boltz-2.
