# configs/ — global Hydra config tree

A single config tree consumed by every entrypoint in `aapg.pipelines`. `config.yaml` is the root
defaults list; each subdirectory is a **config group** with swappable options. `experiment/`
presets override multiple groups to define a named run.

| Group | Purpose |
|-------|---------|
| `paths/`     | data / weights / output roots (per machine or cluster) |
| `model/`     | rfd3, rfd3_ellipsoid, mpnn, mpnn_iter |
| `sampler/`   | diffusion sampler + guidance |
| `ellipsoid/` | on/off-target ellipsoid layouts (module 2) |
| `steering/`  | GRPO hyperparameters (module 1) |
| `reward/`    | reward composition (weighted scorer bundles) |
| `scorer/`    | per-scorer configs (rosetta, protenix, opendde, ...) |
| `pipeline/`  | validation pipeline stage wiring (module 3) |
| `trainer/`   | optimizer / loop / checkpointing / experiment tracking |
| `launcher/`  | local / slurm / debug |
| `experiment/`| named end-to-end presets |

Example (once implemented):

```bash
uv run python -m aapg.pipelines.end2end +experiment=end2end launcher=slurm seed=1
```

**Status:** placeholder values — structure is real, numbers are illustrative.
