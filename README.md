# All-atom protein generations is all you need

A monorepo of four related deep-learning sub-projects for all-atom protein design, plus the
shared infrastructure that lets them compose into one end-to-end pipeline.

> This is a deep-learning / software-engineering project. The modules below orchestrate public,
> open-source protein design models (RFDiffusion3, ProteinMPNN, Protenix, etc.).

## The four modules

| # | Package | Namespace | What it does |
|---|---------|-----------|--------------|
| 1 | `packages/grpo_steer`     | `aapg.grpo`       | Online **GRPO** (Group-Relative Policy Optimization) that steers RFDiffusion3 sampling using a reward computed from scored structures. |
| 2 | `packages/rfd3_ellipsoid` | `aapg.ellipsoid`  | **Ellipsoid-constrained RFD3** — ProtComposer-style conditioning of RFD3 on 3D ellipsoid layouts, for on-target and off-target specification. |
| 3 | `packages/validation`     | `aapg.validation` | **Validation pipeline** — backbones → ProteinMPNN → Protenix v2 → OpenDDE → verdict. |
| 4 | `packages/mpnn_iter`      | `aapg.mpnn_iter`  | **Iterative ProteinMPNN** — a modified ProteinMPNN that loops sequence optimization against Rosetta and other sequence scorers. |
| 5 | `packages/score_head`     | `aapg.score_head` | **Learned structure→score head** — predicts complex scores directly from structure (a fast surrogate for MPNN → Protenix v2 → Rosetta); fast path `RFD3 → score head → scores`. |

They chain into one flow:

```
ellipsoid-constrained + GRPO-steered generation  ->  backbones
      ->  iterative ProteinMPNN sequence design
      ->  validation (ProteinMPNN / Protenix v2 / OpenDDE)  ->  verdict
```

## Repository shape

```
packages/
  core/            aapg.core      shared, pure (structure I/O, geometry, metrics, scoring registry)
  toolkit/         aapg.toolkit   adapters to external tools (Rosetta, Protenix, OpenDDE, ESM, ...)
  rfdiffusion3/    rfdiffusion3   vendored fork (edited upstream for ellipsoid conditioning)
  proteinmpnn/     proteinmpnn    vendored fork (edited upstream for the iterative loop)
  rfd3_ellipsoid/  aapg.ellipsoid module 2
  grpo_steer/      aapg.grpo      module 1
  mpnn_iter/       aapg.mpnn_iter module 4
  validation/      aapg.validation module 3
  score_head/      aapg.score_head module 5 (learned structure->score surrogate)
  pipelines/       aapg.pipelines end-to-end orchestration
configs/           global Hydra config tree (consumed by every entrypoint)
containers/        Docker + Apptainer definitions (one per heavy tool)
scripts/           weight download + SLURM submission helpers
docs/              architecture, per-module docs, extensibility guide
```

See [`docs/architecture.md`](docs/architecture.md) for the package dependency graph, and
[`docs/adding_a_scorer.md`](docs/adding_a_scorer.md) for how to extend the system.

## Quickstart

```bash
# 1. Install everything (editable) into a single workspace venv
make setup            # == uv sync --all-packages

# 2. Sanity-check the install
uv run python -c "import aapg.core, aapg.grpo, aapg.ellipsoid, aapg.mpnn_iter, aapg.validation, aapg.pipelines"

# 3. Run the CPU-only test suite
make test

# 4. Inspect the composed end-to-end config (Hydra)
uv run python -m aapg.pipelines.end2end --help
```

For a CPU/mock health check of a clone (no GPU or weights needed), run
[`scripts/smoke_test.sh`](scripts/smoke_test.sh).

### Real (non-mock) run

The steps above exercise the mock adapters only. A real structure-folding run additionally needs:

```bash
# Fetch the open model weights into data/weights/ (Boltz-2, RFD3; verifies vendored ProteinMPNN)
make weights-download        # == bash scripts/weights/download_all.sh
```

- **Weights** land under `data/weights/` (git-ignored) — see
  [`data/weights/README.md`](data/weights/README.md) and the per-model scripts in
  [`scripts/weights/`](scripts/weights/) (each takes `--dry-run`).
- **PyRosetta** (interface scorer) is separately licensed — a non-PyPI wheel from the
  RosettaCommons channel, installed out-of-band (it is *not* pulled by `--extra tools` on
  python ≥ 3.11).
- **Hardware**: a real run needs a CUDA GPU and the heavy model stacks (torch, boltz, foundry/rfd3),
  installed by the weight scripts or the per-tool containers under `containers/`.

## Design principles

- **uv workspace of independently installable packages** — share one lockfile and one config tree,
  install/version each module on its own.
- **`core` is pure, `toolkit` is heavy** — fast CPU imports and CI; subprocess/container-invoking
  tool wrappers are isolated in `toolkit`.
- **Vendored forks keep their upstream import names** (`rfdiffusion3`, `proteinmpnn`) so diffs stay
  readable and re-vendoring is mechanical (see [`VENDORING.md`](VENDORING.md)).
- **Everything is registry-driven** — scorers, rewards and pipeline stages are added by name from
  config, not by editing call sites.

## Status

Skeleton / scaffolding. Package layout, configs, containers, CI, and the scoring registry are in
place; the four models' algorithms are implemented incrementally on top (see
[`CHANGELOG.md`](CHANGELOG.md)).
