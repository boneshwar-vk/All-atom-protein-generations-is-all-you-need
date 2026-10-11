# Contributing

## Environment

```bash
make setup     # uv sync --all-packages into .venv
make hooks     # install pre-commit
```

Everything runs through `uv`; do not `pip install` into a global environment.

## Layout rules

- **One package per module** under `packages/`. A new module = a new workspace member (its own
  `pyproject.toml` + `src/` + `tests/`) plus a matching group under `configs/`. Nothing else moves.
- **`aapg.*` namespace packages** use PEP 420 — do **not** add `src/aapg/__init__.py` to any
  package. Only the leaf subpackage (e.g. `src/aapg/core/__init__.py`) gets an `__init__.py`.
- **Keep `core` pure.** `core` may depend only on numpy and light structure libs. Anything that
  shells out to an external binary/container (Rosetta, Protenix, OpenDDE, ...) belongs in
  `toolkit`, behind the `core.scoring` interfaces.
- **Vendored forks** (`rfdiffusion3`, `proteinmpnn`) keep their original import names and their own
  upstream `LICENSE`. Record every re-vendor in [`VENDORING.md`](VENDORING.md).

## Adding a scorer / reward / stage

Implement the relevant ABC from `aapg.core.scoring` and register it with the decorator — see
[`docs/adding_a_scorer.md`](docs/adding_a_scorer.md). Do not hard-wire it into a call site; consumers
resolve it by name from config.

## Before pushing

```bash
make check     # ruff + mypy + pytest (same as CI)
```

- Tests must be **CPU-only and tool-free** (mock or skip when a heavy tool/container is absent) so
  CI stays fast. GPU/integration tests are opt-in markers.
- Add/adjust configs under `configs/` for any new knob; keep defaults runnable.
