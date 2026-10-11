# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0 and unversioned.

## [Unreleased]

### Added
- Initial monorepo scaffold: uv workspace, eight packages, global Hydra config tree,
  container definitions (Docker + Apptainer), CPU-only CI, and developer tooling.
- `aapg.core` scoring registry with `Scorer` / `Reward` abstract base classes and a generic
  name->factory `Registry` (the extensibility backbone).
- Import-safe stubs and placeholder tests for all four modules and the two vendored-fork packages.

### Not yet implemented
- The four modules' algorithms (GRPO steering, ellipsoid conditioning, validation pipeline,
  iterative ProteinMPNN). Each lands on top of this skeleton, one package at a time.
- Vendored upstream source for `rfdiffusion3` and `proteinmpnn` (placeholder packages only; see
  `VENDORING.md` for the intended procedure).
