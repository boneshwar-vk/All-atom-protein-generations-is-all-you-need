# aapg-core

Shared, **pure** building blocks used by every other package. Depends only on numpy (plus light
structure libs later) so it imports fast and tests on CPU.

Subpackages:

- `io/` — PDB/mmCIF read-write, parsing, chain/complex handling.
- `repr/` — atom37/atom14 representations, residue types, frames/rigids, featurization.
- `geometry/` — SE(3), transforms, and ellipsoid math (shared with module 2).
- `metrics/` — RMSD, TM-score, clash, SASA, interface metrics (pure/light).
- `scoring/` — `Scorer` / `Reward` abstract base classes + the registry wiring.
- `registry.py` — generic name->factory registry used across the workspace.
- `utils/` — seeding/determinism, logging, config helpers.

Anything that shells out to an external tool/container lives in `aapg-toolkit`, not here.

**Status:** placeholder stubs only.
