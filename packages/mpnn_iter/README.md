# aapg-mpnn-iter (module 4)

A modified ProteinMPNN that runs sequence design as an **iterative optimization loop**:
design → score (Rosetta + other sequence evaluators) → re-design under feedback, repeating with
an acceptance/annealing rule until a budget or convergence.

- Upstream edits live in the vendored `proteinmpnn` fork.
- This package holds the loop (`loop.py`), acceptance rules, and the scorer wiring (via
  `aapg.core.scoring` / `aapg.toolkit`).

Config groups: [`configs/model/mpnn_iter.yaml`](../../configs/model), [`configs/scorer/`](../../configs/scorer).

**Status:** placeholder stubs only.
