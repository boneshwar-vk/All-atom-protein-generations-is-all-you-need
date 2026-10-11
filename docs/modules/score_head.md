# Module 5 — Learned structure→score head

A standalone trainable head that predicts interface/quality scores directly from a complex
structure (target + binder), as a fast surrogate for the `ProteinMPNN → Protenix v2 → Rosetta`
chain. Enables the fast path **RFD3 → score head → scores**, and (via its `Scorer` adapter) drops
into the GRPO cascade by name.

- Package: `packages/score_head` (`aapg.score_head`).
- Inputs: a complex structure from RFD3 directly, or any folder (Protenix v2 / BoltzGen / Boltz2).
- Trained by distillation from the expensive pipeline's scores (labels).
- Registers scorers `structure_head` (real; needs a checkpoint) and `structure_head_mock` (CPU).
- Configs: `configs/scorer/structure_head.yaml`, `configs/model/score_head.yaml`.

**Status:** placeholder (importable stubs; no real model/training). torch is an optional, lazy
extra kept out of the default lock.
