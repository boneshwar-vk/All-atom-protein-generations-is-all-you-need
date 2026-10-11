# aapg-score-head (module 5)

A standalone, trainable **structure → score head**: it takes a complex structure (target +
binder) from any source — RFDiffusion3 directly, or a folder such as Protenix v2 / BoltzGen /
Boltz2 — and predicts interface/quality scores in one forward pass, as a fast learned surrogate
for the expensive `ProteinMPNN → Protenix v2 → Rosetta` chain.

```
RFD3 (structure)  ->  score_head  ->  scores        # fast path: no ProteinMPNN, no Protenix v2
```

## Layout

- `model.py` — `StructureScoreHead` (the network; predicts a dict of scores). torch is lazy.
- `features.py` — complex structure → model inputs (residue graph / tensors).
- `train.py` — training by **distillation**: labels are the expensive pipeline's scores; the head
  learns to predict them from structure alone.
- `inference.py` — `predict_scores(structure_path, model, ...) → {metric: value}`.
- `scorer.py` — `StructureHeadScorer` (registered `"structure_head"`) + `MockStructureHeadScorer`
  (`"structure_head_mock"`). Implements `aapg.core.scoring.Scorer`, so the head drops straight
  into the GRPO cascade by name, or replaces the costly scorers entirely.

## How it plugs in

It is **not** wired into the default GRPO cascade (`configs/steering/grpo.yaml` keeps
`[protenix_v2, rosetta_interface]`). To use the fast path, set the cascade to `[structure_head]`
(config `configs/scorer/structure_head.yaml`, model stub `configs/model/score_head.yaml`). As a
differentiable reward model it could later give a cheap dense signal for the GRPO `rl` mode.

## Status

**Placeholder** — importable stubs only, no real model/features/training. torch is an optional,
lazily-imported extra (`pip install -e packages/score_head[train]`), pinned behind a never-true
marker so it stays out of the workspace lock; a deterministic CPU mock scorer runs without it.
