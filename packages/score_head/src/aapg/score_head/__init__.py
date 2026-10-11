"""aapg.score_head — module 5 (placeholder): learned structure -> score head.

A standalone, trainable neural *head* that takes a complex structure (target + binder) produced
by any structure source -- RFDiffusion3 directly, or a folder such as Protenix v2 / BoltzGen /
Boltz2 -- and predicts interface/quality scores in a single forward pass. It is a fast learned
**surrogate** for the expensive MPNN -> Protenix -> Rosetta scoring chain, enabling a fast path::

    RFD3 (structure)  ->  score_head  ->  scores      # bypasses ProteinMPNN and Protenix v2

Because :class:`~aapg.score_head.scorer.StructureHeadScorer` implements the shared
``aapg.core.scoring.Scorer`` interface and registers itself by name (``"structure_head"``), the
head also drops into the GRPO scoring cascade as a cheap stage, or as a full replacement for the
costly scorers.

Status: PLACEHOLDER. Importable stubs only -- no real model, features or training. torch is an
optional, lazily-imported extra (see ``pyproject.toml``); a deterministic CPU mock scorer
(``"structure_head_mock"``) lets the package run in tests/CI without it.
"""

from __future__ import annotations

# Registration side effect: importing the scorer module registers "structure_head" and
# "structure_head_mock" into aapg.core.scoring.SCORERS. The head/torch load lazily, so this
# import stays light.
from aapg.score_head import scorer  # noqa: F401

__version__ = "0.0.0"

__all__ = ["scorer", "__version__"]
