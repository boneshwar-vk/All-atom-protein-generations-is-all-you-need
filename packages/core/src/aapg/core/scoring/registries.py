"""The shared scorer/reward registries (placeholder).

Import these and use the ``register`` decorator to add a component by name::

    from aapg.core.scoring import SCORERS, Scorer

    @SCORERS.register("my_scorer")
    class MyScorer(Scorer):
        ...

Consumers (GRPO reward, validation, iterative MPNN) resolve components by name from config.
"""

from __future__ import annotations

from aapg.core.registry import Registry
from aapg.core.scoring.base import Reward, Scorer

SCORERS: Registry[Scorer] = Registry("scorer")
REWARDS: Registry[Reward] = Registry("reward")

__all__ = ["SCORERS", "REWARDS"]
