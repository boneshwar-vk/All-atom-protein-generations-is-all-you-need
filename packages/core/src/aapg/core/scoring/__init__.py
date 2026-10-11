"""Scoring interfaces shared across GRPO reward, validation, and iterative MPNN.

Defines the abstract surface (``Scorer``/``Reward``), the shared :class:`Candidate` input
contract and the :func:`combine_scores` helper, plus the registries. No concrete scorers.
"""

from aapg.core.scoring.base import Candidate, Reward, Score, Scorer, combine_scores
from aapg.core.scoring.registries import REWARDS, SCORERS

__all__ = [
    "Scorer",
    "Reward",
    "Score",
    "Candidate",
    "combine_scores",
    "SCORERS",
    "REWARDS",
]
