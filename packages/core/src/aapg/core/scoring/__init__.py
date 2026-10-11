"""Scoring interfaces shared across GRPO reward, validation, and iterative MPNN.

Placeholder: defines the abstract surface and the registries, no concrete scorers.
"""

from aapg.core.scoring.base import Reward, Score, Scorer
from aapg.core.scoring.registries import REWARDS, SCORERS

__all__ = ["Scorer", "Reward", "Score", "SCORERS", "REWARDS"]
