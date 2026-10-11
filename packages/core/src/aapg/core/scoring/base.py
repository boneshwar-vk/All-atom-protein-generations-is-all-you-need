"""Abstract base classes for scoring (placeholder).

A ``Scorer`` maps a protein structure or sequence to one or more numeric scores. A ``Reward``
combines scorer outputs into a single scalar used for optimization (e.g. GRPO advantage, MPNN
acceptance). Concrete implementations live in ``aapg.toolkit`` or in the individual modules and
register themselves (see ``registries``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Score:
    """A named set of numeric scores plus optional metadata (placeholder)."""

    values: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    """Scores a candidate (structure, complex, or sequence)."""

    name: str = "scorer"

    @abstractmethod
    def score(self, candidate: Any) -> Score:
        """Return a :class:`Score` for ``candidate`` (placeholder)."""
        raise NotImplementedError


class Reward(ABC):
    """Reduces one or more :class:`Score` objects to a single scalar reward."""

    name: str = "reward"

    @abstractmethod
    def __call__(self, scores: list[Score]) -> float:
        """Return the scalar reward (placeholder)."""
        raise NotImplementedError
