"""Abstract base classes for scoring.

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


@dataclass
class Candidate:
    """One complex to score: the shared input contract for every :class:`Scorer`.

    A ``Candidate`` is produced by the GRPO rollout, which samples a backbone, threads a
    sequence onto the binder chain(s), and poses it against a fixed target. It bundles
    everything a scorer needs to evaluate that proposal and is consumed unchanged by each
    registered :class:`Scorer`, so scorers can be composed without agreeing on anything
    beyond these fields.

    Attributes:
        structure_path: Path to the complex structure (PDB or mmCIF), or ``None`` when the
            candidate carries only a sequence.
        binder_chains: Chain IDs of the designed binder.
        target_chains: Chain IDs of the fixed target the binder is scored against.
        binder_sequence: The binder sequence, when available separately from the structure.
        name: Optional human-readable identifier for logging and provenance.
        metadata: Free-form extra context (e.g. rollout step, sampling temperature); carried
            through scoring but never required by it.
    """

    structure_path: str | None = None
    binder_chains: tuple[str, ...] = ("A",)
    target_chains: tuple[str, ...] = ("B",)
    binder_sequence: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def combine_scores(scores: list[Score]) -> Score:
    """Merge several :class:`Score` objects into one.

    ``values`` are unioned and ``metadata`` is shallow-merged, both in list order, so later
    scores win on a key clash. An empty list yields an empty :class:`Score`.
    """
    values: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    for score in scores:
        values.update(score.values)
        metadata.update(score.metadata)
    return Score(values=values, metadata=metadata)


class Scorer(ABC):
    """Scores a candidate (structure, complex, or sequence)."""

    name: str = "scorer"

    @abstractmethod
    def score(self, candidate: Any) -> Score:
        """Return a :class:`Score` for ``candidate`` (a :class:`Candidate`)."""
        raise NotImplementedError


class Reward(ABC):
    """Reduces one or more :class:`Score` objects to a single scalar reward."""

    name: str = "reward"

    @abstractmethod
    def __call__(self, scores: list[Score]) -> float:
        """Return the scalar reward (placeholder)."""
        raise NotImplementedError
