"""Scorer adapters exposing the score head through the shared registry (placeholder).

``StructureHeadScorer`` implements ``aapg.core.scoring.Scorer`` so the learned head is a drop-in
cascade stage for the GRPO pipeline (resolved by the name ``"structure_head"``). Scoring a
candidate = run the head on its complex structure and return the predicted scores as a
:class:`~aapg.core.scoring.Score`. The head (and torch) load lazily, so importing this module and
registering the scorers never requires them; ``MockStructureHeadScorer`` ("structure_head_mock")
gives deterministic CPU values for tests and dry runs.
"""

from __future__ import annotations

import hashlib
from typing import Any

from aapg.core.scoring import SCORERS, Candidate, Score, Scorer
from aapg.score_head.model import DEFAULT_TARGETS

__all__ = ["StructureHeadScorer", "MockStructureHeadScorer"]


@SCORERS.register("structure_head")
class StructureHeadScorer(Scorer):
    """Predict interface scores for a complex with the learned score head (placeholder)."""

    name = "structure_head"

    def __init__(
        self, checkpoint: str | None = None, targets: tuple[str, ...] | None = None, **cfg: Any
    ) -> None:
        self.checkpoint = checkpoint
        self.targets = tuple(targets) if targets is not None else DEFAULT_TARGETS
        self.cfg = cfg
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Lazily load the trained head (placeholder; raises until the head is implemented)."""
        if self._model is None:
            if self.checkpoint is None:
                raise RuntimeError(
                    "structure_head requires a trained checkpoint; set scorer.checkpoint or use "
                    "'structure_head_mock' for CPU dry runs."
                )
            from aapg.score_head.model import StructureScoreHead

            self._model = StructureScoreHead.load(self.checkpoint, targets=self.targets)
        return self._model

    def score(self, candidate: Candidate) -> Score:
        if candidate.structure_path is None:
            raise RuntimeError("structure_head requires candidate.structure_path")
        from aapg.score_head.inference import predict_scores

        model = self._ensure_model()
        values = predict_scores(
            candidate.structure_path,
            model,
            binder_chains=candidate.binder_chains,
            target_chains=candidate.target_chains,
        )
        return Score(
            values={k: float(v) for k, v in values.items()},
            metadata={"scorer": self.name, "checkpoint": self.checkpoint},
        )


def _derive_unit(seed: str, salt: str) -> float:
    """Deterministic float in ``[0, 1)`` from ``seed`` and ``salt``."""
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# Plausible ranges for the deterministic mock, mirroring the real scorers' outputs.
_MOCK_RANGES: dict[str, tuple[float, float]] = {
    "ipsae": (0.3, 0.9),
    "iptm": (0.4, 0.95),
    "ipae_min": (2.0, 15.0),
    "ipae_mean": (5.0, 25.0),
    "complex_plddt": (60.0, 95.0),
    "interface_dg": (-45.0, -5.0),
    "shape_complementarity": (0.5, 0.78),
    "contact_molecular_surface": (150.0, 600.0),
}


@SCORERS.register("structure_head_mock")
class MockStructureHeadScorer(Scorer):
    """Deterministic CPU stand-in for :class:`StructureHeadScorer` (no model/torch)."""

    name = "structure_head_mock"

    def __init__(self, targets: tuple[str, ...] | None = None, **cfg: Any) -> None:
        self.targets = tuple(targets) if targets is not None else DEFAULT_TARGETS
        self.cfg = cfg

    def score(self, candidate: Candidate) -> Score:
        seed = candidate.binder_sequence or candidate.name or candidate.structure_path or ""
        values = {
            t: round(
                _MOCK_RANGES.get(t, (0.0, 1.0))[0]
                + _derive_unit(seed, t)
                * (_MOCK_RANGES.get(t, (0.0, 1.0))[1] - _MOCK_RANGES.get(t, (0.0, 1.0))[0]),
                4,
            )
            for t in self.targets
        }
        return Score(values=values, metadata={"scorer": self.name, "mock": True})
