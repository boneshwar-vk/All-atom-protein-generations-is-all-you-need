"""Auto structure-scorer: Boltz-2 by default, Protenix v2 when licensed weights are present.

``AutoStructureScorer`` ("structure_auto") is the project's default interface backend. It picks:

- **Protenix v2** if ``protenix_weights_dir`` exists and is non-empty (i.e. a licensed user has
  placed the proprietary v2 weights there), or
- **Boltz-2** otherwise (MIT, open weights; the always-available default).

This keeps the pipeline fully open out of the box while automatically upgrading to Protenix v2
for users who are licensed and have dropped the weights in place. Both delegate to the concrete
scorers (``boltz2`` / ``protenix_v2``) and return the same metric keys, so the rest of the
pipeline is unchanged. A mock variant ("structure_auto_mock") delegates to the Boltz mock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aapg.core.scoring import SCORERS, Candidate, Score, Scorer

__all__ = ["AutoStructureScorer", "MockAutoStructureScorer"]


def _weights_present(path: str | None) -> bool:
    """True if ``path`` is a directory containing at least one entry."""
    if not path:
        return False
    p = Path(path)
    return p.is_dir() and any(p.iterdir())


@SCORERS.register("structure_auto")
class AutoStructureScorer(Scorer):
    """Delegate to Protenix v2 when licensed weights are in place, else Boltz-2 (default)."""

    name = "structure_auto"

    def __init__(
        self,
        protenix_weights_dir: str | None = None,
        boltz_weights_dir: str | None = None,
        protenix_cfg: dict[str, Any] | None = None,
        boltz_cfg: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.protenix_weights_dir = protenix_weights_dir
        self.boltz_weights_dir = boltz_weights_dir
        self.protenix_cfg = dict(protenix_cfg or {})
        self.boltz_cfg = dict(boltz_cfg or {})
        self.extra = kwargs
        self._backend: Scorer | None = None

    def _select(self) -> Scorer:
        if self._backend is None:
            if _weights_present(self.protenix_weights_dir):
                self._backend = SCORERS.create(
                    "protenix_v2", weights_dir=self.protenix_weights_dir, **self.protenix_cfg
                )
            else:
                self._backend = SCORERS.create(
                    "boltz2", weights_dir=self.boltz_weights_dir, **self.boltz_cfg
                )
        return self._backend

    def score(self, candidate: Candidate) -> Score:
        backend = self._select()
        result = backend.score(candidate)
        result.metadata.setdefault("auto_backend", backend.name)
        return result


@SCORERS.register("structure_auto_mock")
class MockAutoStructureScorer(Scorer):
    """Deterministic CPU stand-in for :class:`AutoStructureScorer` (delegates to the Boltz mock)."""

    name = "structure_auto_mock"

    def __init__(self, **kwargs: Any) -> None:
        self.extra = kwargs
        self._backend = SCORERS.create("boltz2_mock")

    def score(self, candidate: Candidate) -> Score:
        result = self._backend.score(candidate)
        result.metadata.setdefault("auto_backend", "boltz2_mock")
        return result
