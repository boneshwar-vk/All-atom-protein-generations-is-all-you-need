"""Iterative sequence-optimization loop (placeholder).

Sketches the design->score->redesign loop that optimizes sequences for a fixed backbone using
the forked ProteinMPNN, scored by Rosetta and other sequence-based evaluators resolved by name
from config.
"""

from __future__ import annotations

from typing import Any


class IterativeDesigner:
    """Iteratively optimizes sequences for a backbone (placeholder)."""

    def __init__(self, scorers: list[Any] | None = None, **cfg: Any) -> None:
        self.scorers = scorers or []
        self.cfg = cfg

    def optimize(self, backbone: Any) -> Any:
        """Return optimized sequence(s) for ``backbone`` (placeholder)."""
        raise NotImplementedError
