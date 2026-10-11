"""Validation pipeline runner (placeholder).

Chains the configured stages (default: ProteinMPNN -> Protenix v2 -> OpenDDE) and reduces their
outputs to a verdict/report for each input backbone.
"""

from __future__ import annotations

from typing import Any

from aapg.validation.stages import Stage


class ValidationPipeline:
    """Runs an ordered list of stages and produces a verdict (placeholder)."""

    def __init__(self, stages: list[Stage] | None = None, **cfg: Any) -> None:
        self.stages = stages or []
        self.cfg = cfg

    def validate(self, backbone: Any) -> Any:
        """Run the pipeline on one backbone and return a verdict/report (placeholder)."""
        raise NotImplementedError
