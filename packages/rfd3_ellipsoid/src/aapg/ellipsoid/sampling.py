"""Ellipsoid-conditioned sampling API (placeholder).

Thin entry point that drives the forked ``rfdiffusion3`` model with an ellipsoid layout to
produce backbones. The actual conditioning logic/edits live in the vendored fork.
"""

from __future__ import annotations

from typing import Any

from aapg.ellipsoid.layout import EllipsoidLayout


def sample_backbones(layout: EllipsoidLayout, num_samples: int = 1, **kwargs: Any) -> Any:
    """Generate backbones conditioned on ``layout`` (placeholder)."""
    raise NotImplementedError
