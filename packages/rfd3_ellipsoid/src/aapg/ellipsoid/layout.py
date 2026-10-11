"""Ellipsoid layouts for on-target / off-target conditioning (placeholder).

An :class:`EllipsoidLayout` is the set of ellipsoids (each tagged on-target or off-target) that
condition a generation. Featurizers turn a layout into inputs for the forked ``rfdiffusion3``
model; the sampling API consumes it to produce backbones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Target(StrEnum):
    ON = "on_target"
    OFF = "off_target"


@dataclass
class EllipsoidSpec:
    """One ellipsoid plus its on/off-target role (placeholder)."""

    target: Target = Target.ON
    # geometry (center/radii/orientation) references aapg.core.geometry.Ellipsoid


@dataclass
class EllipsoidLayout:
    """A collection of ellipsoids that condition a single generation (placeholder)."""

    ellipsoids: list[EllipsoidSpec] = field(default_factory=list)
