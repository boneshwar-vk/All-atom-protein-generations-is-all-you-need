"""Geometry: SE(3) transforms and ellipsoid math (shared with module 2) (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Ellipsoid:
    """A 3D ellipsoid (center, semi-axes, orientation) used for RFD3 conditioning (placeholder)."""

    # center: (3,), radii: (3,), rotation: (3, 3) — concrete types land with the implementation.
    ...
