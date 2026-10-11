"""Tests for pure-numpy Kabsch self-consistency RMSD."""

from __future__ import annotations

import numpy as np

from aapg.grpo.scrmsd import self_consistency_rmsd


def _coords(n: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.normal(size=(n, 3))


def test_identical_coords_zero():
    coords = _coords()
    assert self_consistency_rmsd(coords, coords.copy()) < 1e-9


def test_pure_rotation_is_zero():
    coords = _coords()
    # A proper rotation about the z axis.
    theta = 0.7
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = coords @ rot.T
    assert self_consistency_rmsd(coords, rotated) < 1e-9


def test_translation_is_zero():
    coords = _coords()
    translated = coords + np.array([5.0, -3.0, 2.0])
    assert self_consistency_rmsd(coords, translated) < 1e-9


def test_rotation_plus_translation_is_zero():
    coords = _coords()
    theta = -1.2
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    )
    moved = coords @ rot.T + np.array([1.0, 2.0, 3.0])
    assert self_consistency_rmsd(coords, moved) < 1e-9


def test_real_deviation_is_positive():
    coords = _coords()
    perturbed = coords.copy()
    perturbed[0] += np.array([3.0, 0.0, 0.0])
    assert self_consistency_rmsd(coords, perturbed) > 0.1


def test_shape_mismatch_raises():
    try:
        self_consistency_rmsd(_coords(10), _coords(11))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched shapes")
