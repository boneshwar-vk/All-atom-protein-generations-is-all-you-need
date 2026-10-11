"""Tests for the ipSAE-gated reward."""

from __future__ import annotations

import math

from aapg.core.scoring import Score
from aapg.grpo.reward import IpSAEGatedReward


def _passing_values() -> dict[str, float]:
    # All hard gates satisfied, all soft bounds satisfied.
    return {
        "ipsae": 0.8,
        "has_clash": 0.0,
        "ipae_mean": 6.0,
        "complex_plddt": 90.0,
        "scrmsd": 1.0,
        "iptm": 0.7,
        "ipae_min": 3.0,
        "interface_dg": -15.0,
        "shape_complementarity": 0.6,
        "contact_molecular_surface": 500.0,
        "sap_normalized": 30.0,
        "isoelectric_point": 7.0,
    }


def _reward() -> IpSAEGatedReward:
    return IpSAEGatedReward(
        primary="ipsae",
        penalty_weight=1.0,
        hard={
            "has_clash": {"max": 0},
            "ipsae": {"min": 0.6},
            "ipae_mean": {"max": 10},
            "complex_plddt": {"min": 80},
            "scrmsd": {"max": 2.0},
        },
        soft={
            "iptm": {"min": 0.6},
            "ipae_min": {"max": 5},
            "interface_dg": {"max": 0},
            "shape_complementarity": {"min": 0.55},
            "contact_molecular_surface": {"min": 400},
            "sap_normalized": {"max": 50},
            "isoelectric_point": {"range": [6.0, 9.0]},
        },
    )


def test_passing_returns_primary():
    r = _reward()([Score(values=_passing_values())])
    assert abs(r - 0.8) < 1e-9


def test_hard_clash_fail_returns_floor():
    values = _passing_values()
    values["has_clash"] = 1.0
    r = _reward()([Score(values=values)])
    # Large negative floor plus a tiny primary term (ipsae=0.8) for ordering.
    assert r < -900.0
    assert abs(r - (-1e3 + 0.8)) < 1e-6


def test_hard_ipsae_below_min_returns_floor():
    values = _passing_values()
    values["ipsae"] = 0.4
    r = _reward()([Score(values=values)])
    assert r < -900.0


def test_soft_penalty_reduces_reward_monotonically():
    reward = _reward()
    base = reward([Score(values=_passing_values())])

    # Worsen a single soft metric progressively; reward must decrease each time.
    prev = base
    for sc in (0.50, 0.40, 0.30):
        values = _passing_values()
        values["shape_complementarity"] = sc  # below the soft min of 0.55
        r = reward([Score(values=values)])
        assert r < prev
        prev = r


def test_missing_keys_are_skipped():
    # Only the primary present: no hard gate can fail (all others missing), no soft penalty.
    r = _reward()([Score(values={"ipsae": 0.75})])
    assert abs(r - 0.75) < 1e-9


def test_range_bound_soft_penalty():
    reward = _reward()
    values = _passing_values()
    values["isoelectric_point"] = 11.0  # above the 9.0 upper bound
    assert reward([Score(values=values)]) < reward([Score(values=_passing_values())])


def test_interface_dg_zero_bound_penalty_does_not_explode():
    # Regression: a soft bound of {max: 0} must NOT give a ~1e-8 scale (near-infinite penalty).
    # With the scale floored at 1.0, a +2 REU dG costs a bounded ~2.0, not ~2e8.
    values = _passing_values()
    values["interface_dg"] = 2.0  # positive dG, violates {max: 0} by 2
    r = _reward()([Score(values=values)])
    assert r > -900.0  # nowhere near the hard-fail floor
    assert abs(r - (0.8 - 2.0)) < 1e-6


def test_nan_hard_metric_fails_closed():
    values = _passing_values()
    values["complex_plddt"] = float("nan")  # hard gate {min: 80}
    r = _reward()([Score(values=values)])
    assert r < -900.0  # non-finite hard metric rejects, not fail-open


def test_nan_soft_metric_penalized_not_ignored():
    values = _passing_values()
    values["contact_molecular_surface"] = float("nan")  # soft gate
    r = _reward()([Score(values=values)])
    assert math.isfinite(r)
    assert r < 0.8  # a non-finite soft metric adds a penalty rather than passing free


def test_missing_primary_rejects():
    # No ipsae at all -> candidate was not properly scored -> reject (not ranked mid-pack at ~0).
    values = _passing_values()
    del values["ipsae"]
    r = _reward()([Score(values=values)])
    assert r <= -1e3
