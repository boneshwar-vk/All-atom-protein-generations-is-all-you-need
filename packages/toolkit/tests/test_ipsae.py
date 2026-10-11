"""Tests for the pure-numpy ipSAE metric (no heavy dependencies)."""

from __future__ import annotations

import numpy as np

from aapg.toolkit.ipsae import calc_d0, calc_d0_array, compute_ipsae


def _two_chain_system(ab_pae: float) -> dict:
    """Build a 3+3 residue A/B system with a constant A<->B PAE block.

    Intra-chain PAE is 0; the A-B and B-A blocks are set to ``ab_pae``. Coordinates place the
    two chains ~6 A apart so all cross-chain CB-CB pairs are in pDockQ contact.
    """
    n = 6
    chain_ids = ["A", "A", "A", "B", "B", "B"]
    pae = np.zeros((n, n), dtype=float)
    a = slice(0, 3)
    b = slice(3, 6)
    pae[a, b] = ab_pae
    pae[b, a] = ab_pae
    # Chain A along x at 0,1,2; chain B shifted +6 in y, same spacing.
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [1.0, 6.0, 0.0],
            [2.0, 6.0, 0.0],
        ]
    )
    plddt = np.full(n, 90.0)
    return {"pae": pae, "coords": coords, "chain_ids": chain_ids, "plddt": plddt}


def test_calc_d0_scalar_and_array() -> None:
    # Matching DunbrackLab/IPSAE, L clamps to a minimum of 27 before the formula, giving a floor
    # d0 = 1.24*(12)**(1/3) - 1.8 ≈ 1.0389. Scalar and array forms must agree.
    floor = 1.24 * (27 - 15) ** (1 / 3) - 1.8
    assert abs(calc_d0(3) - floor) < 1e-12
    assert abs(calc_d0(27) - floor) < 1e-12
    arr = calc_d0_array([1, 3, 27])
    assert np.allclose(arr, floor)
    assert abs(calc_d0(27) - float(calc_d0_array([27])[0])) < 1e-12
    # Large L follows Yang & Skolnick.
    assert calc_d0(100) == max(1.0, 1.24 * (100 - 15) ** (1 / 3) - 1.8)


def test_ipsae_good_better_than_bad() -> None:
    good = _two_chain_system(ab_pae=1.0)
    bad = _two_chain_system(ab_pae=20.0)
    good_res = compute_ipsae(**good)
    bad_res = compute_ipsae(**bad)

    assert 0.0 <= good_res["ipsae"] <= 1.0
    assert 0.0 <= bad_res["ipsae"] <= 1.0
    assert good_res["ipsae"] > bad_res["ipsae"]
    # Bad case: every cross-chain PAE (20) exceeds the 10 A cutoff -> no valid pairs.
    assert bad_res["ipsae"] == 0.0


def test_ipsae_hand_computed_value() -> None:
    # Constant A-B PAE of 2.0, 3 partners/residue -> n0res=3 -> d0res clamps to the L=27 floor
    # d0 = 1.24*(12)**(1/3)-1.8 ≈ 1.038891; ptm = 1/(1+(2/d0)**2); meaned over 3 == that ptm.
    d0 = 1.24 * (27 - 15) ** (1 / 3) - 1.8
    expected = 1.0 / (1.0 + (2.0 / d0) ** 2)
    res = compute_ipsae(**_two_chain_system(ab_pae=2.0))
    assert abs(res["A_B"]["d0res"] - d0) < 1e-9
    assert res["A_B"]["n0res"] == 3.0
    assert abs(res["A_B"]["ipsae"] - expected) < 1e-9
    assert abs(res["ipsae"] - expected) < 1e-9


def test_ipsae_structure() -> None:
    res = compute_ipsae(**_two_chain_system(ab_pae=1.0))
    assert set(res) >= {"A_B", "B_A", "max", "ipsae", "iptm_d0chn"}
    for key in ("ipsae", "iptm_d0chn", "pdockq", "lis", "n0res", "d0res"):
        assert key in res["A_B"]
    # Low cross-chain PAE => strong pDockQ / LIS signal.
    assert res["A_B"]["pdockq"] > 0.0
    assert 0.0 <= res["A_B"]["lis"] <= 1.0
    assert res["max"]["ipsae"] == max(res["A_B"]["ipsae"], res["B_A"]["ipsae"])


def test_ipsae_single_chain_returns_zeros() -> None:
    n = 3
    res = compute_ipsae(
        pae=np.zeros((n, n)),
        coords=np.zeros((n, 3)),
        chain_ids=["A", "A", "A"],
        plddt=np.full(n, 80.0),
    )
    assert res["ipsae"] == 0.0
    assert res["iptm_d0chn"] == 0.0
    assert res["max"]["ipsae"] == 0.0
