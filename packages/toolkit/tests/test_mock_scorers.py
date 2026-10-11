"""Tests for the deterministic CPU mock scorers (no heavy dependencies)."""

from __future__ import annotations

from aapg.core.scoring import Candidate
from aapg.toolkit.protenix import MockProtenixScorer
from aapg.toolkit.rosetta import MockRosettaScorer


def _candidate(seq: str = "MKTAYIAKQR") -> Candidate:
    return Candidate(binder_sequence=seq, binder_chains=("A",), target_chains=("B",))


def test_mock_protenix_keys_and_ranges() -> None:
    score = MockProtenixScorer().score(_candidate())
    assert set(score.values) == {
        "ipsae",
        "iptm",
        "ipae_min",
        "ipae_mean",
        "complex_plddt",
        "has_clash",
    }
    v = score.values
    assert 0.3 <= v["ipsae"] <= 0.9
    assert 0.4 <= v["iptm"] <= 0.95
    assert 2.0 <= v["ipae_min"] <= 15.0
    assert 5.0 <= v["ipae_mean"] <= 25.0
    assert 60.0 <= v["complex_plddt"] <= 95.0
    assert v["has_clash"] in (0.0, 1.0)


def test_mock_rosetta_keys_and_ranges() -> None:
    score = MockRosettaScorer().score(_candidate())
    assert set(score.values) == {
        "interface_dg",
        "shape_complementarity",
        "contact_molecular_surface",
        "sap_normalized",
        "isoelectric_point",
    }
    v = score.values
    assert -45.0 <= v["interface_dg"] <= -5.0
    assert 0.5 <= v["shape_complementarity"] <= 0.78
    assert 150.0 <= v["contact_molecular_surface"] <= 600.0
    assert 10.0 <= v["sap_normalized"] <= 60.0
    assert 4.5 <= v["isoelectric_point"] <= 9.5


def test_mock_scorers_deterministic() -> None:
    cand = _candidate("ACDEFGHIKL")
    assert MockProtenixScorer().score(cand).values == MockProtenixScorer().score(cand).values
    assert MockRosettaScorer().score(cand).values == MockRosettaScorer().score(cand).values


def test_mock_scorers_vary_with_sequence() -> None:
    a = MockProtenixScorer().score(_candidate("AAAAAAAAAA")).values
    b = MockProtenixScorer().score(_candidate("WWWWWWWWWW")).values
    assert a != b
