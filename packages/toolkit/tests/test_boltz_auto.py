"""Tests for the Boltz-2 and auto structure scorers (mocks only; no heavy deps)."""

from __future__ import annotations

import aapg.toolkit  # noqa: F401  (registration side effects)
from aapg.core.scoring import SCORERS, Candidate
from aapg.toolkit.boltz import _plddt_array_100, _to_plddt100


def test_scorers_registered() -> None:
    expected = {"boltz2", "boltz2_mock", "structure_auto", "structure_auto_mock"}
    assert expected <= set(SCORERS.available())


def test_plddt_normalisation_to_0_100() -> None:
    import numpy as np

    # 0-1 scale is scaled up; already-0-100 is left alone.
    assert _to_plddt100(0.85) == 85.0
    assert _to_plddt100(85.0) == 85.0
    assert np.allclose(_plddt_array_100(np.array([0.6, 0.9])), [60.0, 90.0])
    assert np.allclose(_plddt_array_100(np.array([60.0, 90.0])), [60.0, 90.0])


def test_boltz_mock_scale_and_determinism() -> None:
    scorer = SCORERS.create("boltz2_mock")
    cand = Candidate(structure_path="mock://c.cif", binder_sequence="ACDEFGHIKL", name="c0")
    a = scorer.score(cand)
    b = scorer.score(cand)
    assert a.values == b.values
    # complex_plddt is on the 0-100 scale (so the reward's `min: 80` gate is meaningful).
    assert 60.0 <= a.values["complex_plddt"] <= 95.0
    assert {"ipsae", "iptm", "ipae_min", "ipae_mean", "complex_plddt", "has_clash"} <= set(a.values)


def test_auto_mock_delegates_to_boltz() -> None:
    scorer = SCORERS.create("structure_auto_mock")
    cand = Candidate(structure_path="mock://c.cif", binder_sequence="ACDEFGHIKL", name="c0")
    score = scorer.score(cand)
    assert score.metadata["auto_backend"] == "boltz2_mock"
    assert "ipsae" in score.values
