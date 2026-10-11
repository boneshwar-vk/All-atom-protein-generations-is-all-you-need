"""Placeholder tests: import, registration, and the deterministic mock scorer."""

from __future__ import annotations

from aapg.core.scoring import SCORERS, Candidate


def test_score_head_registers() -> None:
    import aapg.score_head  # noqa: F401

    assert "structure_head" in SCORERS.available()
    assert "structure_head_mock" in SCORERS.available()


def test_mock_scorer_is_deterministic() -> None:
    import aapg.score_head  # noqa: F401

    scorer = SCORERS.create("structure_head_mock")
    cand = Candidate(structure_path="mock://c.cif", binder_sequence="ACDEFGHIKL", name="c0")
    a = scorer.score(cand)
    b = scorer.score(cand)
    assert a.values == b.values
    assert "ipsae" in a.values and "iptm" in a.values
    assert 0.3 <= a.values["ipsae"] <= 0.9
    assert a.metadata["mock"] is True
