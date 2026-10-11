"""Importing the toolkit registers the concrete scorers (no heavy dependencies)."""

from __future__ import annotations


def test_scorers_registered() -> None:
    import aapg.toolkit  # noqa: F401
    from aapg.core.scoring import SCORERS

    expected = {
        "protenix_v2",
        "rosetta_interface",
        "protenix_v2_mock",
        "rosetta_interface_mock",
    }
    assert expected <= set(SCORERS.available())


def test_mock_scorers_creatable_by_name() -> None:
    import aapg.toolkit  # noqa: F401
    from aapg.core.scoring import SCORERS

    for name in ("protenix_v2_mock", "rosetta_interface_mock"):
        scorer = SCORERS.create(name)
        assert scorer.name == name
