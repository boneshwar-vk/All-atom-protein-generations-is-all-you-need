"""Placeholder test: the package and its registries import cleanly."""


def test_core_imports():
    import aapg.core  # noqa: F401
    from aapg.core.scoring import REWARDS, SCORERS

    assert SCORERS.kind == "scorer"
    assert REWARDS.kind == "reward"
