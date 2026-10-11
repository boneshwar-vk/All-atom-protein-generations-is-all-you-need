"""Placeholder test: module 1 imports cleanly."""


def test_grpo_imports():
    import aapg.grpo  # noqa: F401
    from aapg.grpo.trainer import GRPOTrainer

    assert GRPOTrainer().cfg == {}
