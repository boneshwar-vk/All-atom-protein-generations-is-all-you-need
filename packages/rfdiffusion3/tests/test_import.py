"""Placeholder test: the vendored-fork package imports cleanly."""


def test_rfdiffusion3_imports():
    import rfdiffusion3

    assert rfdiffusion3.__vendored__ is False  # flip to True once upstream is vendored
