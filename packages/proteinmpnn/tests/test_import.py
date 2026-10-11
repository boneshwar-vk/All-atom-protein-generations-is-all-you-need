"""Placeholder test: the vendored-fork package imports cleanly."""


def test_proteinmpnn_imports():
    import proteinmpnn

    assert proteinmpnn.__vendored__ is False  # flip to True once upstream is vendored
