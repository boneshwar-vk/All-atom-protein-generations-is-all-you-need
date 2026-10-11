"""Placeholder test: module 4 imports cleanly."""


def test_mpnn_iter_imports():
    import aapg.mpnn_iter  # noqa: F401
    from aapg.mpnn_iter.loop import IterativeDesigner

    assert IterativeDesigner().scorers == []
