"""Placeholder test: module 3 imports cleanly."""


def test_validation_imports():
    import aapg.validation  # noqa: F401
    from aapg.validation.pipeline import ValidationPipeline

    assert ValidationPipeline().stages == []
