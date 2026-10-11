"""Placeholder test: module 2 imports cleanly."""


def test_ellipsoid_imports():
    import aapg.ellipsoid  # noqa: F401
    from aapg.ellipsoid.layout import EllipsoidLayout, Target

    assert Target.OFF.value == "off_target"
    assert EllipsoidLayout().ellipsoids == []
