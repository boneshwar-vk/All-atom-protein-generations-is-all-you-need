"""Tests for the generic name->factory :class:`Registry`."""

from __future__ import annotations

import pytest

from aapg.core.registry import Registry


def _fresh_registry() -> Registry[object]:
    reg: Registry[object] = Registry("widget")

    @reg.register("alpha")
    def make_alpha(**kwargs: object) -> dict[str, object]:
        return {"kind": "alpha", **kwargs}

    @reg.register("beta")
    def make_beta(**kwargs: object) -> dict[str, object]:
        return {"kind": "beta", **kwargs}

    return reg


def test_register_and_create_with_kwargs() -> None:
    reg = _fresh_registry()
    created = reg.create("alpha", size=3)
    assert created == {"kind": "alpha", "size": 3}


def test_available_is_sorted() -> None:
    reg = _fresh_registry()
    assert reg.available() == ["alpha", "beta"]


def test_duplicate_registration_raises_value_error() -> None:
    reg = _fresh_registry()
    with pytest.raises(ValueError) as excinfo:
        reg.register("alpha")(lambda **_: object())
    message = str(excinfo.value)
    assert "widget" in message
    assert "alpha" in message


def test_create_missing_raises_key_error_with_available() -> None:
    reg = _fresh_registry()
    with pytest.raises(KeyError) as excinfo:
        reg.create("missing")
    message = str(excinfo.value)
    assert "missing" in message
    # available names are listed to help the caller.
    assert "alpha" in message
    assert "beta" in message


def test_contains() -> None:
    reg = _fresh_registry()
    assert "alpha" in reg
    assert "nope" not in reg


def test_get_returns_factory_or_none() -> None:
    reg = _fresh_registry()
    factory = reg.get("beta")
    assert factory is not None
    assert factory()["kind"] == "beta"
    assert reg.get("missing") is None


def test_register_returns_class_unchanged() -> None:
    reg: Registry[object] = Registry("thing")

    @reg.register("c")
    class MyThing:
        pass

    assert reg.get("c") is MyThing
