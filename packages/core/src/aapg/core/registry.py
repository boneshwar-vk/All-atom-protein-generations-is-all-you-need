"""Generic name->factory registry (placeholder).

The registry is the extensibility backbone: scorers, rewards and pipeline stages register
themselves under a string name and consumers resolve them by name from config, so new
components can be added without editing call sites.

Placeholder only — the decorator/lookup surface is sketched; no behavior is finalized.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of factories for a given component kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        # name -> factory; populated by `register`.
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that records a factory under ``name`` (placeholder)."""
        raise NotImplementedError

    def create(self, name: str, **kwargs: object) -> T:
        """Instantiate the component registered under ``name`` (placeholder)."""
        raise NotImplementedError

    def available(self) -> list[str]:
        """Return the registered names (placeholder)."""
        raise NotImplementedError
