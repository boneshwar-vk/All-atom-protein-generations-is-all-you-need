"""Generic name->factory registry.

The registry is the extensibility backbone: scorers, rewards and pipeline stages register
themselves under a string name and consumers resolve them by name from config, so new
components can be added without editing call sites.
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
        """Return a decorator that records a factory under ``name``.

        The decorated class or factory is stored unchanged and returned as-is, so the
        decorator is transparent at the definition site::

            @SCORERS.register("my_scorer")
            class MyScorer(Scorer):
                ...

        Raises:
            ValueError: if ``name`` is already registered for this kind.
        """

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(
                    f"{self.kind} {name!r} is already registered; "
                    f"registered {self.kind}s: {self.available()}"
                )
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, **kwargs: object) -> T:
        """Instantiate the component registered under ``name`` with ``kwargs``.

        Raises:
            KeyError: if ``name`` is not registered, listing the available names.
        """
        try:
            factory = self._factories[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available {self.kind}s: {self.available()}"
            ) from None
        return factory(**kwargs)

    def available(self) -> list[str]:
        """Return the registered names, sorted alphabetically."""
        return sorted(self._factories)

    def get(self, name: str) -> Callable[..., T] | None:
        """Return the factory registered under ``name``, or ``None`` if absent."""
        return self._factories.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._factories
