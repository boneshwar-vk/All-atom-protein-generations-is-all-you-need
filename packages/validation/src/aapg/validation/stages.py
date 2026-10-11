"""Pipeline stage abstraction (placeholder).

Each validation step is a typed ``Stage`` with declared inputs/outputs so stages can be
composed and reordered from config. Concrete stages wrap tools from ``aapg.toolkit``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    """One step of the validation pipeline (placeholder)."""

    name: str = "stage"

    @abstractmethod
    def run(self, inputs: Any) -> Any:
        """Transform ``inputs`` into this stage's outputs (placeholder)."""
        raise NotImplementedError
