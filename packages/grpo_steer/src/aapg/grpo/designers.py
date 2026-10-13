"""Config-switchable sequence-designer backends for the GRPO selection pipeline.

The pipeline designs binder sequences onto a generated backbone with a *designer*. The default
backend is the vendored ProteinMPNN (:mod:`proteinmpnn.adapter`, which supports the ``vanilla``
and ``soluble`` weight sets). :func:`get_designer` is a small name-based factory so the backend
can be switched from config (``designer.name``) or the CLI (``designer=protonmpnn``).

A second backend, **ProtonMPNN**, ships here as a *placeholder* so the switch is wired end to
end before the real model is vendored: its real path raises with a pointer to where the
source/weights get wired, while its mock returns deterministic sequences that are DISTINCT from
the ProteinMPNN mock (a different hash salt) -- so a mock run visibly changes when the backend is
switched. Every designer shares the :class:`proteinmpnn.adapter.MPNNDesigner` interface
(``design(backbone, n_seqs, binder_chains, target_chains, seed) -> list[str]``).
"""

from __future__ import annotations

import hashlib
import sys
from typing import Any

from proteinmpnn.adapter import CANONICAL_AMINO_ACIDS

__all__ = [
    "get_designer",
    "ProtonMPNNDesigner",
    "MockProtonMPNNDesigner",
    "DESIGNERS",
]

# Hash salt that makes the ProtonMPNN mock output DISTINCT from the ProteinMPNN mock for the same
# backbone/seed, so switching backends is observable in a mock run.
_PROTON_MOCK_SALT = "protonmpnn"


class ProtonMPNNDesigner:
    """Placeholder ProtonMPNN sequence designer -- NOT YET WIRED.

    Mirrors the :class:`proteinmpnn.adapter.MPNNDesigner` interface so it is a drop-in alternate,
    but has no real backend yet: :meth:`design` raises :class:`RuntimeError` pointing at where to
    wire the real ProtonMPNN source/weights. Use :class:`MockProtonMPNNDesigner` (via
    ``get_designer("protonmpnn", mock=True)``) for CPU dry runs until the real backend lands.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        temperature: float = 0.1,
        weights_set: str = "vanilla",
        python: str | None = None,
        **cfg: Any,
    ) -> None:
        self.model_name = checkpoint or "protonmpnn"
        self.temperature = temperature
        self.weights_set = weights_set
        self.python = python or sys.executable
        self.cfg = cfg

    def design(
        self,
        backbone: Any,
        n_seqs: int = 8,
        binder_chains: tuple[str, ...] = ("A",),
        target_chains: tuple[str, ...] = ("B",),
        seed: int = 0,
    ) -> list[str]:
        """Placeholder: raises until the real ProtonMPNN backend is wired in."""
        raise RuntimeError(
            "ProtonMPNN backend is a placeholder; point it at the real source/weights -- see "
            "configs/designer/protonmpnn.yaml"
        )


class MockProtonMPNNDesigner(ProtonMPNNDesigner):
    """Deterministic, dependency-free stand-in for :class:`ProtonMPNNDesigner`.

    Returns ``n_seqs`` plausible binder sequences of ``backbone.length`` residues over the
    canonical amino acids, using the same hashing style as the ProteinMPNN mock but a DISTINCT
    salt -- so a mock run with ``designer=protonmpnn`` produces visibly different sequences from
    the ProteinMPNN mock for the same backbone, making the backend switch observable. No heavy
    imports or subprocesses.
    """

    def __init__(self, **cfg: Any) -> None:
        # Mock ignores weights/interpreter; keep a permissive signature (matches MockMPNNDesigner).
        self.model_name = str(cfg.get("checkpoint") or "protonmpnn_mock")
        self.temperature = float(cfg.get("temperature", 0.1))
        self.weights_set = str(cfg.get("weights_set", "vanilla"))
        self.python = sys.executable
        self.cfg = cfg

    def design(
        self,
        backbone: Any,
        n_seqs: int = 8,
        binder_chains: tuple[str, ...] = ("A",),
        target_chains: tuple[str, ...] = ("B",),
        seed: int = 0,
    ) -> list[str]:
        length = int(getattr(backbone, "length", 0))
        backbone_key = str(getattr(backbone, "path", backbone))
        sequences: list[str] = []
        for s in range(n_seqs):
            residues: list[str] = []
            counter = 0
            while len(residues) < length:
                raw = f"{_PROTON_MOCK_SALT}:{backbone_key}:{seed}:{s}:{counter}"
                digest = hashlib.sha256(raw.encode()).digest()
                for byte in digest:
                    if len(residues) >= length:
                        break
                    residues.append(CANONICAL_AMINO_ACIDS[byte % len(CANONICAL_AMINO_ACIDS)])
                counter += 1
            sequences.append("".join(residues))
        return sequences


def _proteinmpnn(mock: bool = False, **cfg: Any) -> Any:
    """Build the vendored ProteinMPNN designer (supports ``weights_set=vanilla|soluble``)."""
    from proteinmpnn.adapter import get_designer as _get_proteinmpnn

    return _get_proteinmpnn(mock=mock, **cfg)


def _protonmpnn(mock: bool = False, **cfg: Any) -> Any:
    """Build the placeholder ProtonMPNN designer (mock stand-in, or the not-yet-wired real path)."""
    if mock:
        return MockProtonMPNNDesigner(**cfg)
    return ProtonMPNNDesigner(**cfg)


# name -> factory(mock, **cfg). A plain dict dispatch (rather than aapg.core.registry.Registry)
# because each factory takes the ``mock`` switch and returns the mock/real variant itself, which
# the Registry's create(name, **kwargs) signature does not model.
DESIGNERS: dict[str, Any] = {
    "proteinmpnn": _proteinmpnn,
    "protonmpnn": _protonmpnn,
}


def get_designer(name: str = "proteinmpnn", mock: bool = False, **cfg: Any) -> Any:
    """Return the sequence designer registered under ``name``.

    ``"proteinmpnn"`` (default) is the vendored ProteinMPNN (``weights_set=vanilla|soluble``);
    ``"protonmpnn"`` is the placeholder alternate. ``mock=True`` selects the deterministic CPU
    stand-in. Extra kwargs are forwarded to the backend constructor.

    Raises:
        ValueError: if ``name`` is not a known designer, listing the known names.
    """
    try:
        factory = DESIGNERS[name]
    except KeyError:
        known = ", ".join(sorted(DESIGNERS))
        raise ValueError(f"unknown designer {name!r}; known designers: {known}") from None
    return factory(mock=mock, **cfg)
