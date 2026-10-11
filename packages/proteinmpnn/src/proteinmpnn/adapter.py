"""Thin adapter over the vendored ProteinMPNN sequence designer (module 1).

Gives the GRPO selection pipeline a stable surface for designing binder sequences onto a
generated backbone, without requiring the heavy ProteinMPNN stack (torch, the vendored
``proteinmpnn`` model) to be importable here. The real designer (:class:`MPNNDesigner`)
imports the backend **lazily** inside :meth:`MPNNDesigner.design` and raises a clear error when
it is absent; a dependency-free :class:`MockMPNNDesigner` returns deterministic plausible
sequences for tests and dry runs. Pick one through :func:`get_designer`.
"""

from __future__ import annotations

import hashlib
from typing import Any

__all__ = [
    "MPNNDesigner",
    "MockMPNNDesigner",
    "get_designer",
    "CANONICAL_AMINO_ACIDS",
]

# The 20 canonical amino acids, in a fixed order, used to decode mock sequences.
CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class MPNNDesigner:
    """Design binder sequences onto a backbone with the vendored ProteinMPNN model.

    The heavy backend is imported lazily in :meth:`design`, so constructing the designer (and
    importing this module) never requires the ProteinMPNN stack.
    """

    def __init__(self, checkpoint: str | None = None, temperature: float = 0.1, **cfg: Any) -> None:
        self.checkpoint = checkpoint
        self.temperature = temperature
        self.cfg = cfg

    def design(
        self,
        backbone: Any,
        n_seqs: int = 8,
        binder_chains: tuple[str, ...] = ("A",),
        target_chains: tuple[str, ...] = ("B",),
        seed: int = 0,
    ) -> list[str]:
        """Design ``n_seqs`` binder sequences for ``backbone`` (lazy heavy import).

        Raises:
            RuntimeError: if the vendored ``proteinmpnn`` backend cannot be imported, with
                guidance to vendor it / provide weights or use the mock designer.
        """
        try:
            import proteinmpnn.protein_mpnn_utils as _mpnn_utils  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ProteinMPNN is not available: could not import the vendored model backend "
                "('proteinmpnn.protein_mpnn_utils'). Vendor the upstream source and provide "
                "weights (set 'checkpoint'), or use get_designer(mock=True) for CPU dry runs."
            ) from exc
        # Real design wiring is added when the ProteinMPNN source is vendored.
        raise NotImplementedError(
            "MPNNDesigner.design is wired up when proteinmpnn is vendored; the lazy import "
            "above succeeded but the real design loop is not implemented in v1."
        )


class MockMPNNDesigner(MPNNDesigner):
    """Deterministic, dependency-free stand-in for :class:`MPNNDesigner`.

    Returns ``n_seqs`` plausible binder sequences of ``backbone.length`` residues over the
    canonical amino acids, derived deterministically from ``seed`` and the backbone identity.
    No heavy imports.
    """

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
            # Deterministic byte stream from (backbone, seed, seq index), expanded as needed.
            residues: list[str] = []
            counter = 0
            while len(residues) < length:
                digest = hashlib.sha256(f"{backbone_key}:{seed}:{s}:{counter}".encode()).digest()
                for byte in digest:
                    if len(residues) >= length:
                        break
                    residues.append(CANONICAL_AMINO_ACIDS[byte % len(CANONICAL_AMINO_ACIDS)])
                counter += 1
            sequences.append("".join(residues))
        return sequences


def get_designer(mock: bool = False, **cfg: Any) -> MPNNDesigner:
    """Return a :class:`MockMPNNDesigner` when ``mock`` else a real :class:`MPNNDesigner`."""
    if mock:
        return MockMPNNDesigner(**cfg)
    return MPNNDesigner(**cfg)
