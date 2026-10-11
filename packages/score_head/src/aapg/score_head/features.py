"""Featurisation: complex structure -> model inputs (placeholder).

Turns a target+binder complex (PDB/mmCIF) into the tensors the score head consumes -- e.g. a
residue graph with per-residue and pair features (coordinates/frames, residue types, a
chain/interface mask). The concrete representation is chosen when the head is implemented; this
stub only fixes the entry point and the structure->features contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComplexFeatures:
    """Featurised complex handed to the score head (placeholder).

    Fields are intentionally untyped-as-``Any`` arrays for now; they become concrete (numpy or
    torch tensors) with the implementation.
    """

    node_features: Any = None  # (N_res, F_node)
    pair_features: Any = None  # (N_res, N_res, F_pair)
    coords: Any = None  # (N_res, 3) representative coordinates
    chain_ids: tuple[str, ...] = ()  # length N_res
    interface_mask: Any = None  # (N_res,) True for binder/target interface residues
    metadata: dict[str, Any] = field(default_factory=dict)


def featurize_complex(
    structure_path: str,
    binder_chains: tuple[str, ...] = ("A",),
    target_chains: tuple[str, ...] = ("B",),
    **cfg: Any,
) -> ComplexFeatures:
    """Featurise a complex structure for the score head (placeholder)."""
    raise NotImplementedError(
        "aapg.score_head.features.featurize_complex is a placeholder; the structure->tensor "
        "featuriser is implemented when the head is built."
    )
