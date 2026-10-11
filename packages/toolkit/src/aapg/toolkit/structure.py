"""Small gemmi-backed structure utilities shared by the structure-prediction scorers.

Kept separate from :mod:`aapg.toolkit.ipsae` (which is pure numpy, no heavy deps): everything
here reads a predicted complex via ``gemmi`` and is imported lazily inside scorer methods, so the
toolkit still imports with gemmi absent (e.g. in CI).
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["interface_clash_from_cif"]

# A heavy-atom pair from two different chains closer than this (Angstrom) is a steric clash: no
# covalent bond crosses a chain boundary, so sub-2 A inter-chain heavy-atom contact is unphysical.
_CLASH_CUTOFF = 2.0


def interface_clash_from_cif(
    cif_path: str,
    binder_chains: Sequence[str],
    target_chains: Sequence[str],
    cutoff: float = _CLASH_CUTOFF,
) -> float:
    """Return ``1.0`` if any binder/target heavy-atom pair clashes (< ``cutoff`` A), else ``0.0``.

    This gives the reward's ``has_clash`` hard gate a real signal on backends (e.g. Boltz-2) whose
    confidence JSON does not report a clash flag. Hydrogens are ignored; only inter-chain
    (binder<->target) pairs are checked, so legitimate bonded/intra-chain contacts never count.
    Missing either chain set yields ``0.0`` (no interface to clash).
    """
    import gemmi
    import numpy as np

    structure = gemmi.read_structure(cif_path)
    model = structure[0]
    binder_set = set(binder_chains)
    target_set = set(target_chains)

    binder_xyz: list[list[float]] = []
    target_xyz: list[list[float]] = []
    for chain in model:
        if chain.name in binder_set:
            dst = binder_xyz
        elif chain.name in target_set:
            dst = target_xyz
        else:
            continue
        for res in chain:
            for atom in res:
                if atom.element.is_hydrogen():
                    continue
                dst.append([atom.pos.x, atom.pos.y, atom.pos.z])

    if not binder_xyz or not target_xyz:
        return 0.0

    binder = np.asarray(binder_xyz, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    cut2 = float(cutoff) ** 2
    # Chunk the binder atoms so the pairwise distance block stays small regardless of complex size.
    for start in range(0, binder.shape[0], 256):
        block = binder[start : start + 256]
        diff = block[:, None, :] - target[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        if d2.size and float(d2.min()) < cut2:
            return 1.0
    return 0.0
