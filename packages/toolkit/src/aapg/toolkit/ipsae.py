"""Pure-numpy implementation of the ipSAE interface metric.

ipSAE ("interface predicted Scaled Aligned Error") is a per-interface confidence score for
predicted protein complexes. For each ordered chain pair it rescales the predicted aligned
error (PAE) through a TM-score-like transform whose ``d0`` normalisation is derived from the
number of interface partners of each residue, then takes the best-scoring residue as the
directional interface score. The symmetric ipSAE of a pair is the maximum over both
directions.

This module is a clean-room pure-numpy reimplementation of just the pieces used by the
validation pipeline (ipSAE, ipTM with chain-pair ``d0``, pDockQ and LIS). It is adapted from,
and cross-checked against, the reference implementation:

    DunbrackLab/IPSAE -- https://github.com/DunbrackLab/IPSAE (MIT License).

and the accompanying preprint:

    Dunbrack, R. L. Jr. "Rēs ipSAE loquuntur: What the Scaled Aligned Error says about itself.
    A metric for the confidence in a predicted protein-protein interface."
    bioRxiv (2025). https://doi.org/10.1101/2025.02.10.637595

The ``d0`` formula follows Yang & Skolnick, *Proteins* 57:702-710 (2004). The pDockQ formula
follows Bryant et al., *Nat. Commun.* 13:1265 (2022); LIS follows Kim et al., bioRxiv (2024)
https://doi.org/10.1101/2024.02.19.580970. No source code is copied from the reference; only
the published formulas are reproduced.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["calc_d0_array", "calc_d0", "compute_ipsae"]

# pDockQ CB-CB contact distance cutoff in Angstrom (Bryant et al. 2022).
PDOCKQ_CONTACT_CUTOFF = 8.0
# LIS PAE threshold in Angstrom (Kim et al. 2024).
LIS_PAE_CUTOFF = 12.0


# The DunbrackLab/IPSAE reference clamps L to a minimum of 27 before the d0 formula, giving a
# floor of d0 = 1.24*(12)**(1/3) - 1.8 ≈ 1.0389. The scalar and array forms below are kept
# identical so a per-residue d0 never disagrees with a chain-pair d0 at the boundary.
_D0_MIN_LENGTH = 27.0


def calc_d0_array(l_array: Any, min_value: float = 1.0) -> np.ndarray:
    """Vectorised TM-score ``d0`` from a chain/interface length array.

    ``L`` is clamped to a minimum of 27 (matching the DunbrackLab/IPSAE reference) before the
    Yang & Skolnick (2004) formula ``d0 = 1.24 * (L - 15) ** (1/3) - 1.8`` is applied, and the
    result is clamped to ``min_value`` (1.0 for proteins).
    """
    length = np.maximum(_D0_MIN_LENGTH, np.asarray(l_array, dtype=float))
    return np.maximum(min_value, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)


def calc_d0(length: float, min_value: float = 1.0) -> float:
    """Scalar TM-score ``d0`` for a single length ``L`` (identical to :func:`calc_d0_array`)."""
    clamped = max(_D0_MIN_LENGTH, float(length))
    d0 = 1.24 * (clamped - 15.0) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def _ptm(pae: np.ndarray, d0: float | np.ndarray) -> np.ndarray:
    """TM-score-like transform of the PAE: ``1 / (1 + (pae / d0) ** 2)``."""
    return 1.0 / (1.0 + (pae / d0) ** 2)


def _empty_pair() -> dict[str, float]:
    return {
        "ipsae": 0.0,
        "iptm_d0chn": 0.0,
        "pdockq": 0.0,
        "lis": 0.0,
        "n0res": 0.0,
        "d0res": 0.0,
    }


def compute_ipsae(
    pae: np.ndarray,
    coords: np.ndarray,
    chain_ids: Sequence[Any],
    plddt: np.ndarray,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> dict[str, Any]:
    """Compute ipSAE and companion interface metrics for a predicted complex.

    Args:
        pae: ``(N, N)`` predicted aligned error in Angstrom, token/residue indexed.
        coords: ``(N, 3)`` representative coordinates (CB, or CA for glycine) used for the
            pDockQ contact distances.
        chain_ids: length-``N`` sequence of chain labels (e.g. ``"A"``/``"B"``).
        plddt: ``(N,)`` per-residue pLDDT on the 0-100 scale.
        pae_cutoff: PAE threshold (Angstrom) for an interface residue pair; the ipSAE default
            is 10.
        dist_cutoff: interface distance threshold (Angstrom); kept for ipSAE CLI parity (the
            headline metrics below do not depend on it, pDockQ uses its own 8 A contact cut).

    Returns:
        A dict keyed by ordered pair ``f"{c1}_{c2}"`` mapping to
        ``{ipsae, iptm_d0chn, pdockq, lis, n0res, d0res}``, plus a ``"max"`` entry holding the
        symmetric best over all pairs, and (for the two-chain case especially) flat top-level
        ``"ipsae"`` and ``"iptm_d0chn"`` keys equal to that best. A single-chain input yields
        all zeros.
    """
    pae = np.asarray(pae, dtype=float)
    coords = np.asarray(coords, dtype=float)
    plddt = np.asarray(plddt, dtype=float)
    chains = np.asarray(list(chain_ids))

    # All four inputs must index the same token/residue axis of length N. A mismatch here is a
    # caller bug (e.g. the PAE token axis not aligned to the structure residues) and otherwise
    # surfaces as a cryptic IndexError deep in the loop or a silently-wrong score.
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"pae must be a square 2D matrix; got shape {pae.shape}")
    n = pae.shape[0]
    if not (coords.shape[0] == chains.shape[0] == plddt.shape[0] == n):
        raise ValueError(
            "ipSAE inputs must share one token-axis length N: "
            f"pae={n}, coords={coords.shape[0]}, chain_ids={chains.shape[0]}, "
            f"plddt={plddt.shape[0]}"
        )

    results: dict[str, Any] = {}
    # Preserve first-seen chain order.
    unique_chains = list(dict.fromkeys(chains.tolist()))

    if len(unique_chains) < 2:
        results["max"] = _empty_pair()
        results["ipsae"] = 0.0
        results["iptm_d0chn"] = 0.0
        return results

    # Pairwise CB-CB distances for pDockQ contacts.
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))

    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            m1 = chains == c1
            m2 = chains == c2

            # --- ipTM with chain-pair d0: d0 from L = n(c1) + n(c2), no PAE cutoff. ---
            n0chn = int(m1.sum() + m2.sum())
            d0chn = calc_d0(n0chn)
            ptm_chn = _ptm(pae, d0chn)
            if m1.any() and m2.any():
                # For each c1 residue, mean ptm over all c2 residues; take max over c1.
                iptm_byres = ptm_chn[:, m2].mean(axis=1)
                iptm_d0chn = float(iptm_byres[m1].max())
            else:
                iptm_d0chn = 0.0

            # --- ipSAE (headline, d0res): per-residue partner-count-based d0. ---
            valid = np.outer(m1, m2) & (pae < pae_cutoff)
            n0res_byres = valid.sum(axis=1)
            d0res_byres = calc_d0_array(n0res_byres)
            scores = np.zeros(n, dtype=float)
            for i in np.where(m1)[0]:
                row = valid[i]
                if row.any():
                    scores[i] = _ptm(pae[i], d0res_byres[i])[row].mean()
            c1_idx = np.where(m1)[0]
            if c1_idx.size > 0:
                best_i = int(c1_idx[np.argmax(scores[c1_idx])])
                ipsae_dir = float(scores[best_i])
                n0res_sel = float(n0res_byres[best_i])
                d0res_sel = float(d0res_byres[best_i])
            else:
                ipsae_dir, n0res_sel, d0res_sel = 0.0, 0.0, 0.0

            # --- pDockQ: CB-CB contacts < 8 A across the pair. ---
            contact = np.outer(m1, m2) & (distances <= PDOCKQ_CONTACT_CUTOFF)
            npairs = int(contact.sum())
            if npairs > 0:
                involved = contact.any(axis=1) | contact.any(axis=0)
                mean_plddt = float(plddt[involved].mean())
                x = mean_plddt * np.log10(npairs)
                pdockq = float(0.724 / (1.0 + np.exp(-0.052 * (x - 152.611))) + 0.018)
            else:
                pdockq = 0.0

            # --- LIS: mean of (12 - pae) / 12 over the c1-c2 block where pae < 12. ---
            block = pae[np.outer(m1, m2)]
            below = block[block < LIS_PAE_CUTOFF]
            lis = float(((LIS_PAE_CUTOFF - below) / LIS_PAE_CUTOFF).mean()) if below.size else 0.0

            results[f"{c1}_{c2}"] = {
                "ipsae": ipsae_dir,
                "iptm_d0chn": iptm_d0chn,
                "pdockq": pdockq,
                "lis": lis,
                "n0res": n0res_sel,
                "d0res": d0res_sel,
            }

    # Symmetric best over every ordered pair.
    best_pair = max(results.values(), key=lambda v: v["ipsae"])
    max_entry = {
        "ipsae": max(v["ipsae"] for v in results.values()),
        "iptm_d0chn": max(v["iptm_d0chn"] for v in results.values()),
        "pdockq": max(v["pdockq"] for v in results.values()),
        "lis": max(v["lis"] for v in results.values()),
        "n0res": best_pair["n0res"],
        "d0res": best_pair["d0res"],
    }
    results["max"] = max_entry
    results["ipsae"] = max_entry["ipsae"]
    results["iptm_d0chn"] = max_entry["iptm_d0chn"]
    return results
