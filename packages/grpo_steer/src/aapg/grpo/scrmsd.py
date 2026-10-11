"""Self-consistency RMSD (scRMSD) via pure-numpy Kabsch superposition.

scRMSD is the 11th recorded metric: it measures whether the sequence designed onto a backbone
actually folds back to that backbone. The designed binder CA trace (from RFD3) is optimally
superposed onto the predicted-complex binder CA trace (from the structure predictor) with the
Kabsch algorithm, and the residual CA-RMSD is returned. Low scRMSD means the design is
self-consistent; it is used both as a recorded metric and as a hard gate in the reward.
"""

from __future__ import annotations

import numpy as np

__all__ = ["self_consistency_rmsd", "kabsch_rmsd"]


def kabsch_rmsd(mobile: np.ndarray, reference: np.ndarray) -> float:
    """Minimum CA-RMSD between ``mobile`` and ``reference`` after optimal rigid superposition.

    Both arrays are ``(N, 3)`` and paired row-by-row. The optimal rotation+translation (no
    scaling, no reflection) aligning ``mobile`` onto ``reference`` is found with the Kabsch
    algorithm, and the root-mean-square deviation of the aligned coordinates is returned.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mobile.shape != reference.shape:
        raise ValueError(
            f"coordinate shapes differ: {mobile.shape} vs {reference.shape}; scRMSD needs a "
            "one-to-one residue correspondence."
        )
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"expected (N, 3) coordinates, got shape {mobile.shape}")
    if mobile.shape[0] == 0:
        return 0.0

    # Center both point sets on their centroids (removes the optimal translation).
    mobile_c = mobile - mobile.mean(axis=0)
    reference_c = reference - reference.mean(axis=0)

    # Optimal rotation from the SVD of the cross-covariance; correct for reflection.
    covariance = mobile_c.T @ reference_c
    u, _s, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag(np.array([1.0, 1.0, d]))
    rotation = vt.T @ correction @ u.T

    aligned = mobile_c @ rotation.T
    diff = aligned - reference_c
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def self_consistency_rmsd(designed_ca: np.ndarray, predicted_ca: np.ndarray) -> float:
    """scRMSD between the designed binder CA trace and the predicted-complex binder CA trace.

    Thin wrapper over :func:`kabsch_rmsd` naming the two inputs; identical coordinates give 0,
    and a pure rotation or translation of one relative to the other also gives ~0 (superposition
    removes them).
    """
    return kabsch_rmsd(designed_ca, predicted_ca)
