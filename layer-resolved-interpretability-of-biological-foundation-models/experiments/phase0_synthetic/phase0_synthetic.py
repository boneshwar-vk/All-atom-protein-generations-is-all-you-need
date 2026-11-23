#!/usr/bin/env python3
"""
Phase 0 — synthetic calibration + the "geometry kill-test".

Purpose (from MASTER_PLAN.md, run before touching any model weights):
  Settle three things on planted data with known ground truth, cheaply, on CPU:
    (a) Does a geometry-matched CURVED/circular code beat a FLAT Euclidean block of
        equal dimension at MATCHED BITS?  (predicted: TIE -> "curved beats flat" is
        not the contribution; curved dictionaries are a control, per SASA/Dal26)
    (b) Do single-direction (scalar, k=1) codes pay a COVERING-NUMBER penalty
        (atoms needed grows as the resolution tightens)?  (predicted: yes)
    (c) Is the planted angular coordinate RECOVERABLE UP TO ISOMETRY from the
        2-D subspace (circular corr ~ 1 after best rotation+reflection)? (predicted: yes)

This is a falsifiable gate, deliberately designed so the *expected* outcome demotes
the "new geometry" headline and promotes the identity+cause+truth-vs-confidence story.

No external deps beyond numpy. Matplotlib is optional (plots skipped if absent).
Deterministic (fixed seeds). Writes results JSON next to this file.
"""
import json
import os
import numpy as np

RNG = np.random.default_rng(0)


# ----------------------------- planting -----------------------------
def plant_circle(n=20000, d=64, noise=0.05, radial_noise=0.0):
    """x = A @ [cos t, sin t] * r + ambient_noise ;  A has orthonormal columns."""
    t = RNG.uniform(0.0, 2.0 * np.pi, size=n)
    r = np.ones(n) + (radial_noise * RNG.standard_normal(n) if radial_noise > 0 else 0.0)
    z = np.stack([np.cos(t), np.sin(t)], axis=1) * r[:, None]        # (n,2)
    A, _ = np.linalg.qr(RNG.standard_normal((d, 2)))                 # (d,2) orthonormal cols
    x = z @ A.T                                                      # (n,d), signal lives in a 2-plane
    x = x + noise * RNG.standard_normal((n, d))
    return x, t, A


def plant_torus(n=20000, d=64, noise=0.05):
    a = RNG.uniform(0, 2 * np.pi, n)
    b = RNG.uniform(0, 2 * np.pi, n)
    z = np.stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)], axis=1)  # (n,4)
    A, _ = np.linalg.qr(RNG.standard_normal((d, 4)))
    x = z @ A.T + noise * RNG.standard_normal((n, d))
    return x, np.stack([a, b], axis=1), A


# ----------------------------- helpers -----------------------------
def fvu(x, x_hat):
    """Fraction of variance unexplained (0 = perfect, 1 = predicts the mean)."""
    num = np.sum((x - x_hat) ** 2)
    den = np.sum((x - x.mean(0, keepdims=True)) ** 2)
    return float(num / den)


def pca_subspace(x, k):
    """Top-k principal directions (columns), mean-centered."""
    mu = x.mean(0, keepdims=True)
    xc = x - mu
    # SVD on a sample for speed
    idx = RNG.choice(x.shape[0], size=min(4000, x.shape[0]), replace=False)
    U, S, Vt = np.linalg.svd(xc[idx], full_matrices=False)
    return mu, Vt[:k].T  # (d,k)


def reconstruct_subspace(x, mu, V, unit_circle=False):
    """Project onto the subspace; optionally constrain the 2-D code to the unit circle."""
    code = (x - mu) @ V          # (n,k)
    if unit_circle and V.shape[1] == 2:
        nrm = np.linalg.norm(code, axis=1, keepdims=True)
        nrm = np.where(nrm < 1e-9, 1.0, nrm)
        code_used = code / nrm   # project code to S^1 (radius 1)
    else:
        code_used = code
    x_hat = mu + code_used @ V.T
    return x_hat, code


def circular_corr_up_to_isometry(theta_true, code2d):
    """Best alignment of recovered angle to truth over rotation+reflection.
    Returns mean cos residual in [-1,1]; ~1 means recovered up to isometry."""
    phi = np.arctan2(code2d[:, 1], code2d[:, 0])
    best = -2.0
    for s in (+1.0, -1.0):
        ang = s * phi
        phi0 = np.arctan2(np.mean(np.sin(theta_true - ang)),
                          np.mean(np.cos(theta_true - ang)))
        score = float(np.mean(np.cos(theta_true - (ang + phi0))))
        best = max(best, score)
    return best


def scalar_cover_curve(x, mu, V, atom_counts):
    """Emulate a single-direction (k=1 TopK) SAE restricted to the signal 2-plane:
    a dictionary of M unit directions spread around the circle; reconstruct each point
    with its single best-aligned atom. Measures FVU vs number of atoms -> covering penalty."""
    code = (x - mu) @ V                      # (n,2) coordinates in the plane
    var_plane = np.sum((code - code.mean(0)) ** 2)  # variance captured by the plane
    total_var = np.sum((x - mu) ** 2)
    out = {}
    for M in atom_counts:
        ang = np.linspace(0, 2 * np.pi, M, endpoint=False)
        D = np.stack([np.cos(ang), np.sin(ang)], axis=1)   # (M,2) unit atoms in-plane
        proj = code @ D.T                                   # (n,M) coeff along each atom
        j = np.argmax(np.abs(proj), axis=1)                 # pick best single atom (k=1)
        coeff = proj[np.arange(code.shape[0]), j]
        rec_plane = coeff[:, None] * D[j]                   # (n,2) reconstruction in-plane
        resid_plane = np.sum((code - rec_plane) ** 2)
        # FVU over the full ambient space = (in-plane residual + off-plane variance)/total
        off_plane = total_var - var_plane
        out[int(M)] = float((resid_plane + off_plane) / total_var)
    return out


def atoms_to_reach(curve, eps):
    for M in sorted(curve):
        if curve[M] <= eps:
            return M
    return None


# ----------------------------- experiment -----------------------------
def run_circle(noise=0.05, radial_noise=0.0):
    x, t, _ = plant_circle(noise=noise, radial_noise=radial_noise)
    mu, V = pca_subspace(x, 2)

    # (a) curved (unit-circle code) vs flat (free 2-D block), MATCHED dim/bits
    x_flat, code = reconstruct_subspace(x, mu, V, unit_circle=False)
    x_curv, _ = reconstruct_subspace(x, mu, V, unit_circle=True)
    fvu_flat = fvu(x, x_flat)
    fvu_curv = fvu(x, x_curv)

    # (c) identifiability up to isometry
    corr = circular_corr_up_to_isometry(t, code)

    # (b) scalar covering-number penalty
    atom_counts = [2, 4, 8, 16, 32, 64, 128, 256]
    cover = scalar_cover_curve(x, mu, V, atom_counts)
    eps = round(float(max(fvu_flat, 0.02)) + 0.03, 4)  # tolerance just above the block's floor
    scalar_atoms = atoms_to_reach(cover, eps)

    return {
        "noise": noise, "radial_noise": radial_noise,
        "fvu_flat_block_2d": round(fvu_flat, 4),
        "fvu_curved_unitcircle_2d": round(fvu_curv, 4),
        "curved_minus_flat": round(fvu_curv - fvu_flat, 4),
        "angle_recovery_corr_up_to_isometry": round(corr, 4),
        "scalar_cover_curve_fvu_vs_atoms": cover,
        "tolerance_eps": eps,
        "scalar_atoms_to_reach_eps": scalar_atoms,
        "block_atoms_to_reach_eps": 1,
    }


def run_torus(noise=0.05):
    x, ang, _ = plant_torus(noise=noise)
    mu, V = pca_subspace(x, 4)
    x_hat, code = reconstruct_subspace(x, mu, V, unit_circle=False)
    c1 = circular_corr_up_to_isometry(ang[:, 0], code[:, 0:2])
    c2 = circular_corr_up_to_isometry(ang[:, 1], code[:, 2:4])
    return {"noise": noise, "fvu_flat_block_4d": round(fvu(x, x_hat), 4),
            "angle1_recovery_corr": round(c1, 4), "angle2_recovery_corr": round(c2, 4)}


def verdict(circle_clean):
    tie = abs(circle_clean["curved_minus_flat"]) < 0.02
    ident = circle_clean["angle_recovery_corr_up_to_isometry"] > 0.95
    penalty = (circle_clean["scalar_atoms_to_reach_eps"] or 999) > 2
    lines = []
    lines.append("CURVED-vs-FLAT at matched dim: " +
                 ("TIE (as predicted) -> curved geometry is a CONTROL, not the headline"
                  if tie else "NOT a tie -> investigate (unexpected)"))
    lines.append("IDENTIFIABILITY up to isometry: " +
                 ("PASS (angle recovered, corr>0.95)" if ident else "FAIL"))
    lines.append("SCALAR covering-number penalty: " +
                 ("PRESENT (scalar needs many atoms vs 1 block)" if penalty else "absent"))
    go = tie and ident and penalty
    lines.append("GATE G0: " + ("GO -> commit to identity+cause+truth-vs-confidence; "
                                 "drop curved-dictionary machinery as a headline"
                                 if go else "REVIEW -> result deviates from prediction"))
    return go, lines


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results = {
        "circle_clean": run_circle(noise=0.05, radial_noise=0.0),
        "circle_radial_noise": run_circle(noise=0.05, radial_noise=0.15),
        "torus_clean": run_torus(noise=0.05),
    }
    go, lines = verdict(results["circle_clean"])
    results["gate_G0_go"] = go
    results["verdict"] = lines

    out = os.path.join(here, "phase0_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 72)
    print("PHASE 0 — synthetic calibration / geometry kill-test")
    print("=" * 72)
    cc = results["circle_clean"]
    print(f"[circle, clean]  FVU flat-2D-block = {cc['fvu_flat_block_2d']}   "
          f"FVU curved-unit-circle = {cc['fvu_curved_unitcircle_2d']}   "
          f"(curved - flat = {cc['curved_minus_flat']})")
    print(f"[circle, clean]  angle recovery up-to-isometry corr = "
          f"{cc['angle_recovery_corr_up_to_isometry']}")
    print(f"[circle, clean]  scalar atoms to reach FVU<= {cc['tolerance_eps']}: "
          f"{cc['scalar_atoms_to_reach_eps']}  vs  1 block")
    print(f"[circle, clean]  scalar FVU-vs-atoms = {cc['scalar_cover_curve_fvu_vs_atoms']}")
    tc = results["torus_clean"]
    print(f"[torus,  clean]  FVU flat-4D-block = {tc['fvu_flat_block_4d']}   "
          f"angle1 corr = {tc['angle1_recovery_corr']}   angle2 corr = {tc['angle2_recovery_corr']}")
    print("-" * 72)
    for ln in lines:
        print("  " + ln)
    print("-" * 72)
    print(f"results written to: {out}")


if __name__ == "__main__":
    main()
