# Epistemic Geometry in Biological Foundation Models

**What is true vs how sure it is: identifiable, causal epistemic geometry in biological foundation models.**

This project tests whether a biological foundation model represents a physically-defined quantity (backbone
dihedral angles / helix phase in a protein language model; experimental binding strength Kd in a co-folding
model) as an **identifiable, gauge-free coordinate** that (1) matches the physical truth up to isometry on
held-out data, (2) **causally** drives the model's output with a wrap-around test a single-direction feature
cannot fake, and (3) lives in a direction **distinct from the model's own confidence** (pLDDT; ipSAE) — with a
*trust residual* that flags the model's confidently-wrong outputs.

The full plan, claims, protocol, controls, figures, and timeline are in **[`ICML2027_CORE.md`](./ICML2027_CORE.md)**.

> ## ⚠️ Availability / status
> **This code and its data are PRIVATE for now.** The repository, source, cached activations, and datasets are
> not publicly available at this time. **The code will be made public in a while** — a public release (source,
> the physical-ground-truth benchmark suite, and reproduction scripts) is planned to accompany the paper. Until
> then, please do not redistribute. Access and licensing for any pre-release use are at the authors' discretion.
>
> _This is a status notice, not a license. A formal open-source license will be added at public release._

## Repository layout

```
epistemic-geometry/
├── ICML2027_CORE.md            # THE core project document (read this first)
├── README.md                   # this file
├── pyproject.toml              # package metadata (src layout; package: epistemic_geometry)
├── requirements.txt            # dependencies (Phase 0 needs only numpy; later phases add the rest)
├── .gitignore
├── src/epistemic_geometry/     # the library
│   ├── data/                   # activation caching (ESM3 / Boltz); physical ground truth (DSSP φ/ψ, SASA; Kd)
│   ├── dictionaries/           # scalar TopK SAE, Euclidean BSF, circular S¹ / torus T² blocks (curved = CONTROL)
│   ├── probes/                 # linear / MLP probes; the confidence probe (pLDDT / entropy / ipSAE)
│   ├── interventions/          # wrap-around geodesic steering; the persistence / commitment measure
│   ├── metrics/                # circular/toroidal corr, identifiability-up-to-isometry, Betti, trust residual
│   ├── controls/               # nuisance-only baseline, random-init, matched-bits/MDL, leakage probe, validity filter
│   └── viz/                    # figures (money figure = trust-residual / confidently-wrong)
├── experiments/
│   ├── phase0_synthetic/       # DONE — pure-numpy calibration / geometry kill-test (passing)
│   ├── phase1_persistence/     # persistence / commitment gate (the most-likely death; test first)
│   ├── phase2_identity_truthconf/  # identity up to isometry + truth-vs-confidence (the core)
│   ├── phase3_causal_wraparound/   # wrap-around causal dose-response + RoPE-proof validator
│   └── phase4_synthesis/       # cross-variable/model invariance, full sweep, write-up
├── configs/                    # per-experiment YAML configs
├── data/                       # raw/ and activations/ are git-ignored; splits/ holds frozen hashed splits
├── results/                    # figures/, tables/, logs/
├── notebooks/
├── tests/
└── docs/
```

## Quick start

Phase 0 is implemented and needs only numpy:

```bash
python3 experiments/phase0_synthetic/phase0_synthetic.py
```

It confirms the project's framing: a curved code ties a flat block of equal dimension at matched bits (so
geometry is a *control*, not the contribution), the angle is recovered up to isometry, and single-direction
codes pay a covering-number penalty. See `ICML2027_CORE.md` Appendix B for the numbers.

## Scope note

Target venue: ICML 2027 main track (upside), NeurIPS 2027 main / Datasets & Benchmarks (honest fallback). The
protein track is the spine; the binding (Kd / ipSAE) track is **probe-only** in this submission and is
dual-use-sensitive — handled as an auditing / calibration finding, with no steering and no misuse content, and
a dual-use statement that still owes a human / compliance review (see `ICML2027_CORE.md` §15).

_Provenance: the full planning trail (literature survey, idea vetting, independent expert plans) lives in the
parent repository under `docs/research-plan/`. Generated with Claude Code; external claims are UNVERIFIED unless
a primary source is cited in the core document._
