# Epistemic Geometry
### *Where Structure Lives in Protein Language Models*

Layer-resolved interpretability of biological foundation models.

**Locating *where* in a protein foundation model structural geometry, dynamics, and function are
represented — testing whether the model *causally uses* those representations, and how they separate from the
model's own confidence.**

Biological foundation models (protein language models; co-folding / structure models) hold, inside their
per-layer activations, quantities whose true values are known from physics and chemistry — a backbone dihedral
angle is a point on a circle, two of them a point on a torus, the twist of a helix advances at a fixed rate,
and the binding strength of two molecules has a laboratory-measured value. This project uses that exact
external ground truth to ask, **layer by layer**:

1. **Where does it live, and is it really there?** Recover a physically-defined quantity as an *identifiable,
   gauge-free coordinate* and show it matches the physical truth *up to isometry* on held-out data — not merely
   that some subspace reconstructs the activation.
2. **Does the model use it?** Change the coordinate inside the model and test whether the output moves the
   matching way, including a wrap-around test (returning to baseline after a full turn of an angle) that a
   single-direction feature cannot fake.
3. **Truth vs confidence.** Show the physical quantity is stored in a direction *distinct from the model's own
   confidence* (pLDDT; interface scores), and that a *trust residual* read off the truth coordinate flags the
   model's confidently-wrong outputs.

The result is a **layer-resolved map** of which depths encode structural geometry / dynamics / function, which
depths the model actually acts on, and where its internal "what is true" diverges from its internal "how sure."

## Status

Early-stage public scaffold. **Phase 0** (a pure-numpy calibration test) is implemented and passing; the later
phases are placeholders. The project's detailed research plan and the paper draft are kept **internal and are
not included in this repository**. Released under the MIT License (see `LICENSE`).

## Repository layout

```
.
├── README.md                   # this file
├── pyproject.toml · requirements.txt · .gitignore
├── src/epistemic_geometry/     # the library (import package name: epistemic_geometry)
│   ├── data/                   # activation caching; physical ground truth (DSSP φ/ψ, SASA; experimental Kd)
│   ├── dictionaries/           # scalar TopK SAE, Euclidean block, circular S¹ / torus T² blocks (curved = control)
│   ├── probes/                 # linear / MLP probes; the confidence probe
│   ├── interventions/          # wrap-around geodesic steering; the persistence / commitment measure
│   ├── metrics/                # circular/toroidal correlation, identifiability-up-to-isometry, Betti, trust residual
│   ├── controls/               # nuisance-only baseline, random-init, matched-bits, leakage probe, validity filter
│   └── viz/                    # figures
├── experiments/
│   ├── phase0_synthetic/       # DONE — pure-numpy calibration / geometry kill-test (passing)
│   ├── phase1_persistence/     # do mid-computation edits survive generation? (placeholder)
│   ├── phase2_identity_truthconf/  # identity up to isometry + truth-vs-confidence (placeholder)
│   ├── phase3_causal_wraparound/   # wrap-around causal dose-response (placeholder)
│   └── phase4_synthesis/       # cross-variable / cross-model synthesis (placeholder)
├── configs/                    # per-experiment YAML configs
├── data/                       # raw/ and activations/ are git-ignored; splits/ holds frozen hashed splits
├── results/                    # figures/, tables/, logs/
├── notebooks/ · tests/ · docs/
```

## Quick start

Phase 0 needs only numpy:

```bash
python3 experiments/phase0_synthetic/phase0_synthetic.py
```

It checks the project's framing on planted data with known ground truth: a curved code **ties** a flat block of
equal dimension at matched information budget (so geometry is a *control*, not the headline), the planted angle
is **recovered up to isometry** (correlation ≈ 0.999), and single-direction codes pay a **covering-number
penalty** (~16 features to match one 2-D block). Numbers are in `experiments/phase0_synthetic/phase0_results.json`.

## Scope and responsible-use note

The protein (structural-geometry) track is the core. A binding-affinity track is **probe-only** here and is
treated as an auditing / calibration study: it reads internal representations and compares them against
laboratory binding values and the model's own interface-confidence score, and it includes **no steering toward
stronger binding** and no generative-design pipeline. Work is restricted to benign complexes.

_Generated with assistance from Claude Code. Empirical claims in this repository should be reproduced from the
code; the synthetic Phase 0 result is the only finding currently in hand._
