# Research plan — interpretability of biological foundation models

This directory holds the research dossier for a mechanistic-interpretability project on
biological foundation models (protein and genomic/structure models). It was produced through
several rounds of independent literature search, method identification, critique, and planning.

## Start here
- **`MASTER_PLAN.md`** — the current, owned, step-by-step plan. Read this first.
- **`primer_bsf_jacobian_landscape.md`** — plain-language primer: SAE, BSF, Jacobian lens, J++, and
  the current landscape of bio-model interpretability. Written for a newcomer.
- **`PROJECT_BRIEF.md`** — the one-page definition the plan is built on.

## The idea, in one line
On quantities fixed by physics (backbone (phi,psi) and helix phase in ESM3 from DSSP; laboratory Kd
in a co-folding model), test whether a biological foundation model represents the quantity as an
**identifiable, gauge-free coordinate** that (1) matches physical truth up to isometry, (2) causally
drives the output with a wrap-around test a single-direction feature cannot fake, and (3) lives in a
**different direction from the model's own confidence** (pLDDT; ipSAE). Headline = *truth vs confidence*.
"A fancier dictionary geometry" is explicitly **not** the contribution (prior work closed that).

## Supporting documents
- `FINAL_FINDINGS.md` — verified landscape + reading list + model availability + gaps.
- `PITCH_VETTING_consolidated.md` — four candidate ideas, each checked for already-done / disproven.
- `PLAN_ICML_synthesis.md` — the ICML-bar synthesis that preceded the current plan.
- `evidence_base.md`, `technical_plan.md`, `my_pitches.md` — earlier consolidations (kept for provenance).
- `appendix/` — the independent agent reports (deep searches, per-track plans, per-pitch vettings,
  verification passes). These are working notes; where they disagree, `MASTER_PLAN.md` is authoritative.

## Experiments
- `../../experiments/phase0_synthetic/` — Phase 0: a pure-numpy synthetic calibration / "geometry
  kill-test" (no model weights, no network). Run: `python3 experiments/phase0_synthetic/phase0_synthetic.py`.
  First result (`phase0_results.json`): on a planted circle, a curved (unit-circle) code **ties** a flat
  Euclidean block of equal dimension at matched bits (confirming geometry is a control, not the headline);
  the angle is recovered up to isometry (corr ~0.999); single-direction codes pay a covering-number penalty
  (~16 atoms vs 1 block). Honest caveat: on a planted **torus**, naive PCA+atan2 does **not** separate the
  two circular factors (corr ~0.60), so the real (phi,psi) torus identity work needs a genuine factorization
  method, not PCA — recorded as a known issue for Phase 2.

## Status
Decisions are made in `MASTER_PLAN.md` (Section 0). Immediate next action: secure ESM3-open weight access
and begin Phase 1 (persistence/commitment gate) on the protein track. Several facts are marked UNVERIFIED
in the plan (weight terms, dataset counts/licences, the LLM truth-vs-confidence prior art) and need a human
pass before write-up. The dual-use statement for the binding track owes a human/compliance review.

_Provenance: this dossier was generated with Claude Code. Treat external claims as UNVERIFIED unless a
primary source is cited._
