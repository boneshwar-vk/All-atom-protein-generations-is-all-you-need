# CORE PROJECT DOCUMENT — "What Is True vs How Sure It Is: Identifiable, Causal Epistemic Geometry in Biological Foundation Models"

**The single document the project is run from. Target: ICML 2027 main track (upside); NeurIPS 2027 main / Datasets & Benchmarks (honest fallback).**

> **Status of this file.** This is the **core project document**. It **supersedes `MASTER_PLAN.md` and `PROJECT_DOSSIER.md`** (both retained in `docs/research-plan/` as the provenance record, along with `FINAL_FINDINGS.md`, `PITCH_VETTING_consolidated.md`, `PLAN_ICML_synthesis.md`, `PROJECT_BRIEF.md`, `primer_bsf_jacobian_landscape.md`, `technical_plan.md`, `evidence_base.md`, and `appendix/`). Where this document conflicts with the prior planning files, **this document wins**; those files remain the trail of how the plan was reached.
>
> **How it was built.** Merged, 2026-11-22, from **two independent expert passes** over the same prior material: (i) an **architect** pass (ICML-author / mech-interp + representation-geometry + structural-biology role) that supplies the structural backbone, the full method, and the per-phase protocol; and (ii) an **adversarial reviewer** pass (written from the Area Chair's chair, main-track-accept standard) that supplies four simulated reviews + an AC meta-review, the pre-registered-hypothesis table, the full controls battery, and the honest limitations. The two passes agree on the kernel and the gates; where they differed, the reconciliation is stated inline and the more conservative/rigorous option was kept (see the merge notes at the end of §0).
>
> **Research/planning only — no git repository was modified in either pass or this merge.**

**Convention.** `[UNVERIFIED]` (equivalently `UNVERIFIED`) = not confirmed against a primary source in this session (network to arxiv/biorxiv/neurips/goodfire was blocked when the plan was built; a few items were confirmed later). Every `[UNVERIFIED]` tag from both source passes is preserved, and the reviewer pass's additional `UNVERIFIED` flags (where the plan asserted more than the evidence supports) are carried through. `[CONCEDE]` = prior work owns this; we cite, we do not compete. **No citations are invented**; cite-keys (Ste26, Jed26b, Zha26g, Hin26, Dal26/SASA, Mig26/PairSAE, Set26, …) are the project's own and trace to `FINAL_FINDINGS.md` / `appendix/`. Benign/defensive framing throughout; the binding track is dual-use-sensitive and is handled as **auditing / calibration only**, with no misuse guidance, no steering in this submission, and an explicit **human/compliance pass owed** (§15).

**Reported ICML 2027 timeline (verified-this-session for the architect pass, but `[UNVERIFIED]` against the official call, which was not yet posted as of early Nov 2026):** abstract deadline ~**14–16 Jan 2027**, full-paper deadline ~**22 Jan 2027** (both AoE, most-consistently-reported); conference **31 Jul – 6 Aug 2027**, Rio de Janeiro. New-for-2027 reported rules: a 20-submissions-per-author cap and a two-stage review process `[UNVERIFIED]`. ICLR 2027 workshops ~1 Feb 2027 `[UNVERIFIED]`; NeurIPS 2027 abstract ~mid-May 2027 `[UNVERIFIED]`. **From 2026-11-22 the full-paper deadline is ~8.5–9 weeks away** (the two passes quoted ~8.5 and ~9; both `[UNVERIFIED]`). This number drives the entire re-scope in §0 and the timeline in §11. **All venue dates owe a human a primary-source pass (§15).**

---

## Table of contents

0. What this document changes, and why (read this first)
1. Title, abstract, contribution statement
2. The problem, and why ICML (not just a bio venue) cares
3. The method, in full
4. Claims C1–C6, restated precisely and falsifiably
5. The six claims as pre-registered falsifiable hypotheses
6. The controls battery (mandatory vs nice-to-have, with the reviewer who demands each)
7. Simulated reviews and responses (four reviewers + the AC meta-review)
8. Full experimental protocol — per phase, per track
9. Related work and positioning
10. Figure list and the money figure
11. Risk-managed critical path, backward timeline, and the pivot decision point
12. Scope discipline and the Minimum Publishable Result
13. Reproducibility, compute budget, and the data / leakage-audit protocol
14. Honest limitations and the rebuttal that cannot yet be closed
15. Ethics / dual-use statement, and the owed human / compliance pass
- Appendix A — Decision / gate ladder (one screen)
- Appendix B — Phase 0 numbers as a control result (already in hand)
- Appendix C — What stays exactly as the prior plan had it
- Appendix D — One-line prior-art out-positioning ledger

---

## 0. WHAT THIS DOCUMENT CHANGES, AND WHY (read this first)

The surviving kernel the prior vetting rounds converged on — **identity + cause + truth-vs-confidence on exact physical ground truth** — is genuinely good and genuinely un-scooped *as a conjunction*. Both expert passes keep it unchanged. But the plan as previously written was mis-scoped for ICML 2027 on several axes, and the headline claim had one unresolved validity threat that, left as-is, is a reject. The substantive changes imposed, on which both passes agree:

1. **Re-scoped the ICML 2027 submission to a realistic core, stated as a staged target rather than a hedge.** The full six-claim, two-track, five-phase program cannot be executed to main-track standard in ~8.5–9 weeks, with ESM3-open gated-weight access *not yet secured* and Track B (binding) being the leakage-heaviest, most compute-expensive, and only dual-use-sensitive piece. The prior docs themselves concluded "ICML 2027 in ~9 weeks is not realistic" (`FINAL_FINDINGS §6`) and three independent prior passes in `appendix/` reached a ~15% acceptance figure, then the later docs aimed at ICML anyway without resolving the contradiction. We resolve it: **the ICML 2027 submission is Track A (ESM3-open) carrying C1–C5, plus a probe-only Track-B truth≠confidence vignette (C3-B) and one "confidently wrong" example; the full Track-B causal program (mid-diffusion steering, SKEMPI, Feynman–Kac) is explicitly deferred to the camera-ready/appendix or a NeurIPS 2027 follow-up.** A hard **decision gate at ~19 Dec 2026 (mid-December)** decides ICML-submit vs NeurIPS-2027-retarget. Ambition is undiminished (the headline is unchanged); the *surface area* is cut to what can be made airtight.

2. **Rewrote the headline around the LLM-impossible conjunction, not the bare separation.** "A biological model stores truth in a different direction from confidence" is **not**, by itself, a defensible main-track contribution in 2027 — the 2026 LLM truth/confidence wave (2603.25052; "Confidence Manifold" 2602.08159; "Truthfulness Spectrum" 2602.20273; ACL-2026 "LLM Beliefs Are in Their Heads" — **all `[UNVERIFIED]` beyond abstracts**) already asserts near-orthogonal truth/confidence directions. The load-bearing contribution is the **conjunction LLM work structurally cannot have**: *external physical ground truth* (DSSP dihedral, calorimetric Kd) + a *deployed, calibrated confidence head* (pLDDT/ipSAE) + a *causal double dissociation + wrap-around* + *identifiability up to isometry* + *cross-track invariance*. The title claim is that conjunction, tested as one object.

3. **Promoted the headline's single unresolved validity threat to a pre-registered, must-pass gate, and sharpened the claim.** The strongest reviewer rebuttal — "pLDDT is high exactly where backbone geometry is regular, so (φ,ψ) and pLDDT are entangled in the data; your 'separation' is two heads on one correlated latent" — is **not closed on paper** (§14.1). The **confidence-matched negative-pair control + the confidently-wrong slice** is the **gate that decides whether the hook is even claimed** (gate G5). The claim is retargeted away from the near-trivial form ("truth and confidence point in different directions" — almost tautological for two different quantities) toward the **strong, useful form: the recovered truth coordinate predicts ground-truth physics on the model's own confidently-wrong cases better than the model's confidence head does** (the "trust residual"; §3.6, C3). That version is non-trivial, falsifiable, and exports as an auditing primitive.

4. **Demoted Track B from a co-equal track to a probe-only extension + one vignette for this submission**, on three grounds: (i) it is the only dual-use-sensitive component and its dual-use statement still owes a human pass; (ii) it is leakage-fragile (the whole track dies if G-B1 fails) and the data engineering is the plan's acknowledged top effort-sink; (iii) its expensive causal half adds 1.5–3k GPU-h and a model-family switch for a result that is *not load-bearing for acceptance*. **The paper must be acceptable on Track A alone (C1–C5 on ESM3).** Track B earns a place in the main body only after **G-B1 (Kd is learning, not leakage)** and **G-B2 (Kd direction ≠ ipSAE direction)** both clear; then it supplies C6 (cross-track invariance) and the single most quotable confidently-wrong instance. If either gate fails, Track B is a one-figure leakage/Goodhart **negative** in an appendix.

5. **Killed the theorem framing; replaced it with modest, defensible, *empirical* identifiability.** An earlier doc tried to make this a theory paper with a separation theorem (T1/T2/T3). The vetting killed that: **SASA/Dal26 already owns the covering-number theorem, the atoms-vs-covering curve, and the O(1) block remedy** `[CONCEDE]`, and Phase 0 confirmed a flat block ties a curved one at matched bits. The theory is stated as exactly what we can defend (§3.7): a *concession* to SASA for capacity, a *gauge-freeness* argument for why the winding coordinate is the identifiable object, and an *operational* (not asymptotic) identifiability claim validated on planted data. No new theorem is claimed.

6. **Fixed the geometry framing to a control, and went further than the original plan.** Phase 0 shows curved *ties* flat on a clean circle (ΔFVU=0.0022) and curved *loses* once the radius varies (ΔFVU=+0.0213, flat better). So we use **no learnable-curvature machinery at all** (Skopek ill-conditioning + Phase 0 radial-noise penalty); the circular/torus block is one *fixed/assigned* parameterization used as a *tool* inside the identity analysis, judged by identifiability, never by "beats flat." Every headline is reported at **matched bits/MDL**, never matched feature count. The word "geometry" must never appear in the contribution sentence. This removes the single largest scoop exposure and is non-negotiable.

7. **Resolved the Phase-0 torus-factorization caveat instead of leaving it open.** Phase 0 showed naive PCA recovers the 4-D torus subspace (FVU 0.07) but does **not** separate the two circular factors (per-angle corr ~0.60). For the **identity** claim we have per-residue ground-truth φ and ψ, so we never need unsupervised factorization — we align each S¹ factor to its labeled angle on the train split (supervised disentanglement). The unsupervised product-of-circles recovery becomes a *secondary*, gated result (independent-subspace analysis / an MLP disentangler with a total-correlation penalty), not a blocker on the headline (§3.3).

8. **Added the confidence-source audit as a protocol prerequisite.** On *sequence-only* input with the structure track off, it is not obvious ESM3 emits a usable per-residue pLDDT; the confidence contrast may have to use the model's structure-token predictive entropy instead. Flagged as a must-check before C3 (§8), because the whole hook rests on having a *real, deployed* confidence signal.

9. **Made "biology is the instrument, not the application" the explicit ICML positioning** (§2) — the inversion that converts a structural-biology result into an ICML main-track result.

10. **Converted the dual-use posture into an enforced-in-scope boundary for this submission.** For ICML 2027, Track B ships **no steering at all** — only probe-level direction-angle measurements on benign, leakage-split complexes and one read-only confidently-wrong example. The affinity-steering machinery that is the actual hazard is not built, not run, and not released in this version (§15).

11. **Wrote a Minimum Publishable Result reachable from the cheap gates alone** (§12), so a null on the causal half still yields an ICML-or-NeurIPS paper, and stated the realistic venue ladder honestly.

12. **Pre-registered everything before any GPU run** (§5), with numeric thresholds, the exact null, the exact statistical test, the multiple-comparison correction, and the result that refutes. An interpretability paper at this boundary in 2027 that is not pre-registered against these specific confounds (RoPE, leakage, off-manifold collapse, annotation circularity, confidence entanglement) is desk-reject-adjacent. **The controls are not a section; they are the paper** (§6).

13. **Fully cut the Jacobian/J++ transport half** (locked; §12) and **cut the genomic generality panel** (codon-frame is owned by two NeurIPS-2026 papers; cross-track invariance is Track A structural vs Track B thermodynamic, not a third modality).

Everything else from the vetting — curved-as-control, matched-bits/MDL, the nuisance-only baseline, the RoPE-proof phase-slip, the kill-gate ordering, the leakage-aware splits, the honest-negative-as-deliverable stance — is kept and in places sharpened.

**Honest acceptance odds (AC pass).** As the plan stood before this hardening: **~15–20%, ICML 2027 main track.** Not because the kernel is weak, but because (a) the single most likely experimental outcome kills the causal half (persistence/commitment wash-out, two papers forecast it), (b) the headline as written is simultaneously scoop-exposed and vulnerable to the entanglement rebuttal it cannot yet answer, and (c) ~8.5–9 weeks is short for gated GPU campaigns with ≥5 seeds, two model families, and a validity pipeline. This document is written to move that number up *without* lowering the ambition — by hardening the claims, pre-registering the thresholds, and defining the exact point at which the project ships the strong-negative/benchmark paper or slides to NeurIPS 2027 rather than missing both. Done right, the same work is a ~35–45% NeurIPS paper plus a near-certain workshop/D&B result; a rushed ICML submission that drops the controls is the ~15% paper.

**Merge reconciliations (where the two passes genuinely differed, the conservative choice was kept):**
- *Beat-null margin for C1.* Architect wrote "trained−random gap > 0.1"; reviewer wrote "beats random-init/nuisance/scalar by ≥0.15." **Reconciled to ≥0.15** (the more conservative margin) as the pre-registered threshold (§5, C1).
- *Torus topology.* Architect pre-registered the idealized torus Betti (1,2,1); reviewer noted Ramachandran is **non-uniform / basin-structured T²**, not a uniform torus. **Reconciled to the reviewer's more careful statement**: (1,2,1) is the connectivity signature of the underlying T², but the *occupied density* is basin-structured and is what is pre-registered as the realized topology (§3.3, §5 C1).
- *Decision-gate date and criteria.* Architect set a ~19 Dec gate on {C1 + G5}; reviewer set an end-of-Week-5 gate on {G1 + G2 + G-B2}. **Reconciled to a single mid-December (~19 Dec) decision gate evaluating {G1, G2, C1, G5} and the Track-B gates** — the earlier date and the union of both criteria (§11), consistent with the brief's "mid-December decision gate."
- *Deadline wall-clock.* "~8.5 weeks" (architect) vs "~9 weeks" (reviewer); both `[UNVERIFIED]`. **Stated as ~8.5–9 weeks.**

---

## 1. TITLE, ABSTRACT, CONTRIBUTION STATEMENT

### 1.1 Title options
Primary (recommended):
> **"What Is True vs How Sure It Is: Recovering an Identifiable, Causal Truth Coordinate Distinct from Confidence in a Protein Foundation Model."**

Alternatives, by emphasis:
- (epistemics-forward) **"Truth ⟂ Confidence: A Physically-Grounded, Causal Test of the Belief–Confidence Split in Foundation Models."**
- (identifiability-forward) **"Gauge-Free Recovery of Physical Coordinates in Protein Language Models: Identity up to Isometry, Causal Wrap-Around, and a Trust Axis."**
- (auditing-forward) **"A Trust Axis for Generative Biology: Reading a Model's Internal Physics Against Its Own Confidence."**
- (mechanism-forward, risk: reads too bio) **"The Ramachandran Torus Inside ESM3: Identifiable, Steerable, and Separate from the Model's Confidence."**

Recommendation: lead with the primary title. It states the ICML-general question (truth vs confidence in a foundation model) first and the biological instrument second — matching the positioning in §2.

### 1.2 Abstract (≈150 words)
> Do foundation models keep *what is true* in a different place from *how sure they are*? Language-model interpretability cannot answer this cleanly: it has no external referent for "true" and no deployed confidence head. Biology does. We study a protein language model (ESM3-open) on quantities fixed by physics — backbone dihedral angles and helix phase, computed from experimental structures, never from database labels. We show the model holds such a quantity as an **identifiable, gauge-free coordinate** that (i) matches the physical truth *up to isometry* on held-out proteins, (ii) **causally** drives the model's own output with a wrap-around dose-response that returns to baseline after a full turn — a signature a single-direction feature structurally cannot produce — and (iii) lies in a direction **distinct from the model's calibrated confidence**, such that on confidently-wrong cases the truth coordinate still tracks reality. We release the physical-ground-truth suite, the RoPE-proof protocol, and a trust-axis auditing primitive. We report honestly where mid-computation edits wash out.

### 1.3 One-paragraph contribution statement
> We contribute the first **causal + identifiable + truth-vs-confidence** analysis of a biological foundation model on quantities whose true value is fixed *externally* by physics. Concretely: (1) we recover a backbone-angle / helix-phase coordinate from ESM3-open activations and show, on a held-out split with the alignment fit on train only, that it equals the physical coordinate *up to the quantity's isometry group* — an identity claim, not the reconstruction claim prior dictionary-learning work makes; (2) we show the model *uses* it, via a wrap-around dose-response whose return-to-baseline at 2π is a causal signature a scalar latent cannot fake; (3) we show the truth coordinate is geometrically and causally separable from the model's own confidence head, and — the useful form — that a *trust residual* read off the truth axis flags the model's confidently-wrong outputs better than the confidence head alone, exporting as an auditing primitive that is model- and domain-general. We treat curved/block dictionary geometry strictly as a *control* (Phase 0 confirms it ties a flat block of equal dimension at matched bits; SASA owns capacity), release a reusable exact-geometry benchmark with a RoPE-proof positional protocol, and report the honest negative where mid-computation edits are erased by iterative decoding. A probe-level binding vignette (experimental Kd vs the model's interface confidence ipSAE, on leakage-aware splits) shows the separation generalizes to a scalar thermodynamic variable in a second model family.

---

## 2. THE PROBLEM, AND WHY ICML (NOT JUST A BIO VENUE) CARES

### 2.1 The precise problem
A foundation model, asked to produce an output, carries internally both (a) an estimate of a *world quantity* (for a protein model: the local backbone geometry it "thinks" the sequence adopts) and (b) an estimate of *its own reliability* about that quantity (a confidence head). Three questions, which no prior work answers *together* for a quantity with external ground truth:

- **Identity.** Is the internal feature we recover *that physical quantity* — the actual angle on the circle, the actual point on the torus — recovered as a gauge-free coordinate up to an unimportant choice of axes (isometry), rather than merely *some* subspace that reconstructs the activation? Reconstruction ≠ identity.
- **Cause.** If we move that coordinate inside the model, does the model's own output move the matching way, including returning to where it started after a full 2π turn for an angular quantity? Decodable ≠ used.
- **Truth vs confidence.** Does the model keep the real physical quantity in a *different internal direction* from its self-confidence about that quantity — and, the sharp version, does the truth coordinate still know the answer precisely on the cases where the confidence head is high and wrong?

### 2.2 Why this is an ICML problem, not a bio problem
**The question is a general ML question; biology is the one bench that can answer it cleanly.** In 2026 the LLM interpretability community is actively arguing that models maintain an internal correctness/belief axis separate from expressed confidence (the 2026 "truth-vs-confidence" wave — see §9). That program has two structural handicaps it cannot remove from inside language:

1. **No external referent for "truth".** LLM truth-direction work (Azaria–Mitchell; Burns CCS; Marks–Tegmark; and the 2026 wave) must operationalize "truth" as *the model's own answer-correctness within its own domain*. A probed "truth direction" may be reading the model's belief, not the world. (The reported "truth is not a single direction" results are a symptom of this circularity.)
2. **No deployed, calibrated confidence term.** LLM "confidence" is elicited (verbalized) or probed post hoc.

Biology removes both handicaps at once:
- **External physical ground truth.** A backbone dihedral comes from DSSP on an experimental structure; a binding constant comes from a calorimeter. "Truth" is pinned *outside* the model and outside its narration.
- **A deployed, calibrated confidence head.** Protein/structure models *ship* pLDDT / ipSAE and practitioners use them to decide whether to trust a prediction. "Confidence" is a real, externally-meaningful, deployed quantity.
- **Causality and identifiability are available.** Because the true coordinate is known and computable, we can (a) test identity *up to isometry* against it, and (b) run a *causal* double dissociation and a wrap-around test — the two dimensions the LLM program structurally lacks.

So the inversion ICML rewards: **biology is the instrument, not the application.** We take a contested, live ML question and answer it causally, with external ground truth, in the one domain where both terms are pinned independently — and we export the result (the trust axis) back to the general setting as an auditing primitive. The protein torus is the apparatus; the finding is about the epistemic structure of foundation models.

### 2.3 Why it also matters concretely (the defensive payoff)
If a generative design model's internal *truth* coordinate can be read independently of its *gameable* confidence head, a deployer gets a monitor that flags "confidently wrong" outputs by their internal disagreement — a calibration-auditing primitive that does not trust the model's self-report. That is the dual-use-positive artifact (§15) and the reason the separation is worth establishing beyond curiosity.

---

## 3. THE METHOD, IN FULL

Notation. For input (sequence) `s`, the model produces a per-position residual-stream activation `h_ℓ(s, i) ∈ R^d` at layer `ℓ`, position `i` (`d ≈ 1536` for ESM3-open `[UNVERIFIED exact]`). A physical quantity `y(s,i)` is computed by deterministic external code from the experimental structure of `s` (DSSP/mdtraj), never from the model. We write the model's confidence signal as `c(s,i)` (pLDDT or structure-token predictive entropy; see §8 caveat).

### 3.1 The gauge-free coordinate (definition)
A **block** `B` of width `b` is a `b`-dimensional subspace of activation space, spanned by `b` directions `D_B = [d_1,…,d_b] ∈ R^{d×b}`. Projecting an activation gives a block code `z_B(s,i) = D_B^⊤ h_ℓ(s,i) ∈ R^b`. For a circular quantity we use `b = 2` and define the **gauge-free coordinate** as the *angle* of the block code:
> `θ̂_B(s,i) = atan2(z_B[2], z_B[1]) ∈ S¹`.

The key property: within a block, only the *subspace* is pinned down, not the axes (any rotation `R ∈ O(b)` of `D_B` spans the same subspace and is an equally valid featurizer — this is BSF's within-block rotation ambiguity). A rotation/reflection of the axes maps `θ̂ ↦ ±θ̂ + φ₀`. Therefore **rotation-invariant functions of the angle — the winding number, and the angle itself up to an `O(2)` reparametrization — are the well-defined, gauge-free objects.** We only ever read gauge-invariant quantities from a block, and we align to ground truth with an explicitly estimated isometry (§3.2). This is why the ambiguity that is a *caveat* for BSF becomes the natural *coordinate* here: the physical quantity (an angle) is exactly the kind of object that is defined up to isometry, so the method and the target have matched symmetry groups.

For a toroidal quantity (φ,ψ) we use a **product of two circular blocks** `B_φ, B_ψ` (total width 4), each read as its own gauge-free angle; the torus coordinate is `(θ̂_φ, θ̂_ψ) ∈ T² = S¹ × S¹` (factorization: §3.3).

### 3.2 Identifiability up to isometry (definition and estimator)
Let `G` be the isometry group of the quantity's ground-truth space: `O(2)` (rotation+reflection) for a circle/helix phase, the torus isometry group (independent `O(2)` on each factor, plus factor-swap for the symmetric case) for (φ,ψ), and the monotone-reparametrization group for a scalar (Kd). We say the recovered coordinate is **identifiable up to `G`** if there exists `g ∈ G`, *estimated on a train split only*, such that `g(θ̂)` matches `y` on a disjoint held-out split within a pre-registered tolerance.

Estimator and metric (circle): fit `g` by the single-parameter alignment `g(θ̂) = s·θ̂ + φ₀`, `s ∈ {+1,−1}`, `φ₀ = atan2(⟨sin(y − sθ̂)⟩, ⟨cos(y − sθ̂)⟩)` on *train*; report on *test* the **circular correlation** (Jammalamadaka–Sarma) and the **geodesic R²** of `g(θ̂)` vs `y`. (This is the exact `circular_corr_up_to_isometry` routine validated in Phase 0, corr = 0.9987 on a planted circle.) For the torus, fit the per-factor alignments after factor assignment (§3.3); for a scalar, fit an isotonic (monotone) regression on train and report held-out Spearman + the residual.

**What this buys us over reconstruction.** A reconstructing subspace (SASA's object) proves a `b`-dim block can *rebuild* the activation to tolerance ε. Identity-up-to-isometry proves the *coordinate inside the block is the physical one* on data the alignment never saw. The difference is the whole point of C1 and is what SASA, BSF, and Setlur do not claim.

**Honest scoping (reviewer-forced).** This is **empirical recovery up to the variable's isometry group on a held-out split with the alignment fit on train only** — *not* a population-identifiability theorem (§3.7, §14). The **falsifier is pre-registered**: if the recovered coordinate requires a **non-isometric** (e.g. non-monotone, many-to-one) warp to match truth, the identity claim fails.

### 3.3 Torus factorization (resolving the Phase-0 caveat)
Phase 0 established that naive PCA recovers the 4-D torus subspace (FVU 0.0695) but **does not separate the two S¹ factors** (per-angle corr 0.6146 / 0.5934): the two circles get rotated into each other within the 4-D subspace. Two tracks:

- **For the identity claim (C1), supervised disentanglement — no unsupervised factorization needed.** We *have* per-residue ground-truth φ and ψ. We fit two circular blocks and *assign/align* each to its labeled angle on the train split (the alignment of §3.2, done per angle, choosing the assignment and sign/offset that maximize train circular correlation). The held-out metric is then per-angle. This sidesteps the PCA failure entirely: the torus is recovered as *two labeled, individually-aligned circles*, which is exactly the identity statement we want.
- **For the secondary unsupervised-recovery result,** where we claim the torus is recoverable *without* labels, we use an explicit factorization: **independent-subspace analysis (ISA)** on the 4-D code, or a small MLP disentangler trained with a **total-correlation penalty** (minimize statistical dependence between the two 2-D factors), judged by per-angle circular correlation and by the pre-registered topology (below). This is a nice-to-have generality result and a **gated deliverable (G0b)**, not a blocker; if it ties the supervised version it strengthens the paper, if it fails it is reported as "unsupervised factor separation of the torus is hard, identity still holds with labels."

**Pre-registered topology (reconciled).** Helix phase: Betti (b0,b1) = (1,1). For (φ,ψ): the underlying connectivity is that of T² (the idealized Betti signature (1,2,1)), but the Ramachandran map is **non-uniform / basin-structured** — the *occupied density* is not a uniform torus. We pre-register the **basin-structured T²** as the realized topology (reviewer's more conservative statement, kept in the merge), so a mismatched realized topology is a refutation, not a reinterpretation; trained−random topology gap > 0.1.

### 3.4 The wrap-around causal operator (definition)
Given an identified circular block `B` at layer `ℓ`, steering by angle Δ is a **geodesic rotation of the block code within its 2-plane**, applied to the residual stream during generation:
> `h_ℓ(s,i) ← h_ℓ(s,i) + D_B · (R_Δ − I) · z_B(s,i)`,  `R_Δ = [[cosΔ, −sinΔ],[sinΔ, cosΔ]]`.

This rotates the recovered coordinate by Δ while leaving the orthogonal complement untouched (a *retraction* on S¹; projected-renormalize is sufficient and simpler than full geoopt exp/log maps — `plan_task1_geometry §3.1`). The **wrap-around causal test**: sweep Δ ∈ {0, 2π/K, …, 2π} (K ≥ 8), read the model's *own* output head (predicted structure token / residue dihedral — **not** a same-variable probe, for readout independence), and measure (a) monotone tracking of the output angle with Δ over [0, π], and (b) **closure**: the output distance back to baseline at Δ = 2π. A scalar (1-D) latent cannot produce closure — adding `α·d` monotonically and does not return — so closure is the signature that distinguishes a genuine cyclic variable from a scalar one. Controls: matched-norm random 2-D direction (no closure, no signed monotone response) and a scalar structure-SAE direction (Steenwyk-style; cannot close by construction); validity filter on every steered output (§3.8).

### 3.5 The truth ⟂ confidence principal-angle / double dissociation (definition)
Let `U_truth` be the subspace carrying the physical coordinate (the identified block / the probe subspace most predictive of held-out `y`), and `U_conf` the direction most predictive of held-out confidence `c`. Three measures, in increasing strength:

1. **Geometric (weak, near-necessary).** Principal angle(s) between `U_truth` and `U_conf`, with a matched-norm random-direction null **and** a two-heads-on-one-latent synthetic null (MPU-0). A positive angle says they are not collinear. *This is the weak form and is close to trivially true for two different quantities — we report it as a screen but do not rest the claim on it.*
2. **Causal double dissociation (strong).** The 2×2: steer `U_truth` → measure {physics output, confidence head}; steer `U_conf` → measure {physics output, confidence head}. The claim is the off-diagonal is ~0: moving truth moves the physics output but not confidence, and moving confidence moves the confidence head but not the physics output. This requires the persistence gate (G2) to have passed. Reported as a 2×2 standardized-effect grid with cluster-bootstrap CIs and an interaction test.
3. **Confidence-matched + confidently-wrong (decisive, the version that matters).** §3.6.

### 3.6 The trust residual and the confidently-wrong test (the sharp claim)
Define the **trust residual** per input as the component of the model's confidence that is *not* predictable from the recovered truth coordinate:
> `r(s,i) = c(s,i) − ĉ(θ̂(s,i))`, where `ĉ` is fit (train only) to predict `c` from the truth coordinate.

Claim and test (the useful, non-trivial form of the hook):
- **Confidence-matched negative pairs.** Construct input pairs matched on the model's confidence bin (same `c`) but differing in true physics (different `y`). Show the truth coordinate `θ̂` separates them (it tracks `y`), while `c` by construction cannot. This *holds confidence fixed and varies truth*, so any separation is truth-specific — the clean refutation of the entanglement rebuttal, and the control run *first* inside the hook (gate G5).
- **Confidently-wrong slice.** On cases where `c` is high but `y` is atypical/unexpected (the model is confident and the physics is unusual), show `θ̂` still tracks `y`. The **ROC claim**: the trust residual `r` (or `θ̂`-vs-`c` disagreement) flags these calibration failures with higher AUROC than `c` alone. This is what makes the separation *useful* and is the money figure (§10).

Why this retargeting matters: "truth and confidence point in different directions" is almost tautological (they are different quantities); a reviewer dismisses it. "The internal truth axis knows the answer on the cases where the deployed confidence head is confidently wrong, and reading it beats reading confidence" is a falsifiable, useful, non-obvious claim about the model's epistemic structure — and it is the claim the LLM wave cannot make without external ground truth. **This is also the single rebuttal that cannot yet be closed on paper (§14.1): the effect must survive the confidence-matched / confidently-wrong slices, and Track A may not supply enough high-confidence × atypical-geometry pairs to power it — a power analysis is pre-registered up front.**

### 3.7 The modest, defensible theory (what is assumed, what would have to be true)
No new theorem. We state exactly three things and credit prior work for the rest:

- **(Concession, cite SASA/Dal26 `[CONCEDE]`.)** To *reconstruct* a compact/periodic manifold to L2 tolerance ε, a single-direction (scalar) sparse dictionary needs Ω(covering-number(M, ε)) atoms, whereas a block of the manifold's dimension reconstructs with O(1). This is SASA's covering-number result; Phase 0 reproduces the scalar covering curve (16 scalar atoms to match 1 two-dim block) and the curved-ties-flat tie (FVU 0.1361 vs 0.1338). We do **not** claim this; we cite it to justify *using* a block as the tool and to explain why scalar SAEs fragment the angle.
- **(Gauge-freeness, our framing.)** For an angular quantity, the identifiable object is the winding/phase, which is invariant under the block's `O(2)` gauge (§3.1). *Assumption:* the physical quantity's symmetry group matches the block's gauge group (true for circle↔O(2), torus↔product O(2)). *Consequence:* identity "up to isometry" is the right and achievable target; a stronger pointwise-axis identity is neither needed nor well-posed.
- **(Operational identifiability, validated not proven.)** *What would have to be true* for the recovered coordinate to be the physical one: the generative map from `y` to activations must be (i) injective on the manifold (different angles → different activations) and (ii) recoverable by the block up to isometry under the training noise. We do **not** prove this for ESM3; we *validate* it — on planted synthetic data (Phase 0: corr 0.9987) where the generative model is known, and empirically on held-out proteins (C1). The honest statement in the paper: "we claim operational identifiability — recovery up to isometry on held-out data under a train-fit alignment and a planted synthetic validation — not an asymptotic identifiability theorem."

This is the "modest theory you can actually state." It is defensible because it concedes exactly what SASA/Skopek own and claims only what Phase 0 and the held-out protocol support.

### 3.8 Persistence / commitment measure (definition)
Generative biological models build the output over internal steps; a mid-computation edit can be undone. Define, at a (layer ℓ, decode-step t) cell:
> **persistence ratio** `P(ℓ,t) = |Δ_final| / |Δ_single-pass|`,

where `Δ_single-pass` is the output change from one forward pass after the edit (an upper bound) and `Δ_final` is the change after the *full* iterative-unmasking schedule completes. `P ≈ 1` = the edit survives; `P ≈ 0` = washed out. Controls: matched-norm random direction (null band); a **coverage sweep** (fraction of positions edited) to locate where foldability breaks (import Hin26's axis); a three-leg **validity filter** (Mahalanobis typicality vs a random-AA reference per Zha26g — collapse threshold D−k√(2D), D≈d_model `[UNVERIFIED ~1536]`, k=1; pLDDT/pTM; clash + ProteinMPNN-refold designability) so a "change" is not off-manifold garbage. `P` is reported per layer as a map, and tied to the truth/confidence axes (does the model preserve truth-edits and confidence-edits differently?). This is the measure behind C5 and gate G2.

### 3.9 "Curved vs flat geometry" is a CONTROL, not a contribution (locked)
Phase 0 settled it: a curved (unit-circle) code *ties* a flat Euclidean block of equal dimension at matched bits (ΔFVU 0.0022 clean), and *loses* once the radius varies (ΔFVU +0.0213, flat strictly better). SASA owns the capacity theorem and the O(1)-block remedy; Skopek owns learnable signed curvature and documents its ill-conditioning. Therefore: we build **no learnable-curvature machinery**; the circular/torus block is one *fixed* parameterization used as a *tool* inside the identity analysis and judged by identifiability (circular-corr, topology), never by "beats flat"; every headline is reported at **matched bits/MDL**, never matched feature count; the curved-vs-flat tie appears as a pre-registered control result that *credits* SASA. This removes the single largest scoop exposure and is non-negotiable. The word "geometry" never appears in the contribution sentence.

---

## 4. CLAIMS C1–C6, RESTATED PRECISELY AND FALSIFIABLY

Each claim: statement → experiment → figure/table → baselines/ablations → prior work out-positioned → gate → NO-GO reshape. The spine is C1→C2→C3 (Track A); C3 is the hook; C4, C5 are controls elevated to claims; C6 is the generality answer to "cherry-picking". C3-B/C6 use the probe-only Track-B vignette. **The formal, pre-registered version of each claim — with the numeric threshold, exact null, statistical test, and refutation — is in §5; this section gives the narrative and the paper-shaping fallbacks, §5 gives the frozen hypotheses. The two are consistent by construction.**

### C1 — IDENTITY up to isometry
- **Statement.** The coordinate recovered from ESM3-open activations (sequence-only input) equals the physical coordinate up to the quantity's isometry group — O(2) for helix phase, product-O(2) for (φ,ψ) — measured on a held-out split with the alignment fit on train only. Not "a subspace reconstructs the activation".
- **Experiment.** Fit block/probe dictionaries at ~8 layers; align per §3.2/§3.3; report held-out circular/toroidal correlation + geodesic R².
- **Figure/Table.** **Fig 1**: recovered torus overlaid on the held-out Ramachandran plot, colored by geodesic error. **Table 1**: held-out alignment error for {geometry-matched block, Euclidean block, scalar SAE, random-init model} **at matched bits/MDL**.
- **Baselines/ablations.** (a) Random-init same-architecture ESM3 — torus must *not* appear; (b) matched-bits scalar SAE — must fragment into arcs; (c) alignment fit on train, scored on test (no post-hoc gauge cheating); (d) sequence-only input (structure input makes dihedrals trivial — void claim otherwise); (e) local-sequence-window positional baseline **run first** (recovery must not be index-driven); (f) the nuisance-only baseline (§8).
- **Out-positions.** SASA/Dal26 (reconstruction subspace, no identity); Setlur/Set26 (geometry-*annotated* ESM-2 SAE features, no gauge-free coordinate, no isometry guarantee); BSF/Jer26 (Euclidean blocks; within-block rotation a caveat — we make it the guarantee); InterPLM (single-direction features).
- **Gate G3/G1-A.** GO if held-out circular-R² beats scalar-SAE and the local-window baseline, beats random-init / nuisance by the pre-registered **≥0.15** margin (reconciled), topology correct. **NO-GO → MPR** benchmark paper (§12): the exact-geometry suite + the honest statement that identity does not hold.

### C2 — WRAP-AROUND causal dose-response
- **Statement.** Rotating the recovered circular coordinate by Δ moves the model's own output the matching way and returns to baseline at Δ = 2π — a closed causal loop a 1-D latent cannot produce. (§3.4)
- **Experiment.** Geodesic rotation sweep Δ ∈ [0, 2π], K ≥ 8 targets; readout = model's own output head; plot output angle vs Δ and the return-to-baseline residual.
- **Figure/Table.** **Fig 2**: output dihedral vs steering angle tracing a closed loop that returns to baseline at 2π; matched-norm random flat. **Table 2**: wrap-around fidelity (loop correlation, closure residual) for {recovered coordinate, scalar-direction steer (Steenwyk-style), matched-random}.
- **Baselines/ablations.** Matched-norm random 2-D direction; scalar structure-SAE direction (the Steenwyk must-beat — note it steers ESM3 at r=0.87 but *cannot close the loop*, which is the point); validity filter; readout independence (model's own head, not the training probe).
- **Out-positions.** Steenwyk/Ste26 (scalar steering, no 2π closure); Jed26b (decodable≠steerable — we show steerable with a stronger-than-linear signature); Engels/Eng24 (modular-arithmetic circular causal tests on LLMs with hand-set targets, no physical period — we use exact physical ground truth).
- **Gate G4.** GO if output follows the coordinate and closes at 2π above the random band on validity-passing structures. **NO-GO →** report as a decodable-but-not-cyclically-used negative; C1 stands alone.

### C3 — TRUTH ⟂ CONFIDENCE (the hook)
- **Statement.** The physical quantity is separable from the model's confidence, causally dissociable (double dissociation), and — the decisive form — the **trust residual read off the truth coordinate flags the model's confidently-wrong outputs better than the confidence head alone** (§3.6).
- **Experiment.** Estimate `U_truth` (C1) and `U_conf` (predicts held-out pLDDT/entropy). Measure (i) principal angle (vs random-null AND two-heads-on-one-latent null); (ii) the 2×2 double dissociation; (iii) the confidence-matched-pair separation; (iv) the confidently-wrong ROC of the trust residual vs confidence alone.
- **Figure/Table.** **Fig 3 (money figure, §10)**: the confidently-wrong scatter (high confidence, atypical physics) where the truth coordinate still tracks truth, plus the ROC of trust residual vs confidence. **Table 3**: principal angles, double-dissociation effect sizes, confidence-matched-pair accuracy, confidently-wrong AUROC.
- **Baselines/ablations.** Both directions matched-norm; `U_conf` must be *real* (predicts held-out confidence above chance); physics readout independent of the confidence head; **confidence-matched negative-pair control run first** (gate G5); confidence-ablation cross-check (ablate `U_conf`, show the physical coordinate and its causal effect survive — proves the truth feature is not re-encoded confidence); a model-independent descriptor (local geometry regularity for A; buried surface area for B) as a third anchor.
- **Out-positions.** The 2026 LLM truth-vs-confidence wave (`[UNVERIFIED]` IDs 2603.25052; "Confidence Manifold" 2602.08159; "Truthfulness Spectrum" 2602.20273; ACL-2026 "LLM Beliefs Are in Their Heads") — **same hypothesis, no external ground truth, no deployed confidence head, no causal wrap-around, no cross-track**; Dunbrack ipSAE/ipTM (define interface confidence, never ask if it is a separate axis from truth); PairSAE (conflated truth with the model's own confidence). **This claim carries the paper.**
- **Gate G5 (promoted to must-pass).** The hook is claimed **only if** the triple-lock holds: principal angle exceeds both nulls AND the double-dissociation interaction is significant (|d|≥0.5, CI excluding 0 on both off-diagonals) AND the confidence-matched-pair separation holds with confidently-wrong AUROC(trust residual) > AUROC(confidence) with CI separation. **NO-GO →** the entanglement rebuttal wins; reframe to the calibration/benchmark paper and report the honest "truth and confidence are entangled on this distribution" negative (still a contribution: it is the control the field needs).

### C4 — RoPE-PROOF phase slip
- **Statement.** The recovered phase feature tracks structure-derived irregularities (helix kinks, 3₁₀ segments, phase slips) that positional encoding alone cannot manufacture; it departs from index-mod-3.6 exactly at the structural slips. (Precondition: confirm ESM3-open uses RoPE `[UNVERIFIED]`; if it uses learned/absolute positions the test still localizes structural slips but the "RoPE-proof" framing is relabeled.)
- **Experiment.** Within DSSP helices, compare feature phase to (a) index-mod-3.6 and (b) true structural phase including kinks/3₁₀; show the feature follows the slip where the two diverge, via *partial* circular correlation after regressing out index-mod-period.
- **Figure/Table.** **Fig 4**: feature phase vs residue index with structural slip events marked; feature tracks the slip, index baseline does not. **Table 4**: residual at slip residues (feature vs index baseline).
- **Baselines/ablations.** Positional/local-window control run first; helix-vs-coil at matched index (RoPE's phase advance is identical for both at the same index; only biology differs); trained−random-init at the helix frequency; residue-shuffle/phase-randomized surrogates; ≥5 seeds.
- **Out-positions.** The two NeurIPS-2026 codon-frame papers (own index/period-3 — this is the baseline we beat, the signal must be *beyond* index) `[CONCEDE the codon leg]`; the general RoPE-artifact concern; Steenwyk geometry atlas. Note: do **not** advertise phase-velocity as "rebutting Spokes-not-a-Ring" (Gup26b is a different question — a category error a reviewer catches).
- **Gate G1.** GO if helix slope ≫ coil slope at matched index, trained−untrained gap significant at the helix frequency, partial circular-corr(feature, structure | index) > 0. **NO-GO →** the periodicity is positional; retract the periodicity claim, keep the non-periodic SASA control as the honest headline, report the RoPE-artifact finding as a methodological caution (valuable: tells the field its "rings" may be rotary index).

### C5 — PERSISTENCE / COMMITMENT (honest, possibly-negative)
- **Statement.** A mid-computation edit to the recovered coordinate either survives iterative decoding (positive causal result) or washes out (calibration negative), and where it survives is **layer-resolved**: the most-decodable, most-steerable, and most-persistent sites differ. (§3.8)
- **Experiment.** Inject at layer ℓ; let ESM3 iterative unmasking proceed; measure realized final change vs a late/output injection; build a layer (× decode-step) decodable-vs-steerable-vs-persistent map.
- **Figure/Table.** **Fig 5**: layer heatmap of decodability, steerability, persistence with three different argmaxes. **Table 5**: persistence z-score vs matched-random per layer.
- **Baselines/ablations.** Matched-random; output-only baseline; coverage sweep; three-leg validity filter.
- **Out-positions.** Zha26g (off-manifold collapse, ESM-2) + Hin26 (masked-pLM edit persistence) + Jed26b (one layer, Boltz) — we extend to a layer map on ESM3 structure tokens and tie persistence to the truth/confidence axes.
- **Gate G2 (most likely death, tested cheapest-first).** GO if persistence ratio clears the matched-random band (|z|>2), monotone in α, above the collapse threshold, on validity-passing structures. **NO-GO → convert, don't kill:** this is the publishable commitment/persistence negative; the causal half (C2) is then reported as "structurally impossible under commitment", and the paper leans on C1 + C3 + the negative.

### C6 — CROSS-VARIABLE / CROSS-MODEL INVARIANCE (re-based; the anti-cherry-pick answer)
- **Statement (re-scoped from the old cross-track claim).** The epistemic triad (identity-up-to-isometry, causal dose-response where persistence allows, truth⟂confidence) recurs across (i) two periodic *structural* variables in ESM3 — the (φ,ψ) torus and the helix-phase circle; (ii) a second model family at probe level — ESM-2 vs ESM3 (is the torus shared, more causal in the multimodal model?); and (iii) a scalar *thermodynamic* variable at probe level — experimental Kd vs ipSAE in Boltz (the Track-B vignette, C3-B). So the separation is a property of the model class, not one feature in one model.
- **Experiment.** Run C1/C3 on (φ,ψ) and helix phase; add the ESM-2/ESM3 panel; add the probe-only Kd-vs-ipSAE angle on leakage-aware splits (no steering).
- **Figure/Table.** **Fig 6**: side-by-side triad summary (torus | helix-phase | Kd-probe) + the ESM-2/ESM3 panel.  **Table 6**: each claim's effect size and control margins per variable/model.
- **Baselines/ablations.** The **mismatched-variable double dissociation**: a circular code wins on the periodic variables and *loses* on the non-periodic relative SASA — guards against "any rich code recovers anything". The Kd vignette carries the full leakage controls of G-B1/G-B2.
- **Out-positions.** Steenwyk cross-model (feature-sharing, no causal/epistemic axis); Whitfield (concept emergence by depth, no causal, no confidence); establishes generality nobody has claimed for this triad.
- **Gate.** GO if the triad holds on ≥2 variables and the Kd-vs-ipSAE angle is bounded from 0 on the leakage split. **NO-GO →** report per-variable; the invariance claim softens to "holds for the structural variables".

### The Track-B binding vignette (C3-B), scoped to probe-only for this submission
- **Statement.** The internal direction carrying *experimental* Kd is not collinear with the model's interface confidence (ipSAE/ipTM) or with Boltz's own predicted affinity — truth is stored apart from confidence for a scalar thermodynamic variable too; and there exists at least one benign complex where ipSAE is high but true Kd is weak and the recovered truth direction reports "weak" (the confidently-wrong money example, Track B).
- **Experiment (no steering).** Probe Boltz pair/interface reps → {experimental Kd, ipSAE, ipTM, Boltz-predicted affinity, BSA}; measure the three-way angle matrix on the frozen leakage-aware (PLINDER + time) split; find the confidently-wrong instance.
- **Gates G-B1, G-B2 (both probe-level, cheap, run in parallel with Track A).** G-B1: experimental-Kd probe beats the explicit leakage probe by ≥0.15 Spearman on the clustered+time split and beats random-init. G-B2: principal angle between the Kd direction and the ipSAE direction is large (cosine ≤ ~0.5) and Kd retains predictive power with ipSAE partialed out. **NO-GO on either → drop Track B to an appendix negative** ("Boltz's internal affinity feature is a memorization/confidence artifact, not a physical-Kd representation" — a publishable leakage/calibration audit) and make Track A fully load-bearing. The expensive Track-B causal program (SKEMPI, mid-diffusion steering, Feynman–Kac; gate G-B3) is **out of scope for ICML 2027** (deferred; §8, §15).

---

## 5. THE SIX CLAIMS AS PRE-REGISTERED FALSIFIABLE HYPOTHESES

*(Integrated verbatim-in-substance from the adversarial-reviewer pass §2, aligned with the architect's C1–C6 above so the two are consistent, not contradictory. Where a numeric threshold here is more specific than the narrative gate in §4, this section is the frozen version.)*

Format for each: **H / threshold / null / control / test (effect size + CI + multiple-comparison handling) / refutation**. Thresholds are pre-registered *before* any ESM3 run and frozen as a timestamped supplement. All CIs are cluster-bootstrap (sequence-cluster level) unless stated; all headline p-values Holm-corrected across the C1–C6 family; all per-layer/per-variable maps Benjamini–Hochberg FDR at q=0.05. "Nuisance baseline" = the single shared decoder on {position/RoPE phase} ∪ {train-membership + sequence/ligand cluster id} ∪ {model self-confidence (pLDDT for A, ipSAE/ipTM for B)} (§6, item 1; §8).

### C1 — IDENTITY UP TO ISOMETRY
- **Hypothesis.** On held-out proteins (sequence-only input, structure track off), the coordinate recovered from ESM3 activations equals the physical (φ,ψ)/helix-phase coordinate up to the variable's isometry group, with the isometry fit on train only.
- **Pre-registered threshold.** Held-out **circular R² (helix phase) ≥ 0.70** and **toroidal geodesic R² ((φ,ψ)) ≥ 0.55**, AND the recovered coordinate **beats the nuisance baseline by ≥ 0.15** in circular R², AND beats a **matched-bits scalar SAE** and a **random-init same-architecture ESM3** by **≥ 0.15** (reconciled margin), across ≥5 seeds with across-seed SD < 0.05.
- **Exact null.** (i) random-init ESM3 (architecture + RoPE, no training); (ii) nuisance baseline; (iii) matched-bits scalar SAE (expected to fragment the circle into arcs).
- **Exact control.** Train-only isometry fit scored on test (no post-hoc gauge cheating); sequence-only input; local-sequence-window positional control run *first*; time + cluster leakage splits.
- **Test / effect size / CI / MC.** Jammalamadaka–Sarma circular correlation and geodesic R²; effect = held-out R² gap to each null; cluster-bootstrap 95% CI on the gap; Holm-corrected within the C1–C6 family. Topology pre-committed: helix phase Betti (b0,b1)=(1,1); (φ,ψ) = **basin-structured T²** (NOT uniform torus — Ramachandran is non-uniform; underlying connectivity (1,2,1)), pre-registered so a mismatched realized topology is a refutation, not a reinterpretation.
- **Refutation.** Held-out R² below threshold, OR not beating random-init/nuisance by the margin, OR requiring a non-isometric warp to match truth, OR seed-SD ≥ 0.05. (Synthetic floor already passed: Phase 0 clean-circle corr 0.9987; torus factorization is the open sub-risk, gated at G0b.)

### C2 — WRAP-AROUND CAUSAL DOSE-RESPONSE
- **Hypothesis.** Rotating the recovered circular coordinate by θ moves the model's own output head in the matching direction and returns to baseline at θ=2π — a closed loop a scalar latent cannot produce.
- **Pre-registered threshold.** Loop correlation (realized output angle vs commanded θ) **≥ 0.6**; **return-to-baseline residual at 2π ≤ 20%** of peak excursion; monotone over 0→π; **> matched-norm random** at |z|>2; all on **validity-filtered** outputs; ≥5 seeds; mixed-effects slope significant with protein as random effect.
- **Exact null.** Matched-norm random direction at the same site (no signed monotone response, no closure); scalar direction (cannot close the loop by construction — the structural discriminator).
- **Exact control.** Model's own output head as readout (readout-independence, not a same-variable probe); three-leg validity filter on every point; decode-fidelity floor subtracted.
- **Test / effect size / CI / MC.** Mixed-effects circular regression (protein random effect); effect = loop correlation and 2π residual with cluster-bootstrap CI; permutation null at protein level; Holm within family. Gated behind G2 (persistence).
- **Refutation.** No closure, or indistinguishable from matched-random, or only achieved by off-manifold (collapse-filter-failing) structures → report as the decodable-but-not-causal-as-cyclic negative; C1 stands alone.

### C3 — TRUTH ⟂ CONFIDENCE (the hook, as a conjunction)
- **Hypothesis.** The physical quantity is **causally dissociable** from the model's deployed calibrated confidence: steering truth moves physics but not pLDDT/ipSAE, steering confidence moves pLDDT/ipSAE but not physics; and on **confidence-matched inputs** the truth coordinate still separates them; and the **trust residual** flags confidently-wrong cases better than the confidence head alone.
- **Pre-registered threshold (triple-lock, all three must hold).** (a) Principal angle between truth and confidence subspaces **>** the 95th percentile of the matched-norm random-angle null AND **>** the two-heads-on-one-latent synthetic null (MPU-0); (b) double-dissociation interaction effect size (standardized) with 95% CI excluding 0 and |d| ≥ 0.5 on both off-diagonal cells; (c) on confidence-matched pairs, truth-axis separation AUROC **≥ 0.70** (DeLong 95% CI lower bound > 0.5) where the confidence term is at chance by construction, AND confidently-wrong AUROC(trust residual) > AUROC(confidence) with separated CIs.
- **Exact null.** Matched-norm random direction; two linear heads fit on a *single* planted latent (synthetic calibration of "how separate is separate"); confidence term on confidence-matched pairs (chance by construction).
- **Exact control.** Confidence direction must be *real* (predicts held-out pLDDT/ipSAE above chance); physics readout independent of the confidence head; **confidence-matched negative-pair control run FIRST** inside this phase; a model-independent descriptor (local geometry regularity for A; buried surface area for B) as a third anchor.
- **Test / effect size / CI / MC.** Principal angles (subspace); 2×2 interaction via mixed-effects with cluster random effect; DeLong AUROC with CI for the confidence-matched separation and the trust-residual ROC; Holm within family.
- **Refutation.** Any of the triple-lock fails; in particular, collinear subspaces, or separation vanishing on confidence-matched pairs → the "one entangled signal" rebuttal wins → report the entanglement as the finding (the model does *not* separate truth from confidence here — a calibration negative). **This is the claim most at risk; see §14.1.**

### C4 — RoPE-PROOF PHASE SLIP
- **Hypothesis.** The phase feature tracks **structure-derived** slips (helix kinks, 3₁₀ segments) that positional encoding cannot manufacture; it departs from index-mod-3.6 exactly at the structural slips.
- **Pre-registered threshold.** **Partial circular correlation(feature-phase, structure-phase | index-mod-3.6) > 0** with cluster-bootstrap CI excluding 0; helix slope ≫ coil slope at matched index (CI excluding coil); trained−random-init gap significant at the helix frequency.
- **Exact null.** Index-mod-3.6 local-window baseline; residue-shuffle and phase-randomized surrogates; random-init (same RoPE).
- **Exact control.** Positional/local-window control run FIRST (gating pre-condition — the periodic claim is retracted if it fails); sequence-only; ≥5 seeds.
- **Test / effect size / CI / MC.** Partial circular correlation; residual at slip residues (feature vs index baseline); permutation null; BH-FDR across helices. Holm for the headline C4 statistic.
- **Refutation.** Signal equal in coil or in the untrained model, or vanishes once index-mod-3.6 is partialed out → periodicity is positional → retract the periodic claim, keep non-periodic SASA as the honest headline variable, report the RoPE-artifact as a methodological caution (valuable).

### C5 — PERSISTENCE / COMMITMENT (honest, likely-negative)
- **Hypothesis.** Mid-computation edits to the recovered coordinate either survive ESM3 iterative unmasking (positive) or are washed out (negative), and where they survive is **layer-resolved** (argmax-decodable ≠ argmax-steerable ≠ argmax-persistent).
- **Pre-registered threshold.** Persistence ratio (final-decode Δ / single-pass Δ) clears matched-random |z|>2, monotone in α, above the Mahalanobis collapse threshold (D−k√(2D); Zha26g, D≈d_model **UNVERIFIED ~1536**, k=1), on validity-passing structures; coverage sweep locates the foldability break.
- **Exact null.** Matched-random direction; output-only (late) injection baseline.
- **Exact control.** Three-leg validity filter (Mahalanobis typicality ⊕ pLDDT/pTM ⊕ ProteinMPNN refold); coverage sweep; ≥5 seeds.
- **Test / effect size / CI / MC.** Per-layer persistence z vs matched-random, cluster-bootstrap CI; BH-FDR across layers; the three argmaxes reported with CIs.
- **Refutation / reframe.** Null everywhere or collapse-only → the commitment/persistence NEGATIVE is a headline calibration result (extends Hin26/Zha26g/Jed26b to ESM3 structure tokens + layer/coverage map). This claim is designed to be publishable in *either* sign.

### C6 — CROSS-TRACK INVARIANCE
- **Hypothesis.** The triad (C1 identity, C2/C3 cause+separation) holds for a periodic structural variable (ESM3, φ/ψ) AND a scalar thermodynamic one (Boltz, Kd), AND each geometry **loses on a mismatched variable** (circular code fails on non-periodic relative SASA — a double dissociation that guards against "any rich code recovers anything").
- **Pre-registered threshold.** Both tracks independently pass their C1+C3 thresholds; the mismatch control shows the circular code's SASA R² is **not** better than a scalar code's (CI overlapping or worse), establishing specificity.
- **Exact null / control.** Mismatched-variable control (periodic code on SASA); per-track nuisance baselines; Track B gated behind G-B1 (leakage) and G-B2 (Kd⟂ipSAE).
- **Test / effect size / MC.** Per-track effect sizes with CIs side by side; the invariance claim is made only if both clear independently (no pooling that could let one track rescue the other).
- **Refutation.** Either track fails its own thresholds, OR the mismatch control shows the circular code also "recovers" SASA (code is not specific) → drop C6, report per-track results separately; the paper stands on Track A.

---

## 6. THE CONTROLS BATTERY (mandatory vs nice-to-have, with the reviewer who demands each)

*(Integrated from the adversarial-reviewer pass §3. A control is **MANDATORY** if the paper is desk-reject-adjacent or a named reviewer sinks it without the control; **NICE-TO-HAVE** if it strengthens but is not load-bearing. The ordering discipline is: the cheap probe-level controls are run and must clear **before** any expensive steering/decode job — the single biggest compute-saving discipline in the plan.)*

1. **Nuisance-only baseline** — a decoder given ONLY {position/RoPE phase} ∪ {train-membership + sequence/ligand cluster id} ∪ {model self-confidence}. **MANDATORY.** *Why:* pre-empts three of the six deaths at once — RoPE (R1/R3), leakage (R2), confidence entanglement (R4) — and every GO threshold in §5 is "beat this by a pre-registered margin," not "beat zero." *Demanded by:* all four reviewers; the spine control. Compute: cheap, before everything.

2. **Confidence-matched negative pairs** (hold pLDDT/ipSAE fixed, vary true physics) — **MANDATORY, run FIRST inside the C3 phase.** *Why:* the only clean refutation of the "truth and confidence are one entangled signal" rebuttal — holds confidence fixed so any separation is truth-specific; a principal angle alone cannot do this. *Demanded by:* R4 (theory/stats) and the AC; the control the headline survives or dies on. *Risk flagged (§14):* Track A may not supply enough high-confidence/atypical-geometry pairs — power analysis required.

3. **Random-init same-architecture model** (full pipeline) — **MANDATORY.** *Why:* the Heap–Aitchison / Korznikov table-stakes control; separates "learned" from "architectural/RoPE." A 2027 submission without it is desk-rejectable at this boundary. *Demanded by:* R1, R3.

4. **≥5 seeds + stability statistic (+ Tournament-TopK)** — **MANDATORY.** *Why:* BSF is seed-fragile (Jerpelea); Shahsavand shows reproducibility ≠ meaning. Report across-seed SD of every headline statistic and the mean−2·SD margin to baseline. *Demanded by:* R1, R4.

5. **Positional / RoPE / local-window control, run FIRST** (sequence-only input for Track A) — **MANDATORY and gating.** *Why:* ESM3 uses RoPE, a false-positive machine for constant phase velocity; helix/codon phase may be trivially positional. The structure-derived phase-slip partial correlation (C4) is the RoPE-proof test. *Demanded by:* R2, R3; the periodic claim is retracted if it fails.

6. **Matched-bits / MDL comparisons (identical code for block and SAE), never matched feature count** — **MANDATORY.** *Why:* a block has strictly more capacity per active unit; "fewer units" is partly definitional. Any win appearing only at matched-units is an artifact. *Demanded by:* R1.

7. **Matched-norm random-direction control for every steering claim** — **MANDATORY.** *Why:* the Jed26b/Xun26 null band; without it a "steering effect" is uninterpretable. *Demanded by:* R3.

8. **Leakage-aware time + scaffold/sequence-cluster splits + explicit leakage probe** — **MANDATORY** (Track A dihedrals *and* Track B Kd). *Why:* ESM3/Boltz trained on PDB; co-folding Kd notoriously inflates on random splits (PairSAE circularity). The explicit leakage probe (nuisance identifiers only) upper-bounds what memorization alone predicts; the feature must beat it on the clustered/time split. *Demanded by:* R2 (and it is the whole of G-B1 for Track B).

9. **Validity filter (Mahalanobis typicality ⊕ pLDDT/pTM/clash ⊕ ProteinMPNN refold) on every steered output** — **MANDATORY.** *Why:* Zha26g — strong steering drives activations off-manifold to random-AA statistics while the oracle still scores "success"; an unfiltered "effect" is a collapse artifact. Report Δ only for candidates that pass; report the α at which collapse begins. *Demanded by:* R2, R3.

10. **Readout-independence** (model's own output head, not a same-variable probe) — **MANDATORY** for every causal claim. *Why:* traversing a dihedral coordinate and reading a dihedral probe is circular. *Demanded by:* R3.

11. **Decode-fidelity calibration** (structure-token round-trip (φ,ψ) noise floor) — **MANDATORY** for Track A causal readout. *Why:* a Δ(φ,ψ) below the decode noise floor is meaningless; all causal Δ(φ,ψ) reported in units of, and gated above, that floor. If the floor is too high, the causal readout falls back to ProteinMPNN-refold. *Demanded by:* R2.

12. **Hierarchical / cluster-respecting resampling for ALL inference** — **MANDATORY.** *Why:* residues within a protein and proteins within a cluster are not independent; residue-level i.i.d. CIs are invalid and far too narrow. Cluster bootstrap at the sequence-cluster level for every CI; mixed-effects models with protein (and cluster) as random effects for all dose-response slopes; permutation nulls that permute at the protein level. *Demanded by:* R4; invalidates the paper's stats if missing.

13. **Two-tier multiple-comparison correction** (Holm–Bonferroni for the C1–C6 family; BH-FDR q=0.05 for sweeps) — **MANDATORY.** *Why:* layer × variable × method sweeps are a garden of forking paths. The layer chosen for the causal campaign is fixed by the *decodability* gate (G1) before any steering, so the steering result is not post-selected. *Demanded by:* R4.

14. **Mismatched-variable double dissociation** (circular code loses on non-periodic SASA) — **NICE-TO-HAVE → MANDATORY for the C6 invariance claim.** *Why:* guards against "any rich code recovers anything." Only load-bearing if C6 is claimed.

15. **Discovery hedge** (an unlabeled identified feature predicting a held-out physical property) — **NICE-TO-HAVE.** *Why:* pre-empts "you only re-found physics" (R2) without being load-bearing; pre-registered as exploratory.

16. **Regularizer-insensitivity sweep** (if any learnable-κ block is used) — **MOOT / CUT:** we use a **fixed** 2-D block (Skopek ill-conditioning + Phase 0 radial-noise penalty), so no κ to sweep. Kept here only to note why learnable curvature is out.

---

## 7. SIMULATED REVIEWS AND RESPONSES (four reviewers + the AC meta-review)

*(Integrated from the adversarial-reviewer pass §1 and the AC meta of §5. Four adversarial reviewers plus the AC. For each: who they are, their top objections, and the **concrete experiment / control / statistic** that answers each. Initial scores are "as the plan stood"; post-hardening ceilings are after the §5/§6 hardening lands. If an objection has no answer yet, it is said so and feeds §14.)*

### Reviewer 1 — The geometry / SAE / dictionary-learning expert
*(Goodfire/Columbia/Stanford-adjacent; owns BSF, Tournament-TopK, SASA, InterPLM.)* **Initial score: 3 (weak reject). Post-hardening ceiling: 6–7.**

- **O1.1 "Curved/block geometry is a solved non-result — SASA (Dal26) owns the covering-number theorem and the O(1)-block remedy, and a flat Euclidean block of equal dimension already ties. Your Phase 0 confirms the tie. What is the methodological contribution?"** *Answer:* Concede completely and in print. The dictionary is a *tool*, not a claim; the contribution is **identity up to isometry** (SASA recovers a reconstructing subspace, never claims identity) plus **causality** plus **truth⟂confidence**. *Statistic that settles it:* Table 1 reports held-out circular/geodesic error for {fixed 2-D block, Euclidean block, scalar SAE, random-init} **at matched bits/MDL**, and the headline number is the isometry-aligned held-out error, not any FVU win. We pre-register the curved-vs-flat comparison to TIE and present it as a control (Phase 0: ΔFVU = 0.0022 clean; +0.0213 — flat better — under radial noise). We never pitch "curved beats flat."
- **O1.2 "BSF is seed-fragile (Jerpelea/Ananthram): manifold splitting, within-block gauge ambiguity. Your 'identity' is a seed artifact."** *Answer / control:* (a) winding/phase is **rotation-invariant**, so it sidesteps the within-block O(group_size) gauge ambiguity that bites BSF — stated as the reason the identity claim is well-posed where BSF's generic feature is not; (b) **≥5 seeds + Tournament-TopK**, reporting seed-variance of the isometry-aligned held-out circular correlation as a first-class stability statistic; (c) pre-registered stability floor: across-seed SD of circular-R² < 0.05 and mean − 2·SD still clears the nuisance baseline.
- **O1.3 "Decodability ≠ meaning (Heap–Aitchison; Shahsavand; Korznikov). Your probe reads a torus because dihedrals are linearly present, not because the model *uses* them."** *Answer:* Precisely why identity (C1) is **not** the headline and why C2/C3 exist. The decodable≠steerable map (G3) and the causal double dissociation (C3) are the model-uses-it evidence; the trained−random-init gap is the "it is learned, not architectural" evidence.
- **O1.4 "Matched-units is not matched-bits; your block has strictly more capacity per active unit."** *Answer:* All headline comparisons at **matched bits/MDL with an identical code for block and SAE**; any result that appears only at matched-units is reported as an artifact. In the pre-reg.

### Reviewer 2 — The structural-biology / protein-ML expert
*(EvolutionaryScale/AlQuraishi/Baker-adjacent; cares whether the biology and the ground truth are real.)* **Initial score: 4. Post-hardening ceiling: 7.**

- **O2.1 "You are re-finding physics. Dihedrals and SASA are computable from coordinates — of course they're decodable. Where is the biological content or discovery?"** *Answer:* Reframe as *measurement rigor and model epistemics*, not discovery (gap #12). The content: the model holds the quantity as a *causally used, identifiable* coordinate that is *separate from its own confidence* — a statement about the model's internal epistemic structure, not about protein physics. One **discovery hedge** ready: an identified feature with no DSSP/annotation label that predicts a held-out physical property (pre-registered as exploratory, not load-bearing).
- **O2.2 "Sequence-only input is mandatory and you know it — if ESM3 sees structure tokens, dihedral decoding is trivial and the whole paper is circular."** *Answer / gating pre-condition:* sequence-only input is a **gate**, not a footnote; structure track OFF for all Track-A caching. Fig 1's caption and the methods lead with it.
- **O2.3 "Your ground truth leaks. ESM3 was trained on PDB; the 'held-out' dihedrals are from structures whose sequences the model memorized."** *Answer:* Two-level defense: (a) **time-split** on deposition date (train pre-cutoff, test post-cutoff) *and* (b) **sequence-cluster split** (MMseqs2 ≤30% identity, no cluster spans train/test); (c) the **nuisance-only baseline** includes a train-set-membership flag + cluster ID, so "predictable from memorization alone" is explicitly upper-bounded and the feature must beat it by the pre-registered margin. Leakage audit protocol: §13.
- **O2.4 "ESM3 structure-token → backbone (φ,ψ) decode may not be faithful enough at your length regime to measure a dihedral change." (UNVERIFIED that it is.)** *Answer:* Pre-register a **decode-fidelity calibration**: round-trip (coords → structure tokens → coords) on held-out experimental structures, report per-residue (φ,ψ) reconstruction error; all causal Δ(φ,ψ) reported in units of, and gated above, that decode noise floor. If the floor is too high, the causal readout falls back to ProteinMPNN-refold (§3).
- **O2.5 "pLDDT is not 'confidence about the dihedral' — it's a per-residue positional confidence. You are contrasting the truth of φ,ψ against the wrong confidence object."** *Answer (a real threat; see §14):* define the confidence target per-variable and pre-register it — for Track A the confidence direction is the activation direction most predictive of held-out **pLDDT at that residue**, and we test and report that pLDDT is itself informative about local geometry regularity (so the contrast is meaningful) while remaining a *distinct* readout. If the confidence head is not meaningfully about the same object, the clean contrast moves to Track B (ipSAE is explicitly an interface-quality score for the Kd object).

### Reviewer 3 — The causal-interpretability skeptic
*(Jedryszek/Crook/Oxford; Xun26 "readable before causal"; the "decodable ≠ steerable" crowd. The hardest reviewer.)* **Initial score: 3. Post-hardening ceiling: 6. This reviewer decides the paper.**

- **O3.1 "Your edits wash out. Hin26 shows masked-pLM activation edits die in the activation→token-sample→re-embed cycle and foldability collapses past ~15% coverage; Zha26g shows strong steering drives ESM-2 off-manifold toward random-AA statistics while the oracle still scores 'success.' ESM3's iterative unmasking + discrete token commitment is the same mechanism. Your causal claims (C2, C5) are dead on arrival."** *Answer:* This is **G2, the single most likely death, gated first among the causal tests, before any sweep.** Design: persistence ratio = final-decode Δ(φ,ψ) / single-pass Δ(φ,ψ), with a coverage sweep (import Hin26's axis), a three-leg validity filter (Mahalanobis typicality per Zha26g, pLDDT/pTM, ProteinMPNN-refold designability), ≥5 seeds, GO only if persistence clears the matched-random band (|z|>2), is monotone in α, **above** the collapse threshold. *Pre-registered refutation and pivot:* if null everywhere or only "works" via collapse, C2/C5 become the **honest commitment/persistence negative** (extends Hin26/Zha26g/Jed26b to a layer map on ESM3 structure tokens) and the causal half becomes a calibration result — still publishable, and this reviewer *respects* an honest negative far more than an overclaimed positive.
- **O3.2 "Readable ≠ causal (Xun26): your identity result (C1) is the readable half and proves nothing about use."** *Answer:* Exactly why C1 is explicitly *not* sufficient and the contribution is the conjunction. The **causal-efficacy ratio** (realized Δ / decodability) made quantitative on ESM3, layer-resolved (G3a: argmax-decodable ≠ argmax-steerable), is Xun26 ported to bio with a statistic. *Statistic:* per-layer efficacy ratio with cluster-bootstrap CI; the claim is that the decodable-argmax layer has a significantly lower efficacy ratio than the steerable-argmax layer (paired, across proteins, Holm-corrected).
- **O3.3 "Scalar already steers ESM3 — Steenwyk (Ste26) gets dose r=0.87 at layer 33. Your 'geometry-matched beats scalar' is a re-skin."** *Answer:* Concede scalar steers (the old "scalar can't steer" sub-claim is dead). The claim is the **quantitative geometry lift at matched norm** on *validity-filtered, wrap-aware* realized Δ(φ,ψ) (G3b), plus the **wrap-around closure** (C2) — a 2π return a scalar direction structurally cannot produce. *Pre-registered refutation:* if curved ties scalar at matched norm, the geometry-lift claim is dropped and G3a (the layer-resolved efficacy map) + wrap-around carry the causal section; "geometry vs scalar" demotes to a control.
- **O3.4 "Readout circularity: traversing a dihedral coordinate and reading a dihedral *probe* is circular."** *Answer:* **Readout-independence control (mandatory):** the causal readout is the model's **own output/structure-token head decoded to backbone**, never a same-variable probe. Stated in the C2/C4/C5 methods.
- **O3.5 "Direction-correct but magnitude-wrong is not a pass."** *Answer:* Pre-registered: magnitude agreement is required for a full pass; direction-correct/magnitude-wrong is logged as a **partial** pass, not folded into the headline.

### Reviewer 4 — The theory / identifiability / statistics reviewer
*(Cares what "identifiable up to isometry" means, whether the stats are valid under non-independence, and whether "truth ⟂ confidence" is well-posed.)* **Initial score: 4. Post-hardening ceiling: 7.**

- **O4.1 "'Identifiable up to isometry' is asserted, not proved. You fit an alignment and report a correlation — that is recovery under *a* transformation you chose, not an identifiability theorem. What is the identifiability class and what breaks it?"** *Answer:* Scope honestly: we claim **empirical recovery up to the variable's isometry group on a held-out split with the alignment fit on train only** — not a population guarantee. State the isometry group per variable (SO(2)/O(2) for helix-phase circle; the torus isometry group for (φ,ψ); monotone reparam for Kd). The *falsifier* is pre-registered: if the recovered coordinate requires a **non-isometric** (non-monotone, many-to-one) warp to match truth, the identity claim fails. Phase 0 is the synthetic proof-of-metric (clean circle corr 0.9987); the honest caveat is also pre-registered: **naive PCA does not factor the torus** (Phase 0 per-angle corr 0.61/0.59), so a genuine product-manifold factorization (not PCA) is required and its success is itself a gated deliverable (G0b on real data).
- **O4.2 "Circular correlation CIs and p-values assume i.i.d. residues. Residues within a protein, and proteins within a cluster, are not independent — your CIs are far too narrow and your 'significant' principal angle may be noise."** *Answer (the statistics the paper lives or dies on):* all inference uses **hierarchical / cluster-respecting resampling** — cluster bootstrap at the sequence-cluster level for every CI; mixed-effects models with protein (and cluster) as random effects for all dose-response slopes; permutation nulls that permute at the protein level. No residue-level i.i.d. test appears in the paper.
- **O4.3 "'Truth ⟂ confidence' via principal angle > 0 is a near-empty claim — two independently fit linear heads on correlated data are almost never exactly collinear. What is the *null* and the *effect size*, and is it bigger than you'd get from two heads on one latent?"** *Answer:* The principal-angle claim is **not** the evidence; it is a screen. The evidence is (i) the **causal double dissociation** (2×2 effect-size grid with cluster-bootstrap CIs and an interaction test) and (ii) the **confidence-matched negative-pair control** (hold pLDDT/ipSAE fixed, vary true physics; the truth coordinate separates, confidence by construction cannot). The null for the principal angle is the **matched-norm random-direction angle distribution** AND the **two-heads-on-one-synthetic-latent angle** (MPU-0 plants an orthogonal confidence channel to calibrate what "separate" looks like under the null). The claim is a **triple-lock**, not a bare angle.
- **O4.4 "Multiple comparisons: you sweep 6–8 layers × several variables × several methods and report the best. Garden of forking paths."** *Answer:* Two-tier correction, pre-registered. **Headline family** (C1–C6, one pre-registered statistic each) controlled at family-wise error with **Holm–Bonferroni**. **Exploratory sweeps** (per-layer maps) controlled with **BH-FDR at q=0.05** and reported as maps, never a cherry-picked cell. The layer for the causal campaign is fixed by the *decodability* gate (G1) before any steering, so the steering result is not post-selected.

### Reviewer 5 — The Area Chair meta-review (ethics, scope, novelty, venue-fit)
- **Scope:** too broad as originally scoped (two methods, four models). Enforce the cuts (§12): one method-as-tool, ESM3 load-bearing, Boltz gated extension, Jacobian + genomic panel out.
- **Novelty:** the bare truth/confidence claim is scoop-exposed; the conjunction is not. The paper must foreground the four LLM-impossible differentiators in the abstract.
- **Ethics:** Track B is capability-adjacent; the dual-use statement is **owed a human/compliance pass** (an automated safeguard has repeatedly truncated drafts). A submission blocker, not a nicety (§15).
- **Venue fit / timeline:** ICML 2027 main by late Jan is ~8.5–9 weeks and not realistic at this rigor; the AC's recommendation is the pivot ladder in §11. Honest plan of record: ICLR-2027-workshop stake → NeurIPS 2027 main (or D&B floor), with ICML 2027 attempted only if the gates clear on the aggressive schedule.

---

## 8. FULL EXPERIMENTAL PROTOCOL — PER PHASE, PER TRACK

Phases ordered cheapest-decisive-first; every probe-level gate clears before any costly steering/decode job launches (steering is 10–100× the cost and wasted if a probe gate fails). The single **nuisance-only baseline** is threaded through every gate as the common null: a decoder given only {position index / RoPE phase} ∪ {train-set-membership + sequence/ligand cluster id} ∪ {model self-confidence (pLDDT for A, ipSAE/ipTM for B)}. Every GO threshold is "beat the nuisance baseline by a pre-registered margin", not "beat zero". This one artifact pre-empts the RoPE, leakage, and confidence confounds at once.

### Phase 0 — Synthetic calibration (DONE; CPU; `experiments/phase0_synthetic/`)
Status: run and passed (gate G0 GO). Results: curved vs flat tie (FVU 0.1361 vs 0.1338, Δ 0.0022); angle recovery up to isometry corr 0.9987; scalar covering penalty present (16 scalar atoms to match 1 two-dim block at ε≈0.164); torus caveat (per-angle corr 0.6146/0.5934 under naive PCA → §3.3 factorization needed). **Extension for this submission (MPU-0, still CPU, no weights):** plant a torus **and an orthogonal "confidence" channel** in R⁶⁴ **with a tunable correlation ρ between them**; verify the block recovers the coordinate up to isometry (C1 metric) *and* that the trust-residual metric (§3.6) separates planted truth from planted confidence, that confidence-matched pairs behave as designed, and **measure the ρ at which the method can no longer distinguish "two readouts of one latent" from "two latents"** — calibrating the hook's resolution before any model run (de-risk MPU-0). Also implement/unit-test the ISA/total-correlation factorization (§3.3; G0b) and the wrap-around identity retraction here, so Track A inherits tested code.

### Phase 1 — Persistence / commitment gate (the most-likely death; test before any steering)
**Track A (ESM3-open, iterative unmasking).** Cache residual stream at one mid-depth layer (start L≈24 of 48; also probe the Steenwyk L33 site) for ~1–2k sequence-only chains. Fit a cheap scalar direction for a crude helix-phase coordinate just to *have* a direction to push (identity comes in Phase 2). During iterative unmasked generation, add α·d at the chosen layer/step; let decoding finish; recompute the realized structural quantity (decode structure tokens → coords → DSSP). Subtract the matched-norm random band; apply the three-leg validity filter; coverage sweep. Metric: persistence ratio (§3.8), monotone in α, z vs random. **Gate G2.** Compute: 24 GB GPU feasible; ~30–80 GPU-h `[UNVERIFIED]`.
**Track B (Boltz, mid-diffusion) — persistence pilot only, N≈100 benign complexes.** Cache the single representation (+ interface-residue pair subset) at a coarse layer×denoising-step grid; fit a cheap BSA direction; inject inside one reverse step; recompute final-structure BSA; matched-random + validity. **This is the only Track-B *generation* step in scope, and only to decide whether the appendix vignette can later be extended; it is not load-bearing.** Compute: 1×80 GB; ~80–200 GPU-h `[UNVERIFIED]`. (If deadline pressure bites, Track-B persistence is cut entirely and the vignette stays probe-only.)

### Phase 2 — Identity + truth⟂confidence (the core)
**Track A caching plan.** Input **sequence-only** (structure track OFF — else dihedrals are trivially present and the claim is void). Corpus ~5k experimental-structure PDB chains, MMseqs2-clustered to ≤30% identity, time-split, ~200 res each ≈ 1M residue tokens. Sites: residual stream post-block at 8 layers spanning depth (e.g. 6,12,18,24,30,36,42,48) + L33; fp16; ~3.07 GB/layer → ~24.6 GB for 8 layers (stream shards to GPU). Ground truth from DSSP/mdtraj on *experimental* coords: φ, ψ, ω, helix phase (index mod 3.6 within DSSP H), relative SASA (non-periodic control). **Confidence source — must-verify (§14):** on sequence-only input with the structure track off, confirm ESM3 emits a usable per-residue pLDDT; if not, use structure-token predictive entropy as `c`. Record `c` per residue.
**Trained per layer:** scalar TopK SAE (b=1) · Euclidean BSF (b=2,3; Vanilla/Grassmann/Group-Lasso, Tournament-TopK) · circular S¹ (b=2) · torus T² (b=4) · ridge probe · 1-hidden-MLP probe · confidence probe (predicts `c`). Sweep b, l0, λ; 5 seeds.
**Identity metrics:** held-out circular/toroidal correlation + geodesic R² after train-fit alignment (§3.2/§3.3); topology pre-registration (basin-structured T² / circle (1,1)) with trained−random gap > 0.1; RoPE-proof phase-slip panel with the local-window control run first.
**Truth⟂confidence metrics:** principal angle(s) `U_truth`↔`U_conf` (vs random-null and two-heads-on-one-latent null); confidence-matched-pair separation (run first, gate G5); confidently-wrong AUROC of the trust residual vs confidence; depth at which they diverge. **Gate G1-A / G5.** Compute: 24 GB fits the fits; caching streamed; ~100–250 GPU-h `[UNVERIFIED]`.
**Track B (probe-only, this submission).** Frozen PLINDER-leakage-aware + time split; N≈300–500 benign complexes with measured Kd (PDBBind/BindingDB); cache trunk/pairformer at the final (or a few) denoising steps + interface-pair subset. Probe → {experimental Kd, BSA, ipSAE, ipTM, Boltz-predicted-affinity}. Metrics: held-out Kd Spearman on the leakage split vs the explicit leakage probe (G-B1); three-way angle matrix {Kd, ipSAE, Boltz-pred-affinity} (G-B2); the confidently-wrong instance (high ipSAE, weak Kd, truth-direction reports weak). **No steering.** Compute: 1–2×80 GB for the extra sampling; ~150–400 GPU-h `[UNVERIFIED]` — the data engineering (parsing, de-dup, benign allow/deny list, split hashing), not the GPU, is the effort sink.

### Phase 3 — Cause (wrap-around) + RoPE-proof validator (Track A only, this submission)
Only reachable if Phase-1A persistence passed. Pick the layer with best identity+persistence. Geodesic traversal of the recovered circular coordinate 0→2π (§3.4; projected-renormalize retraction); inject into the residual stream during generation; readout = ESM3's own output head (not the training probe). Controls: matched-norm random; output-only; local-window; three-leg validity filter on every point. Metrics: monotone dose-response over 0→π; wrap-around closure at 2π vs the flat/scalar code that cannot close; z vs random; % passing validity; curved vs scalar vs random at matched norm (geometry-lift, reported as a *control* per §3.9). Also build the layer-resolved decodable≠steerable map (G3a) — the most-decodable layer is not the most-steerable layer. **Gates G3, G4.** Compute: 1–2×80 GB; ~100–250 GPU-h `[UNVERIFIED]`. **Track-B causal (SKEMPI, mid-diffusion steering, Feynman–Kac; G-B3) is explicitly deferred — not in the ICML 2027 submission.**

### Phase 4 — Cross-variable/model synthesis, full sweep, write-up
Full dictionary × probe × ≥5-seed × Tournament-TopK sweep on Track A; MDL/rate-distortion frontier for every headline (matched **bits**); random-init control for every figure; the ESM-2 vs ESM3 generality panel; the mismatched-variable double dissociation (circular code loses on SASA); unify truth⟂confidence across the structural variables and the probe-level Kd vignette; negative-result bookkeeping (every failed gate becomes a reported calibration finding). **Gate G-final.** Compute: small-cluster tier for the full ensemble `[UNVERIFIED]` ≈ few k GPU-h; the *core* result needs far less (the above phases).

### Statistics (applies to all phases)
≥5 seeds, report median + a stability statistic (seed variance of the recovered coordinate's alignment; Tournament-TopK for block selection). Every headline reported as **trained − random-init** gap, not absolute (Heap–Aitchison trap). Held-out splits frozen and hashed before any fitting (Track A: MMseqs2 30%-identity + time; Track B: PLINDER leakage-aware + deposition-date time split). Pre-register — timestamped before fitting — the curved-ties-flat tie, the topology/Betti signatures, the gap thresholds, and the confidently-wrong AUROC direction. Circular statistics throughout (Jammalamadaka–Sarma correlation; von Mises where a parametric test is needed). **All inference cluster-respecting** (cluster bootstrap at the sequence-cluster level; mixed-effects with protein/cluster random effects; protein-level permutation nulls — no residue-level i.i.d. test). Effect sizes with CIs for every dissociation. **Two-tier multiple-comparison control**: Holm–Bonferroni across the C1–C6 family; BH-FDR q=0.05 across the 8-layer/variable/method sweeps (report the layer-resolved map, correct for layer selection when quoting a single "best layer" number).

---

## 9. RELATED WORK AND POSITIONING (drafted to ICML standard)

**Truth, belief, and confidence in foundation models.** A line of work probes an internal "truth" or "belief" direction in LLMs (Azaria & Mitchell; Burns et al. CCS; Marks & Tegmark), and a 2026 wave argues models keep an internal correctness axis separate from expressed confidence (`[UNVERIFIED IDs]` "Confidence Manifold" 2602.08159; "Truthfulness Spectrum" 2602.20273; 2603.25052; ACL-2026 "LLM Beliefs Are in Their Heads"; EMNLP-2025 verbal-uncertainty work). These share our hypothesis but, by construction, define "truth" as the model's own answer-correctness and "confidence" as an elicited/probed quantity — no external referent, no deployed calibrated head, and no causal or identifiability apparatus. We import the question into the one domain where truth (physics) and confidence (pLDDT/ipSAE) are pinned independently of the model, and add the causal double dissociation and identity-up-to-isometry the LLM program lacks. The trust-axis construction exports back to the LLM setting. (The "Truthfulness Spectrum" result that "truth is not one direction" is itself a warning that the bare separation is contested even in LLMs — another reason to lead with the conjunction, not the separation.)

**Interpretability of protein/structure models.** Sparse autoencoders on protein LMs (InterPLM, Simon & Zou, Nature Methods 2025; Adams et al.; Gujral et al., PNAS 2025; Reticular Matryoshka SAEs), transcoder circuits (ProtoMech; ProGenMech), and geometry-annotated features (Setlur et al., 2608.26419 — the closest protein prior art, which *annotates* ESM-2 SAE features with Cα pseudo-torsion/curvature but claims no gauge-free coordinate, no isometry, no causal or confidence axis). On structure/co-folding models: SAEs on Boltz-2 pair tensors (PairSAE, Migliorini et al., 2606.27440 — predicts the model's *own* predicted affinity, R² 0.528, no causal/leakage controls), the decodable≠steerable result on Boltz-1 (Jedryszek et al., 2608.11475), ESM3 scalar-SAE steering at L33 (Steenwyk, r=0.87), and SAEs on all-atom diffusion (VFUSE, biosecurity framing). We differentiate on the conjunction identity + cause + truth-vs-confidence with external ground truth, and on the wrap-around closure a scalar cannot fake.

**Dictionary geometry and capacity.** Block-sparse featurizers (Fel et al., 2606.25234; critique Jerpelea & Ananthram, 2608.27515) and subspace-aware SAEs (SASA/Dal26, 2606.06333, NeurIPS 2026 — the covering-number theorem + O(1)-block remedy) own the capacity story; mixed-curvature VAEs (Skopek 2019) own learnable curvature and document its ill-conditioning. We **concede** all of this: we use fixed blocks as a *tool*, compare at matched bits, and report the curved-ties-flat tie (confirmed on synthetic data) as a control, not a contribution.

**Steering, persistence, and the decodable≠causal gap.** Readable-before-causal in LLMs (lagged coupling, Xun26, 2026); off-manifold collapse (Zha26g, 2608.18597) and masked-pLM edit persistence (Hin26) forecast that mid-computation edits wash out under iterative decoding. We extend these from one layer / ESM-2 / Boltz to a layer-resolved persistence map on ESM3 structure tokens and report the negative honestly.

**Circular features and physical periodicity.** Circular representations in LLMs (Engels et al., modular arithmetic), and two NeurIPS-2026 codon-frame papers that own the DNA period-3/Z₃ story end-to-end `[CONCEDE the codon leg]`. We differentiate on structure-derived phase slips (kinks, 3₁₀) that positional encoding cannot manufacture — the one periodicity signal RoPE cannot fake.

**Benchmarks and cautions.** The field's own reviews (Orlov et al., "What Do Biological Foundation Models Compute?", bioRxiv 2026) and ground-truth-SAE-eval work flag circular validation against trained-on databases, missing random-init baselines, and feature death. Our exact-geometry suite (physics-computed targets, never database labels; RoPE-proof protocol; nuisance-only baseline) is a direct response to open gap #8 (no standardized interpretability benchmark for bio models) and #3b (the continuous/periodic slice with exact geometric ground truth).

**Must-read-before-submission (human pass).** The 2026 LLM truth-vs-confidence papers in full (the nearest conceptual scoop); goodfire.ai/research for Oct–Nov 2026 posts; a reported BioEmu+ESM3 SAE-steering camera-ready and HyperSAE `[UNVERIFIED]`; dataset licences (§13). See §15 for the full owed-human-pass list.

*(A one-line prior-art out-positioning ledger is in Appendix D.)*

---

## 10. FIGURE LIST AND THE MONEY FIGURE

- **Fig 1 — Identity.** Recovered (φ,ψ) torus overlaid on the held-out Ramachandran plot, colored by geodesic error; inset: scalar-SAE fragmentation of the same angle into arcs at matched bits. *Shows C1 — the coordinate is the physical one, not a reconstructing subspace.*
- **Fig 2 — Causal wrap-around (crown-jewel mechanism figure).** Model's own output dihedral vs steering angle Δ, tracing a closed loop that returns to baseline at 2π; scalar and matched-random overlaid flat. *Shows C2 — the signature a 1-D latent cannot fake.*
- **Fig 3 — Trust residual / confidently-wrong (THE MONEY FIGURE).** Scatter of cases where the model's confidence is high but the true physics is atypical, with the recovered truth coordinate still tracking truth; adjacent ROC showing the trust residual flags these calibration failures with higher AUROC than confidence alone. *Shows C3 in its useful form — the internal truth axis knows when the deployed confidence head is confidently wrong.* This is the panel a non-bio ICML reviewer remembers; it is also the auditing-primitive figure and the dual-use-positive artifact. It foregrounds the **four LLM-impossible differentiators** (external physical truth; deployed calibrated confidence; causal dissociation + wrap-around; cross-track invariance).
- **Fig 4 — RoPE-proof phase slip.** Feature phase vs residue index with structural slip events (kinks, 3₁₀) marked; the feature tracks the slip, index-mod-3.6 does not. *Shows C4 — biology, not position.*
- **Fig 5 — Persistence map (honest negative).** Layer (× decode-step) heatmap of decodability, steerability, and persistence, with three distinct argmaxes. *Shows C5 — decodable ≠ steerable ≠ persistent; where edits survive.*
- **Fig 6 — Invariance.** Side-by-side triad summary for the (φ,ψ) torus, the helix-phase circle, and the probe-level Kd-vs-ipSAE vignette, plus the ESM-2/ESM3 generality panel and the mismatched-variable (SASA) double dissociation. *Shows C6 — a property of the model class, not a cherry-picked feature.*
- **Fig S1 — Synthetic calibration** (Phase 0 / MPU-0): curved-ties-flat at matched bits, scalar covering curve, angle recovery corr 0.9987, the planted truth-vs-confidence separation across the tunable ρ. *The pre-registered control face of the theory.*

**The money figure is Fig 3.** Rationale: Fig 1 (identity) and Fig 2 (wrap-around) are beautiful but read as "re-finding physics" to a skeptic; Fig 3 is the one that converts the apparatus into a general, useful claim about foundation-model epistemics, is legible to a reviewer with no structural-biology background, and is the artifact the abstract and the auditing contribution both point at.

---

## 11. RISK-MANAGED CRITICAL PATH, BACKWARD TIMELINE, AND THE PIVOT DECISION POINT

*(Reconciles the architect's backward-from-ICML timeline with the adversarial-reviewer pass's kill-gate ladder. Both order the gates by cost of the decisive test, cheapest first, so the ~800–1500 GPU-h steering budget is never spent until four CPU/probe-level gates clear. The decision gate is unified at mid-December — see the merge note in §0.)*

**Deadline anchor (reported, `[UNVERIFIED]` against the official call):** ICML 2027 abstract ~14–16 Jan 2027, full paper **~22 Jan 2027** (AoE); ICLR 2027 workshops ~1 Feb 2027; NeurIPS 2027 abstract ~mid-May 2027. Today is 2026-11-22 → **~8.5–9 weeks to the ICML full-paper date.** Three facts force the reduced cut: (a) ESM3-open gated-weight access is *not yet secured*; (b) Track B is leakage-heavy, dual-use-sensitive, and owes a human pass; (c) the full causal two-track program is 4–8k GPU-h. The ambition (the headline C1–C3 + the trust axis) is kept; the surface area is cut to what can be made airtight by 22 Jan.

### The kill-gate ladder (cheapest-decisive first)
```
WEEK 0–1  (CPU + cheap GPU; no steering)
  G0  synthetic matched-bits + identifiability + torus factorization + MPU-0 confidence channel
        [DONE in part: Phase 0 passed; torus-factorization (G0b) + confidence-channel (MPU-0) still to add]
  G-B1 Track-B Kd-leakage probe (parallel; cheap GPU; probes only)            <- kills Track B early if leakage
WEEK 1–3  (probes on cached ESM3 activations; no steering)
  G1  RoPE / positional control (helix-vs-coil at matched index; trained-minus-random;
        structure-phase-slip partial corr)                                    <- kills the periodic claim if positional
WEEK 3–5  (ONE decode cell; the pivotal gate)
  G2  persistence / commitment on the single best (layer,variable) cell       <- MOST LIKELY DEATH of the causal half
  G-B2 ipSAE-collinearity (Kd direction vs ipSAE direction; angle only)       <- kills Track-B truth/confidence if collinear
  C1 + G5  identity up to isometry (held-out) + confidence-matched pairs + confidently-wrong  <- THE HOOK GATE
WEEK 5–7  (expensive; only if G1+G2 GO)
  G3  decodable != steerable efficacy map + geometry-lift at matched norm
WEEK 7–8  (expensive; only if G2+G3 GO)
  G4  wrap-around dose-response (crown jewel)
  G-B3 ipSAE-Goodhart + SKEMPI agreement (Track B; DEFERRED — not in the ICML 2027 submission)
WEEK 8–9  write-up, >=5-seed sweeps on whatever cleared, pre-registered predictions frozen, dual-use human pass
```

**The three gates most likely to kill, earliest testable** (each converts to a publishable result, not a dead end): **G2** (persistence — kills the causal half; test right after G1), **G5** (the entanglement rebuttal — kills the hook if truth and confidence are entangled on-distribution), **G1** (RoPE manufactures the periodicity — test before G2). **G-B1** (Kd is leakage) kills only the Track-B vignette, not the paper.

### Backward milestone timeline (architect spine)
| By when | Milestone | Gate / decision |
|---|---|---|
| **now – 29 Nov** | Human: secure ESM3-open weight access + confirm licence; verify RoPE usage, d_model, confidence-source on seq-only input; run MPU-0 (synthetic truth-vs-confidence metric de-risk) | MPU-0 GO (metric behaves) |
| **29 Nov – 6 Dec** | Track A caching (8 layers, ~5k chains, seq-only) + DSSP ground truth; **G1 RoPE/positional probes (no steering)**; in parallel Track B **G-B1 leakage probe** + **G-B2 ipSAE-collinearity** (both probe-only) | G1 GO; G-B1/G-B2 decide if Track B stays as a vignette or drops to an appendix negative |
| **6 Dec – 13 Dec** | **G2 persistence gate** on the single best (layer) cell (one decode cell, days on 1 GPU) | **G2 GO/NO-GO — the fork:** GO → causal half alive; NO-GO → paper becomes identity + truth + honest persistence negative |
| **13 Dec – 19 Dec** | Phase-2 identity (C1) end-to-end on held-out; **G5 confidence-matched-pair + confidently-wrong** (the hook gate) | **~19 Dec DECISION GATE** (reconciled, evaluates G1+G2+C1+G5 and the Track-B gates): if the gates close cleanly → commit ICML 2027; else → retarget NeurIPS 2027 |
| **19 Dec – 3 Jan** | If committed: Phase-3 wrap-around (C2) + decodable≠steerable map (G3/G4) on the best cell; RoPE-proof phase-slip panel (C4) | G3/G4 GO → C2 figure; NO-GO → C2 reported as negative, C1+C3 stand |
| **3 Jan – 12 Jan** | Phase-4 partial: ≥5-seed + matched-bits + random-init on Track A; C6 (helix-phase + ESM-2/ESM3 panel + Kd probe vignette); freeze all pre-registered predictions | G-final: survives seeds/bits/random-init on Track A end-to-end |
| **12 Jan – 22 Jan** | Write-up; all six figures; dual-use statement human-passed; reproducibility appendix; **abstract ~14 Jan, paper ~22 Jan** | submit |

### The pivot decision point — stated plainly
**Decision gate, ~19 Dec 2026 (after G1 + G2 + C1 + G5, and the Track-B gates, report):**

- **If G1 GO and G2 GO** (persistence holds) and C1 + G5 close cleanly: proceed to G3/G4; an ICML 2027 full-paper submission is *attemptable* on the aggressive schedule — but only if G3 clears by end of Week 7 and ≥5-seed sweeps finish by the full-paper date. This is the ambitious path and it is not softened; it is simply gated on the data cooperating within ~8.5–9 weeks.
- **If G2 NO-GO** (edits wash out — the forecast outcome): **do NOT chase ICML with a broken causal half.** Pivot the causal half to the **honest persistence/commitment negative** (C5 headline) and ship the **MPR benchmark + negative paper** (§12). Target: **NeurIPS 2027 main (~May abstract), with the NeurIPS Datasets & Benchmarks track as the floor**, and an **ICLR 2027 workshop (~Feb) stake** in the meantime for priority + feedback. Not a reduction in ambition — the correct paper for the result the experiment gave, and the field actively wants the negative.
- **If G1 NO-GO** (periodicity is positional): retract the periodic claim, keep non-periodic SASA as the honest headline variable + the RoPE-artifact caution; same NeurIPS/benchmark pivot.
- **If G5 NO-GO** (truth and confidence entangled on-distribution): the entanglement rebuttal wins; reframe the hook to the calibration/benchmark paper and report the honest entanglement negative (still the control the field needs).
- **If G-B1 or G-B2 NO-GO:** Track B becomes an appendix leakage/Goodhart negative; the paper is Track-A-only; the venue decision is driven by G1/G2/G5 as above.

**The blunt AC statement.** ICML 2027 main by late January is the **upside case, conditional on G1+G2+G3 (and G5) all clearing on an aggressive gated schedule with no reruns.** The modal outcome, given that two papers forecast the G2 wash-out, is the strong negative/benchmark paper at **NeurIPS 2027 (main or D&B) with an ICLR 2027 workshop stake.** Plan for NeurIPS, attempt ICML only if the mid-December gate is green. Do not let the ICML date pressure the team into skipping the ≥5-seed / validity-filter / leakage controls — a rushed ICML submission that drops those is a ~15% paper; the same work done right is a ~35–45% NeurIPS paper plus a near-certain workshop/D&B result.

---

## 12. SCOPE DISCIPLINE AND THE MINIMUM PUBLISHABLE RESULT

*(Integrated from the adversarial-reviewer pass §5 and the architect's MPR. The single biggest failure mode is a sprawling paper that doubles its attack surface. Decisions, made — not deferred.)*

- **Jacobian / J++ transport half: FULLY OUT.** Confirmed from the prior passes and locked. Reasons: Ostrow (mean-Jacobian need not resemble any individual; prominence unrelated to variance explained) likely fails on bidirectional ESM3; ill-defined across the AlphaGenome U-Net neck and on bidirectional ESM3; naming collision with the categorical Jacobian (Zhang PNAS 2024, nucleotide-dependency maps); must beat attribution patching (not just activation norm) to matter; owned in spirit by Aghazadeh's transcoders. **Not even an appendix in the ICML submission** — a separate follow-up.
- **Genomic generality panel (Evo 2 / AlphaGenome / codon phase): OUT of the ICML paper.** Cross-track invariance (C6) is carried by Track A (structural) vs Track B (thermodynamic), which is cleaner and on-theme for truth-vs-confidence. A third modality adds scope, scoop exposure (Goodfire owns Evo 2), and the codon leg is already owned by two NeurIPS 2026 papers. Keep it as a one-paragraph "we expect this generalizes" + future work, with at most a single exploratory figure if time permits — never load-bearing.
- **Track B (Boltz / Kd vs ipSAE): a GATED EXTENSION, not the spine, relegated to an appendix / one figure if it does not clear its gates.** The paper must be acceptable on Track A alone (C1–C5 on ESM3). Track B is admitted to the main body only if **G-B1 (Kd is learning, not leakage)** and **G-B2 (Kd direction ≠ ipSAE direction)** both clear; then it supplies C6 and the single most quotable confidently-wrong instance (high ipSAE, weak Kd). If either gate fails, Track B is a one-figure leakage/Goodhart **negative** in an appendix, and the paper is Track-A-only. *Rationale:* leakage-prone, dual-use-sensitive, compute-heaviest (mid-diffusion causal + SKEMPI + FK baseline), and the Boltz lane is crowded (PairSAE, Jed26b). It is where the generality and the best example live, but it must never be able to sink acceptance.
- **Models / layers.** **ESM3-open (1.4B, 48 layers) is the one load-bearing model.** Do NOT sweep all 48 layers for the headline: pre-register a **coarse 6–8-layer grid** (e.g. ~L12/18/24/30/36/42 of 48; exact grid `[UNVERIFIED]` pending d_model confirmation), fix the causal-campaign layer by the *decodability* gate (G1) before any steering, then do a fine sweep only around the winner for the efficacy-ratio map (G3a). An **ESM-2-vs-ESM3** generality panel is **appendix-only and optional** (scoop-exposed to Steenwyk/Whitfield; keep minimal). Boltz is the only Track-B model; AF3/Chai are named alternates, not run.
- **Geometry machinery.** Fixed/assigned 2-D block only; no learnable curvature (Skopek; Phase 0 radial-noise penalty). The torus factorization (genuine product-manifold, not PCA — Phase 0 flagged PCA fails, per-angle corr 0.6146/0.5934) is the one new piece of machinery and is a gated deliverable (G0b on synthetic, then on ESM3). Everything geometric is a control, never a claim.

### The Minimum Publishable Result (if the causal claims fail)
A **leakage-aware, exact-geometry benchmark + calibration/negative paper**, any subset of which publishes (NeurIPS Datasets & Benchmarks is the explicit floor), reachable from the cheap gates alone and robust to a null on the causal half:

1. **The physical-ground-truth suite + RoPE-proof protocol + nuisance baseline** — a reusable benchmark the field lacks (open gap #8). Reviewer-proof because it concedes SASA/Steenwyk/Skopek up front.
2. **Identity-up-to-isometry recovery** (C1 + the torus factorization) — with the honest "curved ties flat" control result crediting SASA.
3. **The ESM3 commitment/persistence negative** (C5/G2) — extends Hin26/Zha26g/Jed26b to a layer map on ESM3 structure tokens. A genuinely wanted negative.
4. **The truth≠confidence calibration** (C3/G-B2) — the angle between the physical-quantity direction and the model's deployed confidence (or the honest finding they are collinear) — a trust/audit result, plus the trust-axis monitor.

This MPR is reachable even if C2/C4/C6/G-B3 all NO-GO, and it is scoop-robust by construction. It is an ICML-borderline / strong-NeurIPS paper on its own.

---

## 13. REPRODUCIBILITY, COMPUTE BUDGET, AND THE DATA / LEAKAGE-AUDIT PROTOCOL

### Reproducibility (what a reviewer must be able to trust)
- **Release:** the ground-truth suite (code that computes (φ,ψ)/helix-phase/SASA from experimental coordinates via DSSP/mdtraj), the exact split definitions (time + cluster), the nuisance-baseline code, cached-activation hashes, all seeds, the pre-registration document (this §5), the analysis code, and the trust-axis monitor. The point of an exact-physics benchmark is that it is deterministic and re-runnable. **Not released:** any affinity-maximizing steering pipeline (§15).
- **Pre-registration:** §5 thresholds, nulls, tests, and corrections are frozen before any ESM3 run and included as a supplement; deviations are logged. The single highest-credibility move available and the direct answer to R4's garden-of-forking-paths objection.
- **Seeds/stability:** ≥5 seeds everywhere (`base.yaml` seed list `[0,1,2,3,4]`); report across-seed SD and mean−2·SD margins; Tournament-TopK for block selection. Deterministic cuDNN where feasible; log torch/esm/boltz/geoopt/CUDA versions into each run's `run_meta.json`; content-hash every activation shard and ground-truth table into a `MANIFEST`. A random-init same-architecture control model instantiated once per track; every headline reported as trained−random gap. Held-out splits frozen and hashed before any fitting. Config-driven, no magic numbers; each phase emits one `REPORT.md` + figures.
- **Environment / tooling ledger:** ESM3-open `esm3-sm-open-v1` (pip `esm`; HF gated, non-commercial — access `[UNVERIFIED]`, d_model≈1536 `[UNVERIFIED]`); Boltz-2 MIT `[CONFIRMED]`; ipSAE `DunbrackLab/IPSAE` + PyPI `ipsae` 1.0.1 `[CONFIRMED]` (contrast only); geoopt `[CONFIRMED]` (used minimally — projected-renormalize preferred over exp/log); DSSP/mdtraj/FreeSASA/ProteinMPNN standard; PLINDER `[CONFIRMED split machinery]`; PDBBind/BindingDB/SKEMPI/CASF counts+licences `[UNVERIFIED — verify before use]`; ripser/gudhi for Betti. ESM3-open is gated/non-commercial — access terms are a human action item (§15).

### Compute budget (order-of-magnitude, `[UNVERIFIED]` — dominated by activation caching + interventions, not by block training)
- **Phase 0 / G0 / MPU-0:** CPU, hours. Done in part.
- **G-B1 (Kd leakage probe):** cheap GPU, probes only.
- **G1 (RoPE probes on ESM3):** activation caching + probes, ~30–80 GPU-h, 1×24 GB feasible.
- **G2 (persistence, one cell):** steer + full iterative decode + refold, ~1 GPU, days (~30–80 GPU-h Track A; Track-B pilot ~80–200 GPU-h *if run*).
- **Phase 2A identity + truth⟂confidence:** ~100–250 GPU-h; **Phase 2B probe-only Track B:** ~150–400 GPU-h (1–2×80 GB).
- **G3/G4 (ESM3 steering sweep):** ~100–250 GPU-h, 1–2×80 GB; the layer × 3-direction × α × seed campaign + validity pass is the bulk of the protein budget.
- **Track B causal (G-B2/G-B3, Boltz mid-diffusion + SKEMPI + FK baseline):** heaviest, up to ~1.5–3k GPU-h; **DEFERRED — out of scope for this submission.**
- **Full ≥5-seed / matched-bits / random-init ensemble:** small cluster, ~4–8k GPU-h — but the **core result needs far less**; the ensemble is robustness, not the headline, and is *not* on the critical path.
- *Budget discipline:* four CPU/probe gates (G0, G-B1, G1, G-B2) gate the entire steering budget; if any fails, the expensive campaigns never launch. Activation caching is a one-time cost per corpus; dictionaries/probes train on the cache with no grad through the big model; interventions (re-running the model) are the real compute sink, which is why persistence (G2) is gated first.

### Data / leakage-audit protocol (the part a reviewer will not take on faith)
1. **Two independent splits, both enforced:** (a) **time split** on PDB deposition date (train strictly pre-cutoff, test strictly post-cutoff — post-cutoff structures the model could not have trained on); (b) **sequence-cluster split** (MMseqs2 ≤30% identity; no cluster spans train/test). Report every headline number on **both**, and on the **intersection** (post-cutoff AND novel-cluster) as the strict condition.
2. **Explicit leakage probe (upper bound on memorization):** a baseline seeing ONLY nuisance identifiers — train-membership flag, sequence/ligand cluster ID, length, (Track B) ligand MW/heavy-atom count, protein family — and no learned geometry. Every feature claim must beat this probe on the clustered/time split by the pre-registered margin; a random-split win that collapses to the leakage probe under clustering is reported as leakage, not learning.
3. **Track B Kd specifics:** target is **experimental Kd only, never the model's own predicted affinity** (the PairSAE R²=0.528 circularity); leakage-aware PLINDER/time splits; SKEMPI ΔΔG as a built-in mutation-level causal anchor (deferred to the causal extension). Dataset counts/licences (PDBBind, PLINDER, BindingDB, SKEMPI, CASF) are **UNVERIFIED** and owe a human a pass before use.
4. **Validity gating of all steered data** (§6 item 9) so "predictive/causal" features are never reading off-manifold garbage.
5. **Decode-fidelity floor** reported and subtracted for all Track-A causal Δ(φ,ψ).

---

## 14. HONEST LIMITATIONS AND THE REBUTTAL THAT CANNOT YET BE CLOSED

*(Integrated from both passes: the architect's §10 limitations and the adversarial-reviewer pass's §4, which names the single rebuttal that cannot yet be closed on paper.)*

### 14.1 The rebuttal that cannot currently be answered: "truth and confidence are one entangled signal" `[CONCEDE — open]`

> *"Your truth axis and your confidence axis are two readouts of the same underlying 'this region is well-structured' signal. pLDDT is high exactly where backbone geometry is regular, so (φ,ψ) and pLDDT are entangled in the data distribution; a principal angle > 0 and a 'double dissociation' can both arise from fitting two linear heads on one correlated latent, and your steering may not be selective enough to prove otherwise."*

This is the crux risk and it is **not closed on paper today.** Two prongs:

- **The entanglement prong.** The double dissociation only partly answers it (selectivity is a matter of degree). The real answer is the **confidence-matched negative-pair control** + the **confidently-wrong slice** (high pLDDT, atypical dihedral; high ipSAE, weak Kd). *The problem:* for **Track A** there may not be enough high-pLDDT/atypical-geometry residues to power the slice, and for **Track B** leakage muddies it. We cannot currently guarantee the effect survives on those slices.
  - **Minimal experiment that closes it.** (1) A **power analysis up front**: count, in the held-out set, residues in the high-confidence × atypical-geometry cell; pre-register the minimum n for 80% power to detect AUROC 0.70 vs 0.5 (DeLong). If Track A cannot supply it, the confidently-wrong argument must come from Track B, which makes Track B less optional than the plan wants — flag. (2) **MPU-0 synthetic calibration:** plant a torus AND an orthogonal "confidence" channel in R⁶⁴ with a *tunable correlation ρ*; show the confidence-matched control recovers the planted separation across ρ, and measure the ρ at which the method can no longer distinguish "two readouts of one latent" from "two latents." This calibrates the claim's resolution before any model run. (3) The **causal** selectivity test (steer truth at a confidence-matched operating point and show pLDDT does not move beyond the random band) is the part a "two-readouts-of-one-latent" account cannot mimic — pre-register it as the decisive cell.
- **The scoop prong.** Truth⟂confidence is a live 2026 LLM topic (2603.25052; "Confidence Manifold" 2602.08159; "Truthfulness Spectrum" 2602.20273; ACL-2026 "LLM Beliefs Are in Their Heads" — **ALL `[UNVERIFIED]` beyond abstracts; a human must read the primary texts before committing the hook**). The "Truthfulness Spectrum" result that "truth is not one direction" is itself a warning that the bare separation is contested even in LLMs.
  - **Minimal experiment / framing that closes it.** Foreground, in the abstract and Fig 3, the **four differentiators LLM work structurally cannot have**: (i) truth pinned by **external physics** (not answer-correctness in the model's own domain); (ii) a **deployed, calibrated confidence head** (not an elicited/probed confidence); (iii) a **causal double dissociation + wrap-around** (not a probe); (iv) **cross-track invariance**. The experiment that makes this concrete is the **trust-axis auditing primitive**: a per-input trust residual = the component of the confidence head NOT predictable from the recovered truth coordinate, and a test that it flags confidently-wrong cases **better than the confidence head alone** (AUROC with DeLong CI). This exports back to LLMs as a construction — the paper's reach — and it is the thing no LLM truth/confidence paper has, because they lack an external truth anchor to form the residual against.

**Honest statement for the limitations section:** *"We show the truth and confidence coordinates are geometrically and causally dissociable on our distribution and on confidence-matched pairs; we do not claim they are dissociable on all inputs, and where the model's confidence and the physical regularity of the structure are perfectly correlated, our method cannot separate their representations — we report the correlation regime in which the separation is resolvable."*

### 14.2 Other limitations forced into the paper (so a reviewer cannot "discover" them)
- **The causal half may be a negative.** G2 persistence is the most likely death; the paper is written so a null there is a headline calibration result, not a failure. State up front — it reads as rigor, not weakness.
- **"Recovery of physics," not discovery.** Conceded explicitly; the contribution is epistemic structure + measurement rigor, with one exploratory discovery hedge.
- **The identifiability claim is operational, not a theorem** — recovery up to isometry on held-out data under a train-fit alignment plus planted-synthetic validation, not an asymptotic guarantee (§3.7). We assume the quantity's symmetry group matches the block gauge and that the generative map is injective on the manifold. Scope the word "identifiable" precisely in the claim.
- **The truth⟂confidence hook rests on gate G5**; if the confidence-matched-pair / confidently-wrong tests do not separate, the entanglement rebuttal stands and we report the honest negative. Disclosed as the crux risk (§14.1).
- **pLDDT may be the wrong confidence object for a per-residue dihedral** (R2's O2.5). The cleaner contrast may be Track B (ipSAE is literally an interface-quality score for the Kd object). If the Track A contrast is weak, the hook leans on Track B — which raises the leakage/ethics stakes. Flagged.
- **The confidence source on sequence-only input is unverified** — if ESM3 does not emit a usable per-residue pLDDT there, we use structure-token entropy, a weaker but still deployed signal; we disclose which.
- **ESM3 structure-token → (φ,ψ) decode fidelity is `[UNVERIFIED]`** at the chosen length regime; gated by the decode-fidelity calibration.
- **Curved-vs-flat is a control, not a result** — no geometry contribution is claimed (SASA owns capacity).
- **Track B is probe-only in this submission** — the binding generality is a vignette, not a causal result; its causal extension is deferred (also for the ethics reason in §15).
- **RoPE dependence** of the phase-slip framing is contingent on confirming ESM3 uses RoPE `[UNVERIFIED]`.
- **Every venue date, the LLM scoop papers, d_model (~1536), and all dataset counts/licences are `[UNVERIFIED]`** and owe a human a primary-source pass (§15).

---

## 15. ETHICS / DUAL-USE STATEMENT, AND THE OWED HUMAN / COMPLIANCE PASS

> This work is framed strictly as **auditing and calibration** of generative biological models, not as a design tool. Track A (backbone dihedral angles, helix phase, relative SASA) concerns low-level structural geometry and carries negligible misuse potential. Track B (interface binding) is capability-adjacent — locating where a model represents binding strength is one step from optimizing it — so for this submission it is constrained to the strictest defensive posture: **no steering at all**, only probe-level direction-angle measurements on benign, leakage-split complexes and one read-only confidently-wrong example; experimental Kd and model-independent geometric descriptors (buried surface area, contacts) are the only targets; the model's own confidence (ipSAE/ipTM) is used **only as a contrast, never as an optimization objective** (steering toward a model's own confidence is both circular and a Goodhart hazard); evaluation is restricted to benign, open complexes with hazard-adjacent classes excluded (toxins, viral entry/fusion, immune-evasion, select-agent-adjacent organisms) via a hard allow/deny list; and **no affinity-maximizing steering pipeline and no per-step binder-optimization recipe is built, run, or released.** The primary value of locating where a property is causally and confidently encoded is defensive: it yields a **trust-axis monitor** that reads a design model's internal truth coordinate independently of its gameable confidence head, flagging "confidently wrong" outputs for a deployer.

**Owed human / compliance pass (submission blocker, not optional).** An automated safety classifier repeatedly truncated earlier attempts to draft this statement, so the final dual-use section — and the specific exclusion list and release policy for Track B — **must be written and signed off by a person / compliance review before submission.** This document provides the scope boundaries that review should enforce, not a substitute for it. Additionally, a human must, pre-submission:
- read the primary texts of the 2026 LLM truth/confidence papers (the nearest scoop) `[UNVERIFIED]`;
- confirm ESM3-open weight access/terms and d_model (~1536) `[UNVERIFIED]`;
- confirm ESM3 uses RoPE and whether it emits a usable per-residue pLDDT on sequence-only input `[UNVERIFIED]`;
- verify all dataset counts/licences (PDBBind, PLINDER, BindingDB, SKEMPI, CASF) `[UNVERIFIED]`;
- verify all venue deadlines (ICML/ICLR/NeurIPS 2027) `[UNVERIFIED]`;
- check goodfire.ai/research (Oct–Nov 2026) + the reported BioEmu+ESM3 SAE-steering camera-ready + HyperSAE for a live scoop `[UNVERIFIED]`.

---

## 16. REVISION R1 (2026-11-23) — on/off-target specificity, PPI interface, and thermodynamics as layer-resolved targets

**What this adds and why.** The project's spine is "where does a physically-defined quantity live in the model,
is it identifiable, does the model use it, and is it separate from the model's confidence." R1 extends the set of
quantities from backbone geometry/dynamics to the quantities that matter for **interactions and stability**, and
makes the **per-layer localization map ("which layers encode X")** a first-class, repeated deliverable across all
of them — the literal realization of the project title *where structure lives*. The method (identity → cause →
truth-vs-confidence) and all controls are unchanged; these are new target quantities, mostly on Track B.

**Scope guardrail (unchanged posture).** For the ICML 2027 submission these additions are **probe-only and
localization-first — no steering.** The spine remains Track A (structural geometry in ESM3). The expensive
causal/steering versions carry the heaviest dual-use weight and are **deferred** (consistent with §12 scope and
§15 ethics). ipSAE / ipTM / pLDDT remain **confidence contrasts, never labels or steering targets**; all binding
quantities use **experimental** values, never the model's own predicted affinity (the PairSAE lesson).

### 16.1 On-target vs off-target (binding specificity)
- **Question.** Which layers encode discrimination between a cognate (on-target) partner and a decoy / off-target
  partner — and is the model's confidence calibrated to true specificity or merely to affinity magnitude?
- **Ground truth.** Labeled binders vs non-binders: PLINDER / PDBBind actives vs **property-matched** decoys
  (DUD-E-style, matched on physicochemical properties so the probe cannot cheat on size/charge); for PPI, supported
  interacting pairs vs curated negatives (e.g. Negatome / property-matched random pairs). `[availability UNVERIFIED]`
- **Deliverable.** A **per-layer specificity-AUROC curve** (on vs off target) → the depth where discrimination
  peaks ("where specificity lives"); identity of the specificity direction; a truth-vs-confidence contrast
  (does ipSAE/ipTM track true specificity or just affinity?). Causal test **deferred/probe-only**.
- **Controls.** Property-matched decoys (not random); leakage-aware time + scaffold splits; nuisance-only baseline;
  random-init; the model's own interface score as a confidence contrast.
- **Dual-use.** Closest to binder-design capability of anything here → **probe-only, benign complexes, no steering**
  toward altered on/off-target binding. Framed strictly as "where is specificity represented, and is confidence
  calibrated to it." Owes the §15 human/compliance pass.
- **Gate G-spec.** GO if the specificity probe beats the property-matched-decoy baseline and random-init by a
  pre-registered margin on the leakage split; else report the negative (specificity is not linearly localized).

### 16.2 PPI interface features
- **Quantities.** Per-residue is-interface, interface contact count / buried surface area, and interface
  **hot-spots** (residues whose mutation costs binding), with ΔΔG from SKEMPI.
- **Ground truth.** Computed from experimental complex structures (interface residues, BSA) and SKEMPI (hot-spot ΔΔG).
- **Deliverable.** Per-layer map of where interface identity and interface geometry are encoded; identity of the
  hot-spot direction; truth-vs-confidence vs ipSAE. Causal test **deferred/probe-only**.
- **Controls.** Interface labels from structure (not the model); sequence/position baseline; leakage splits;
  random-init; nuisance-only baseline.
- **Gate G-ppi.** GO if the interface probe beats a sequence/position baseline and random-init per layer.

### 16.3 Thermodynamics
- **Quantities.** Binding free energy ΔG / Kd (continuous), mutational binding ΔΔG (SKEMPI), and fold **stability**
  — melting temperature Tm / folding ΔG — where data exists (mega-scale stability datasets; `[availability UNVERIFIED]`).
- **Deliverable.** Is there an **identifiable thermodynamic axis** (a scalar coordinate; "isometry" here = a
  monotone reparametrization, scored by held-out Spearman)? Which layers carry it? Is it separate from the model's
  confidence (a ΔG axis vs pLDDT/ipSAE)? Causal test for a scalar = monotone dose-response, **deferred**.
- **Controls.** Leakage-aware time/scaffold splits + explicit leakage probe; affinity-matched negatives;
  random-init; **experimental** ΔG/Kd only (never the model's predicted affinity).
- **Gate G-thermo.** GO if the experimental-ΔG/Kd probe beats the leakage probe by a pre-registered Spearman margin.

### 16.4 How R1 slots into the existing claims, phases, and gates
- **Claims.** Identity (C1) and truth-vs-confidence (C3) extend to each new quantity. Wrap-around (C2) applies only
  to genuinely **cyclic** quantities (angles), **not** to scalars (ΔG, specificity) — for scalars the causal test is
  monotone dose-response and is deferred. Cross-variable invariance (**C6**) now spans {(φ,ψ)-torus, helix-phase,
  specificity, interface, ΔG} — strengthening the anti-cherry-pick argument with structural *and* interaction *and*
  thermodynamic variables.
- **Localization map.** A per-layer "which layers encode X" curve is produced for **every** quantity (dynamics/
  angles, on/off-target, PPI interface, thermodynamics). This layer-resolved atlas is a headline deliverable in its
  own right and is the public repo's framing ("where structure lives").
- **Phases.** All R1 work is **probe-level** and lands in Phase 2 (identity + localization + truth-vs-confidence) on
  Track B; no new steering phase is added for the ICML submission. New probe-level gates **G-spec, G-ppi, G-thermo**
  run in parallel with the Track-A gates and feed the mid-December decision gate (§11).
- **Models / data additions.** A co-folding model (Boltz) for complexes + interface scores; SKEMPI for ΔΔG /
  hot-spots; PLINDER/PDBBind + property-matched decoys for specificity; stability datasets for Tm/ΔG. All counts and
  licences `[UNVERIFIED]` — verify before use.
- **Cost / risk caveat.** R1 broadens Track B's **data engineering** (the plan's acknowledged top effort-sink) and
  its **leakage surface**; keeping it probe-only and localization-first is deliberate so it does not threaten the
  Track-A spine or the deadline. The expensive causal/steering versions of specificity and thermodynamics stay
  deferred and are the most dual-use-sensitive parts of the whole program.

### 16.5 Metric/engineering note (from the first-phase build)
The synthetic de-risk (MPU-0) passed: the circular coordinate is recovered up to isometry (~0.999) and the trust
residual flags confidently-wrong cases in the partial-coupling regime, degrading only at full truth/confidence
entanglement (the §14.1 limitation, now demonstrated quantitatively). The reusable metrics core (circular identity,
trust residual, AUROC) is implemented and tested in the public repo and will carry directly to the R1 scalar
quantities (with the scalar analogues: monotone/Spearman identity, and the same trust-residual construction against
pLDDT/ipSAE).

### 16.6 Datasets (reconciled 2026-11-23; full tables, sizes, licenses in the public repo `docs/datasets.md`)
Decided from three independent reviews (facts/sizes, interpretability suitability, adversarial reviewer). Roles,
cleanliness-ranked (lowest leakage/bias first), confirm the Track-A-load-bearing / Track-B-probe-only posture from
data properties alone:
1. **Dihedrals / SS / SASA / contacts — LOAD-BEARING.** PDB chains (PISCES/CATH cull), labels by DSSP/FreeSASA;
   cache ~200–300 non-redundant chains/split; MMseqs2 ≤30% ∩ deposition-date split; CAMEO/CASP post-cutoff
   held-out. Structure track OFF; SASA = the non-periodic double-dissociation control.
2. **PPI interface / BSA — LOAD-BEARING (probe-only).** PINDER (2.3M systems; interface-structural splits) /
   PPIRef; ~150–250 non-redundant complexes; interface-similarity split; DIPS CUT (DB5 leak).
3. **Fold stability ΔG/ΔΔG — LOAD-BEARING, the thermodynamics ANCHOR = Tsuboyama/Rocklin mega-scale (776,298 ΔG;
   479 domains; CC BY 4.0).** Single assay → clean monotone ΔG axis + dose-response + powers the confidently-wrong
   slice; WT-domain-cluster split, NEVER per-mutant; 148 de-novo designs = anti-memorization OOD. FireProtDB/
   Meltome → appendix/OOD only.
4. **Dynamics — PROBE-ONLY (localization only; drop identity).** ATLAS (1,390/1,938) or mdCATH (5,398), converged
   RMSF, fold split; B-factors control only; no clean GT.
5. **PPI affinity / ΔΔG — DEFERRED → hot-spot classification.** SKEMPI 2.0 (7,085 muts / 345 complexes),
   complex-cluster split; continuous/causal version deferred.
6. **Protein–ligand Kd — PROBE-ONLY + GATED (G-B1).** PDBBind under PLINDER PL50/TIME or LP-PDBBind splits;
   ~800–1,500 non-redundant complexes; experimental Kd only, single assay; BindingDB CUT (pooled assays, no
   structures); appendix-negative if G-B1 fails.
7. **On/off-target specificity — PROBE-ONLY + APPENDIX, most dual-use.** LIT-PCBA (+ AVE-dedup) over DUD-E;
   Bernett-2024 / Negatome for PPI; requires a protein-dependence / pocket-ablation control or the claim is dropped.

**Minimum-N principle.** The binding constraint is non-redundant CLUSTERS after leakage-aware splitting, not rows:
~150–300 clusters/split for localization+identity, and ~45–55 cases in the confidently-wrong slice (DeLong, 0.70
vs 0.50 at 80% power) for the truth-vs-confidence AUROC — a per-quantity power check is owed (the §14.1 limit made
concrete). **Auto-reject traps** (see datasets.md §8): structure-track-ON for Track A; random/legacy or per-mutant
splits on a pretraining corpus; DIPS/crystal-contact interfaces; DUD-E specificity without a protein-dependence
control; pooled-assay BindingDB; target = the model's own predicted affinity; B-factors as the sole dynamics label;
any number not reported trained−random and not beating the explicit leakage probe. **Additions:** ProteinGym (DMS),
S669/Ssym (leakage-filtered ΔΔG), FLIP, MISATO. **Verify before paper:** CATH S40 count; PINDER license; the exact
Tsuboyama Zenodo version/ML-ready count; PDBBind/CASF/FireProtDB/Negatome/Bernett counts+licenses; BindingDB
usable-Kd subset; and ESM3 `d_model` / RoPE / sequence-only pLDDT (governs the Track-A confidence object).

---

## Appendix A — Decision / gate ladder (one screen)
```
G0  synthetic matched-bits + identifiability (CPU)            DONE: GO (tie; corr 0.9987; covering penalty present)
MPU-0 synthetic truth-vs-confidence metric de-risk (CPU)     de-risk the hook's metric before any model (tunable rho)
G0b torus product-manifold factorization (ISA / TC penalty)  gated deliverable; identity does not depend on it
   || G-B1 Kd-leakage probe (cheap GPU)  || G-B2 ipSAE-collinearity (cheap GPU)   [decide Track-B vignette vs drop]
 -> G1  RoPE/positional control (probes only, no steering)                         [kills death: position-artifact]
 -> G2  persistence/commitment (ONE decode cell)             <- MOST LIKELY DEATH; the fork in the road
 -> C1  identity up to isometry (held-out)
 -> G5  confidence-matched pairs + confidently-wrong          <- THE HOOK GATE (promoted to must-pass)
 -> ~19 Dec DECISION GATE: G1+G2+C1+G5 clean -> commit ICML 2027; else retarget NeurIPS 2027
 -> G3/G4 decodable!=steerable + wrap-around dose-response (expensive; only if G2 GO)
 -> Phase 4 seeds/bits/random-init + C6 invariance + write-up
 (G-B3 ipSAE-Goodhart + SKEMPI + Feynman-Kac — DEFERRED, not in the ICML 2027 submission)
```
Three gates most likely to kill, earliest testable: **G2** (persistence — kills the causal half), **G5** (the entanglement rebuttal — kills the hook if truth and confidence are entangled on-distribution), **G1** (RoPE manufactures the periodicity). G-B1 (Kd is leakage) kills only the Track-B vignette, not the paper.

## Appendix B — Phase 0 numbers as a control result (already in hand)
- Clean circle: FVU curved 0.1361 vs flat 0.1338 (Δ=+0.0022) → **tie** (curved is a control).
- Radial-noise circle: FVU curved 0.1526 vs flat 0.1313 (Δ=+0.0213) → **flat strictly better** → an extra reason to use a fixed block and never pitch curvature.
- Identifiability up to isometry (circle): angle-recovery corr **0.9987** → metric validated.
- Scalar covering penalty: **16** k=1 atoms to match **1** 2-D block at ε≈0.164 → topological penalty present (you cannot tile a circle with a bounded number of monotone 1-D features without a seam).
- Torus caveat: naive PCA recovers the 4-D subspace (FVU 0.0695) but does NOT factor the two angles (per-angle corr 0.6146 / 0.5934) → a genuine product-manifold factorization is required (gated G0b).
- Gate G0: GO.

## Appendix C — What stays exactly as the prior plan had it (so nothing is lost)
Nuisance-only baseline as the common null; matched-bits/MDL (never matched feature count); ≥5 seeds + Tournament-TopK; random-init control on every figure; sequence-only input for Track A; leakage-aware time+scaffold splits + explicit leakage probe for Track B; validity filter (pLDDT/pTM/clash/ProteinMPNN-refold) on every steered output; curved/block dictionary as a control, not a headline; the kill-gate ordering (probes before any steer/decode job); readout-independence for every causal claim; cluster-respecting resampling for all inference; two-tier multiple-comparison correction; the honest-negative-as-deliverable stance. These are good and are retained verbatim in intent.

## Appendix D — One-line prior-art out-positioning ledger (for the related-work section)
- **SASA/Dal26 (2606.06333):** owns capacity theorem + O(1)-block remedy (Euclidean). We claim identity up-to-isometry + causal + epistemics; concede capacity, do not compete. Phase 0 confirms the tie. `[CONCEDE capacity]`
- **Steenwyk/Ste26:** scalar SAE steers ESM3 L33 (r=0.87, 4% token flips). We add wrap-around (a scalar cannot close), matched-norm geometry lift, the layer-resolved efficacy map, and a validity filter it lacks.
- **Skopek/Sko19:** learnable signed curvature is ill-conditioned. We use fixed-geometry blocks; cite as why we avoid learnable curvature.
- **PairSAE/Mig26 (2606.27440):** predicted Boltz-2's OWN affinity (R²=0.528, circular). We use experimental Kd; ipSAE strictly a contrast.
- **Jedryszek/Jed26b (2608.11475):** decodable ≠ steerable on Boltz-1 (β-strand F1 0.82, steer −0.001). We extend to ESM3 + wrap-around + layer/step persistence map.
- **Zha26g (2608.18597) + Hin26:** off-manifold collapse / masked-pLM edits don't survive decode (foldability breaks >~15% coverage). We adopt their filters and extend to an ESM3 layer map; they forecast our G2.
- **Setlur/Set26 (2608.26419):** geometry-ANNOTATED ESM-2 SAE features (closest protein prior art). We add a gauge-free coordinate, causality, and the confidence axis.
- **Xun26 (readable before causal, LLM):** our efficacy ratio is this made quantitative on ESM3.
- **Zho26h:** logit/sampling control of ESM3 — the non-interp baseline to beat on realized change.
- **Engels/Eng24:** circular features / modular arithmetic in LLMs with hand-set targets, no physical period. We use exact physical ground truth + 2π closure.
- **LLM truth/confidence wave (2603.25052; 2602.08159; 2602.20273; ACL-2026 — ALL `[UNVERIFIED]`):** same hypothesis, no external truth, no deployed confidence head, no causal/identifiable/cross-track dimensions. Must-cite framing antecedents AND the live scoop; foreground the four differentiators.
- **Codon-frame NeurIPS-2026 papers (×2):** own the DNA period-3/Z₃ story. `[CONCEDE the codon leg]`; we differentiate on structure-derived phase slips RoPE cannot fake.

---

_End of core document. Research/planning only; no repository changes made in the two source passes or in this merge. Supersedes `MASTER_PLAN.md` and `PROJECT_DOSSIER.md` as the core document; those are retained in `docs/research-plan/` as the provenance record, alongside `FINAL_FINDINGS.md`, `PITCH_VETTING_consolidated.md`, `PLAN_ICML_synthesis.md`, `PROJECT_BRIEF.md`, `primer_bsf_jacobian_landscape.md`, `technical_plan.md`, `evidence_base.md`, and `appendix/`. Merged 2026-11-22 from two independent expert passes (architect backbone + adversarial-reviewer hardening). All `[UNVERIFIED]` / `UNVERIFIED` and `[CONCEDE]` tags preserved; no citations invented; the dual-use statement and the primary-source checks still owe a human / compliance pass (§15)._
