# Scientific narrative & novelty plan — CLAIMS → EXPERIMENTS → FIGURES → PAPER
Scientific Narrative / Novelty Lead pass, 2026-11-22. Research/planning only; no repo changes.
Authoritative inputs: PROJECT_BRIEF.md, PITCH_VETTING_consolidated.md, FINAL_FINDINGS.md, PLAN_ICML_synthesis.md,
primer_bsf_jacobian_landscape.md, plan_task3_steering.md, plan_icml_novelty.md.
Benign/defensive framing throughout; Track B (binding) handled as auditing/calibration on benign complexes only.
"UNVERIFIED" = not confirmed to a primary source this pass.

================================================================================
## 1. THE ONE-SENTENCE CONTRIBUTION + WHY IT CLEARS A TOP-VENUE BAR
================================================================================

### One sentence
> On quantities whose true value is fixed externally by physics — backbone (phi,psi) from DSSP in a protein
> language model, and lab-measured binding strength Kd in a co-folding model — we show that a biological
> foundation model represents the quantity as an **identifiable, gauge-free coordinate** that (i) equals the
> true physical value **up to isometry** on held-out data, (ii) **causally** drives the model's own output with
> a **wrap-around dose-response** that a single-direction latent structurally cannot produce, and (iii) lives in
> a direction **provably distinct from the model's own calibrated self-confidence** (pLDDT for the angle, ipSAE
> for the interface) — so the model keeps *what is true* in a different place from *how sure it is*, and that
> separation is a **convergent property across both a periodic structural variable and a scalar thermodynamic
> one, in two model families**.

### Why this clears the bar (4 bullets)
1. **A causal + identifiability + epistemics result on EXACT external ground truth — a conjunction nobody holds.**
   Prior bio interpretability either reconstructs a subspace (SASA), annotates features (Setlur), or steers a
   scalar direction (Steenwyk) — none shows the recovered coordinate *is* the physical one up to isometry AND
   that the model causally uses it AND that truth is stored apart from confidence. The three together, on a
   quantity we can compute independently of the model, is the contribution.
2. **It imports a live, contested ML question into the one domain that can answer it cleanly.** In 2026 the LLM
   community is actively arguing that models keep an internal correctness axis separate from expressed confidence
   (2603.25052; "Confidence Manifold" 2602.08159; "Truthfulness Spectrum" 2602.20273; ACL-2026 "LLM Beliefs Are
   in Their Heads") — but they must *define truth as the model's own answer-correctness*, with no external
   referent, and their "confidence" is probed/elicited. Biology supplies what that program lacks: truth pinned by
   physics and a **deployed, calibrated confidence head** (pLDDT/ipSAE). We turn a contested phenomenon into a
   grounded, causal one. Biology is the *instrument*, not the application — the inversion top venues reward.
3. **Cross-track invariance makes it a property of the model class, not an artifact.** The same epistemic triad
   holds for a 2-torus (periodic, structural, ESM3) and a scalar (thermodynamic, Boltz) — ruling out
   "you found one cherry-picked feature in one model".
4. **Honest negatives are a deliverable, not a risk.** The persistence/commitment gate and the RoPE-proof
   control convert the most likely failure points into a calibration/auditing finding; paired with the trust-axis
   monitor (section 8), a null still yields a defensive-interpretability paper rather than a dead end.

================================================================================
## 2. THE CLAIMS (falsifiable), WITH EXPERIMENT / FIGURE / CONTROL / PRIOR TO OUT-POSITION
================================================================================
Six claims. C1–C3 are the spine (Track A), C3 is the hook, C4–C5 are the controls-elevated-to-claims, C6 is the
cross-track payoff. Each is stated so a single clean experiment can falsify it.

--------------------------------------------------------------------------------
### CLAIM 1 — IDENTITY / RECOVERY UP TO ISOMETRY
**Statement.** The coordinate recovered from the model's activations (sequence-only input) equals the true
physical coordinate up to the quantity's isometry group — SO(2)/O(2) reparametrisation for the helix-phase
circle, the torus isometry group for (phi,psi), a monotone reparam for Kd — measured on a held-out split with
the alignment **fit on train only**. Not merely "a subspace reconstructs the activation".
- **Experiment.** Fit a block/geometry-matched dictionary (a *tool*, not the contribution) on ESM3-open
  activations at one layer; align the recovered 2-D block coordinate to ground-truth (phi,psi) with an isometry
  estimated on a train split; report geodesic/circular error on held-out residues. Track B: recovered 1-D
  binding-strength coordinate aligned monotonically to experimental Kd on a leakage-aware split.
- **Figure/Table.** Fig 1: recovered torus overlaid on the ground-truth Ramachandran plot (held-out), coloured
  by geodesic error. Table 1: held-out alignment error for {geometry-matched block, Euclidean block, scalar SAE,
  random-init model} **at matched bits/MDL** (not matched feature count).
- **Control (bulletproofing).** (a) Random-init same-architecture ESM3 — the torus must NOT appear; (b)
  matched-bits scalar SAE — must fragment into arcs; (c) isometry fit on train, scored on test (no post-hoc gauge
  cheating); (d) sequence-only input (structure input makes dihedrals trivial); (e) local-sequence-window
  positional control (run first) so recovery is not index-driven.
- **Out-positions.** SASA/Dal26 (proves a subspace *reconstructs*; never that the coordinate *is* the physical
  one up to isometry); Setlur/Set26 (geometry-*annotated* ESM-2 features, no gauge-free coordinate, no isometry
  guarantee); BSF/Jer26 (Euclidean blocks; within-block rotation is a *caveat* — we make it the guarantee);
  InterPLM (single-direction features).

--------------------------------------------------------------------------------
### CLAIM 2 — WRAP-AROUND CAUSAL DOSE-RESPONSE
**Statement.** Rotating the recovered circular coordinate by angle theta moves the model's own output in the
matching direction and **returns to baseline at theta=2*pi** — a closed causal loop a 1-D/scalar latent cannot
produce. Track B: steering the binding-strength coordinate gives a monotone dose-response whose *sign* agrees
with SKEMPI delta-delta-G on point mutations.
- **Experiment.** Inject a rotation of the block coordinate across theta in [0,2*pi]; read the model's OWN output
  head (predicted structure token / residue dihedral), plot realised output angle vs theta. Track B: sweep the
  coordinate, read predicted interface geometry; cross-check mutation sign against SKEMPI.
- **Figure/Table.** Fig 2: output dihedral vs steering angle tracing a closed loop returning to baseline at
  2*pi; matched-norm random direction flat. Table 2: wrap-around fidelity (loop correlation, return-to-baseline
  residual) for {recovered coordinate, scalar-direction steer (Steenwyk-style), matched-random}.
- **Control.** Matched-norm random direction at the same site; validity filter (pLDDT/pTM/clash/ProteinMPNN
  refold) so the steered output is on-manifold, not garbage; readout is the model's own head, not a same-variable
  probe (readout independence).
- **Out-positions.** Steenwyk/Ste26 (scalar SAE steers ESM3, r=0.87 — but a scalar direction has no 2*pi return;
  the wrap-around is the test it structurally cannot pass); Jed26b (decodable != steerable — we show steerable
  with a stronger-than-linear signature); PairSAE/Mig26 (steered the model's own affinity — we steer toward
  external physics).

--------------------------------------------------------------------------------
### CLAIM 3 — TRUTH ⟂ CONFIDENCE  (THE HOOK)
**Statement.** The physical quantity is linearly separable from the model's own confidence about it, and the two
are **causally dissociable**: steering the truth coordinate moves the physical output but not the confidence head,
and steering the confidence direction moves pLDDT/ipSAE but not the physical output (a double dissociation). On
**confidence-matched inputs** (same pLDDT/ipSAE, different true physics) the truth coordinate still separates
them, which confidence by construction cannot.
- **Experiment.** Estimate the truth subspace (C1 coordinate) and the confidence subspace (direction most
  predictive of held-out pLDDT for A / ipSAE for B). Measure (i) principal angle; (ii) the 2x2 double
  dissociation (steer truth|confidence x measure physics|confidence); (iii) behaviour on "confidently-wrong"
  cases (high ipSAE, weak Kd; or high pLDDT, atypical dihedral) — does the truth axis still track physics there.
- **Figure/Table.** Fig 3: the 2x2 dissociation grid + a "confidently-wrong" scatter where the confidence head is
  high but truth coordinate and ground truth say otherwise. Table 3: principal angles, dissociation effect sizes,
  truth-axis accuracy on the confidence-matched pairs.
- **Control.** Both directions matched-norm; the confidence direction must be *real* (predicts held-out
  pLDDT/ipSAE above chance); the physics readout independent of the confidence head; **confidence-matched
  negative-pair control** (section 8, Sharpening 3) to kill the "same entangled signal" rebuttal.
- **Out-positions.** Dunbrack ipSAE / ipTM (define interface confidence; never ask whether it is a separate
  representational axis from truth); the LLM truth/confidence line (2603.25052, 2602.08159, 2602.20273, ACL-2026)
  — same hypothesis but **no external ground truth, no deployed calibrated confidence head, no causal wrap-around,
  no cross-track test**; PairSAE (conflated truth and the model's own confidence). This claim carries the paper.

--------------------------------------------------------------------------------
### CLAIM 4 — RoPE-PROOF PHASE-SLIP
**Statement.** The recovered phase feature tracks **structure-derived** irregularities — helix kinks, 3-10 helix
segments, phase slips — that positional encoding (RoPE) alone cannot manufacture; it departs from index-mod-3.6
exactly at the structural slips.
- **Experiment.** Within DSSP helices, compare feature phase to (a) the naive index-mod-3.6 baseline and (b) the
  true structural phase including kinks/3-10 transitions; show the feature follows the structural slip where the
  two diverge.
- **Figure/Table.** Fig 4: feature phase vs residue index with structural phase-slip events marked; feature
  tracks the slip, index baseline does not. Table 4: residual at slip residues (feature vs index baseline).
- **Control.** Positional/local-window control run FIRST; sequence-only input; random-init.
- **Out-positions.** The two NeurIPS-2026 codon-frame papers (own index/period-3 periodicity — this is the
  baseline we must beat, so the signal must be *beyond* index); the general RoPE-artifact concern; Steenwyk
  geometry atlas.

--------------------------------------------------------------------------------
### CLAIM 5 — PERSISTENCE / COMMITMENT (honest, possibly-negative)
**Statement.** A mid-computation edit to the recovered coordinate either survives iterative decoding to the final
output (positive causal result) or is washed out (calibration negative), and where it does is **layer-resolved**:
the most-decodable, most-steerable, and most-persistent sites differ.
- **Experiment.** Inject at layer l, let ESM3 iterative unmasking proceed, measure realised change in the final
  output vs a late/output injection; build a layer (x denoising-step, Track B) decodable-vs-steerable-vs-persistent
  map. Track B: inject inside a Boltz diffusion step, measure final-structure persistence.
- **Figure/Table.** Fig 5: layer (x step) heatmap of decodability, steerability, persistence with three different
  argmaxes. Table 5: persistence z-score vs matched-random per layer.
- **Control.** Matched-random; output-only baseline; validity filter.
- **Out-positions.** Zha26g off-manifold collapse + Hin26 masked-pLM edit persistence + Jed26b (one layer) — we
  extend to a layer map on ESM3 structure tokens and **tie persistence to the truth/confidence axes** (does the
  model preserve truth-edits and confidence-edits differently?).

--------------------------------------------------------------------------------
### CLAIM 6 — CROSS-TRACK INVARIANCE (the payoff; my sharpening, section 8.2)
**Statement.** The epistemic triad (identity-up-to-isometry, causal dose-response, truth⟂confidence) holds across
a **periodic structural** variable (phi,psi / helix phase, ESM3) and a **scalar thermodynamic** variable (Kd,
Boltz), and across model families — so truth/confidence separation is a convergent property of biological FMs,
not a quirk of one variable or one model.
- **Experiment.** Run C1–C3 on both tracks; add a small ESM-2-vs-ESM3 generality panel (is the torus shared,
  more causal in the multimodal model?).
- **Figure/Table.** Fig 6: side-by-side triad summary (A vs B) + the ESM-2/ESM3 generality panel. Table 6: each
  claim's effect size and control margins per track.
- **Control.** Each geometry wins on its matched variable and **loses on a mismatched one** (double dissociation:
  circular code loses on non-periodic relative SASA) — guards against "any rich code recovers anything".
- **Out-positions.** Steenwyk cross-model (feature-sharing, no causal/epistemic axis); Whitfield (concept
  emergence by depth, no causal, no confidence); establishes generality nobody has claimed for this triad.

================================================================================
## 3. BUILD ORDER — MINIMUM PUBLISHABLE UNIT → FULL PAPER
================================================================================
**Spine recommendation: TRACK A is the spine; A = core, B = extension. One paper.**
Rationale: Track A has exact deterministic ground truth (DSSP), the cheapest decisive tests, the lowest dual-use
risk, and no leakage problem; it can carry C1–C5 alone. Track B generalises the hook to a scalar thermodynamic
variable with real lab ground truth and supplies the single most striking "confidently-wrong" instance (high
ipSAE, weak Kd), but it is dual-use-sensitive, leakage-prone, and compute-heavier. If B slips on ethics/leakage/
compute, the paper stands on A. So B is a **detachable extension that upgrades the paper from "one model, one
variable" to the cross-track invariance (C6)** — it is where the generality and the most quotable example live,
but it is never load-bearing for acceptance.

**Staged build (each stage is a go/no-go gate; cheapest-decisive first):**
- **MPU-0 (synthetic, ~1-2 CPU-h, no weights).** Plant a torus AND an orthogonal "confidence" channel in R^64;
  verify the block recovers the coordinate up-to-isometry (C1 metric) AND that the truth⟂confidence metric
  separates the planted truth from the planted confidence. *De-risks the metric before any model.* (My added
  control — section 8.3.) GO if both metrics behave as designed.
- **MPU-1 (ESM3, ~tens GPU-h).** C1 identity on held-out, sequence-only, one layer + the **G0 persistence gate**
  (do edits survive iterative decode?) + local-window/random-init controls. GO if a block decodes the torus
  up-to-isometry where scalar SAE fragments AND edits persist (if persistence fails, pivot C5 to the honest
  negative and keep the paper). *This stage alone is a workshop-grade stake.*
- **MPU-2.** C2 wrap-around dose-response on the ESM3 output head + matched-random + validity filter. Crown jewel
  for Track A.
- **MPU-3.** C3 truth⟂confidence double dissociation ((phi,psi) vs pLDDT) + confidence-matched-pair control.
  **This is the hook; once MPU-1→3 close, the core paper exists.**
- **Unit-4.** C4 RoPE-proof phase-slip panel.
- **Unit-5 (extension = Track B).** Boltz, Kd vs ipSAE: C1 (monotone to Kd), C2 (dose-response + SKEMPI sign),
  C3 (confidently-wrong instance), C5 (persistence in diffusion time). Leakage-aware splits mandatory.
- **Unit-6.** C6 cross-track invariance synthesis + ESM-2/ESM3 generality panel.
- **Full paper.** C1–C6 + honest negatives + the trust-axis auditing primitive (section 8.1) as the discussion
  payoff and dual-use-positive artifact.

**Decision gate.** If MPU-1 (identity) or MPU-3 (truth⟂confidence) does not close cleanly, the paper reframes to a
calibration/auditing-benchmark contribution (still publishable) rather than the causal-epistemics headline.

================================================================================
## 4. THE "TRUTH vs CONFIDENCE" THROUGHLINE — developed as the paper's intellectual hook
================================================================================
**The question.** Does a biological foundation model maintain an internal representation of *what is physically
true* that is distinct from its internal representation of *how confident it is*? Concretely: is the (phi,psi)
the model "believes" stored in a different direction from pLDDT, and is Kd stored in a different direction from
ipSAE — such that you can move one without the other, and such that on the cases where the model is **confident
and wrong** the truth axis still knows the answer?

**Why this is the right spine, not a garnish.** It subsumes the other claims into a single intellectual arc:
identity (C1) establishes that there *is* a truth coordinate to talk about; causality (C2) establishes the model
*uses* it; truth⟂confidence (C3) establishes that the model's epistemics are *structured* — truth and self-
assessment are different objects. The paper stops being "we found a torus in a protein model" (re-finding physics,
a reviewer's yawn) and becomes "we show a biological model has a two-part epistemic structure — a world-model
coordinate and a self-confidence coordinate — that are geometrically and causally separable."

**Why it is interesting beyond biology.** This is the mechanistic core of calibration and model honesty, and in
2026 it is a live, unresolved question in LLM interpretability — but one the LLM community cannot settle cleanly,
for two structural reasons:
1. **No external ground truth for "truth".** LLM truth-direction work (Azaria-Mitchell; Burns CCS; Marks-Tegmark;
   and the 2026 wave, 2603.25052 / Confidence Manifold / Truthfulness Spectrum / ACL-2026 "LLM Beliefs") must
   operationalise "truth" as answer-correctness *within the model's own domain*, so a probed "truth direction" may
   be learning the model's belief, not the world. The Truthfulness-Spectrum result that "truth is not one
   direction" is a symptom of this. Biology hands us truth from **outside the model** — a dihedral from DSSP, a Kd
   from a calorimeter — so "the truth axis" is anchored to the world, not to the model's self-report.
2. **No deployed, calibrated confidence term.** LLM "confidence" is elicited (verbalized) or probed. Biological
   FMs *ship* a calibrated confidence head (pLDDT, ipSAE) that is used in practice to decide whether to trust a
   prediction. That makes "confidence" a real, externally-meaningful quantity, not a construct.
So biology is the clean bench for a general ML question: it is the one place where **both** terms — truth and
confidence — are pinned independently of each other and of the model's narration, and where we can add the two
dimensions the LLM work lacks: a **causal** double dissociation and an **identifiable** (up-to-isometry) truth
coordinate. The paper's reach is therefore: *a validated, causal, physically-grounded instance of the "models
separate truth from confidence" hypothesis, which the LLM program is currently asserting without external grounding
— and a construction (the trust axis, section 8.1) that exports back as an auditing primitive.*

**The money example.** A complex where ipSAE is high (the model is sure the interface is good) but the true Kd is
weak (it binds poorly). If the recovered truth coordinate still reports "weak", the model *internally knew* and the
confidence head overrode it — a mechanistic anatomy of a calibration failure, and a direct argument for monitoring
the truth axis rather than the (gameable) confidence head.

================================================================================
## 5. NOVELTY AUDIT (candid, post-vetting)
================================================================================
### Genuinely new
- **C3 truth⟂confidence on exact physical ground truth, using the model's OWN calibrated confidence head, with a
  causal double dissociation.** Unclaimed in biology (web-confirmed: closest is CONFIDE / uncertainty-of-
  representations, neither of which poses truth and confidence as separable *directions*). This is the headline.
- **C1 recovery-up-to-isometry as a gauge-free IDENTITY claim** (not reconstruction) on a periodic physical
  variable, tied to C2's causal test. SASA/BSF/Setlur do not make the identity-up-to-isometry claim.
- **C6 cross-track invariance** of the epistemic triad across variable type and model family — my sharpening;
  no prior work tests the same epistemic structure on both a periodic structural and a scalar thermodynamic
  variable.
- **The trust-axis auditing primitive** (section 8.1) — converts the finding into a defensive capability; new.

### Incremental / table-stakes (cite and own honestly, do not oversell)
- **C2 wrap-around dose-response** — a genuinely stronger steering test (periodic return a scalar cannot fake),
  but adjacent to Steenwyk (steers ESM3 already) and Jed26b; a hostile reviewer calls it "a nicer control".
- **C5 persistence layer map** — extends Zha26g/Hin26/Jed26b; valuable as calibration, not a headline.
- **C4 RoPE-proof phase-slip** — a control elevated to a claim; necessary, not novel on its own.
- **The block/geometry-matched dictionary machinery** — explicitly NOT the contribution (vetting killed the
  "curved beats flat" headline: SASA owns capacity+remedy, Skopek owns ill-conditioned learnable curvature, a
  flat block of equal dim ties at matched bits). We use blocks as a *tool* and compare at matched bits as a
  *control*, never as the pitch.

### The strongest reviewer rebuttal I cannot yet fully answer
> "Your truth axis and your confidence axis are two readouts of the same underlying 'this region is well-
> structured' signal. pLDDT is high exactly where backbone geometry is regular (helices/sheets), so (phi,psi) and
> pLDDT are entangled in the data distribution; a principal angle > 0 and a 'double dissociation' can both arise
> because you fit two different linear heads on one correlated latent, and your steering simply isn't selective
> enough to prove otherwise. Show the separation is not an artifact of the training-distribution correlation —
> and note the 2026 LLM papers (2603.25052, Confidence Manifold) already report near-orthogonal truth/confidence
> directions, so even the qualitative claim is not yours; what's left is 'we did it in proteins'."

This is the crux risk and it has two prongs. (a) *The entanglement prong* is only partly answered by the double
dissociation; the real answer is the **confidence-matched negative-pair control** (section 8.3) and the
**confidently-wrong slice** — but for Track A we may not have enough high-pLDDT/atypical-dihedral cases, and for
Track B leakage muddies the slice. I cannot yet guarantee the effect survives on those slices; that is the
experiment that must close before the hook is safe. (b) *The scoop prong* is real: truth⟂confidence is a fresh
2026 LLM topic, so the bio instantiation must foreground what LLMs cannot do (external ground truth + calibrated
deployed confidence head + causal wrap-around + cross-track invariance + the auditing primitive), or a reviewer
reads it as derivative. Both prongs are survivable but neither is closed on paper today — flag for the human.

================================================================================
## 6. PRIOR-ART OUT-POSITIONING LEDGER (one line each)
================================================================================
- **SASA (Dal26, 2606.06333)** — reconstruction subspace + capacity theorem; we claim *identity up-to-isometry* +
  causal + epistemics. Beaten by C1/C3, not on its own turf (we do not pitch capacity).
- **Steenwyk (Ste26)** — scalar SAE steers ESM3 (r=0.87); beaten by C2 (wrap-around a scalar cannot do) + C3.
- **Skopek (Sko19)** — learnable signed curvature is ill-conditioned; we use *fixed*-geometry blocks as a tool and
  do not stake the headline on curvature. Cite as why we avoid learnable curvature.
- **PairSAE (Mig26, 2606.27440)** — predicted Boltz-2's OWN affinity (circular); Track B uses **experimental Kd**,
  ipSAE strictly as the confidence CONTRAST, never the target.
- **Jedryszek (Jed26b, 2608.11475)** — decodable != steerable on Boltz-1; we show steerable with a stronger
  signature (C2) + a layer/step persistence map (C5), extended to ESM3.
- **Zha26g (2608.18597) + Hin26** — off-manifold collapse / masked-pLM edits don't survive decode; extended by C5
  into a layer-resolved map tied to the truth/confidence axes.
- **Codon-frame papers (NeurIPS 2026)** — own index/period-3; beaten by C4 (structure-derived phase slips beyond
  index).
- **Setlur (Set26, 2608.26419)** — geometry-annotated ESM-2 SAE features (closest protein prior art); no gauge-free
  coordinate, no causal, no confidence axis. Must cite + differentiate on C1/C2/C3.
- **LLM truth/confidence wave (2603.25052, "Confidence Manifold" 2602.08159, "Truthfulness Spectrum" 2602.20273,
  ACL-2026 "LLM Beliefs Are in Their Heads", EMNLP-2025 Verbal-Uncertainty) — UNVERIFIED beyond abstracts** — same
  hypothesis, no external ground truth, no deployed confidence head, no causal/identifiable/cross-track dimensions.
  The must-cite framing antecedents AND the live scoop; foreground our differentiators. Verify primary texts.

================================================================================
## 7. DUAL-USE / ETHICS PARAGRAPH (draft for the paper; owed a human/compliant pass)
================================================================================
> This work is framed as auditing and calibration of generative biological models, not as a design tool. Track A
> (backbone dihedral angles, helix phase, relative SASA) concerns low-level structural geometry and carries
> negligible misuse potential. Track B (interface binding) is capability-adjacent — knowing where a model
> represents binding strength is one step from optimising it — so we constrain it to a defensive posture: the
> steering target is experimental Kd and model-independent geometric interface descriptors, with the model's own
> confidence (ipSAE/ipTM) used only as a *contrast*, never as an optimisation objective (steering toward a model's
> own confidence is both circular and a Goodhart hazard); we evaluate only benign, open complexes and exclude
> hazard-adjacent classes (toxins, viral entry/fusion, immune-evasion, select-agent organisms); and we release no
> affinity-maximising steering pipeline and no per-step binder-optimisation recipe. We argue the primary value of
> locating *where a property is causally and confidently encoded* is defensive: it yields a monitor (the trust
> axis, below) that reads a generative design model's internal truth coordinate independently of its gameable
> confidence head, so a deployer can flag "confidently wrong" outputs. This section requires a human/compliance
> review before submission (a prior automated attempt to draft it was repeatedly truncated).

================================================================================
## 8. MY ORIGINAL SHARPENINGS (contributed this pass), with justification
================================================================================
### 8.1 (LEAD) The TRUST AXIS — truth⟂confidence reframed as a concrete auditing primitive
**What.** Define a per-input **trust residual** = the component of the model's confidence head (pLDDT/ipSAE) that
is NOT predictable from the recovered truth coordinate. Claim and test: the trust residual flags the model's
calibration failures (confidently-wrong cases) *better than the confidence head alone*, because it reads the truth
axis the confidence head ignores. Deliverable: a monitor that, for a generative design model, reports "the
internal truth coordinate disagrees with the emitted confidence here".
**Why it's worth adding.** (1) It converts an interpretability curiosity (two directions are separate) into the
*reason the separation matters* — an actionable, defensive capability, which is also the dual-use-positive use
(section 7). (2) It is biology-independent: the same construction audits LLM calibration, so it is the bridge that
gives the paper reach beyond proteins. (3) It differentiates us from the LLM confidence-steering papers
(2603.25052 reads the internal estimate and *steers the output to match it* for better calibration): we **monitor**
a deployed design model against **external physical truth**, we do not retrofit the model's self-report.
**Risk.** It leans on C3 surviving the entanglement rebuttal (section 5); if C3 is weak, the trust axis is weak.

### 8.2 The CROSS-TRACK INVARIANCE claim (C6)
**What.** Elevate "A and B both work" into a falsifiable invariance: the epistemic triad is a convergent property
across a periodic structural variable and a scalar thermodynamic one, in two model families.
**Why.** It is the honest answer to "did you cherry-pick one feature in one model?" and it is the structural reason
to keep Track B in the paper at all (B earns its place by generalising the hook, not by being a second result).

### 8.3 A new CONTROL — the confidence-matched negative-pair test
**What.** For C3, construct input pairs matched on the model's confidence (same pLDDT/ipSAE bin) but differing in
true physics (different (phi,psi)/Kd); show the truth coordinate separates them while the confidence term, by
construction, cannot.
**Why.** It is a cleaner refutation of the "same entangled signal" rebuttal (section 5) than a principal angle:
it holds confidence fixed and varies truth, so any separation it finds is truth-specific. Also the right place to
spend the confidently-wrong instances. This is the control I would run FIRST inside MPU-3, before trusting the
double dissociation.

================================================================================
## 9. STITCHED STORY (how the paper reads, one paragraph)
================================================================================
A biological foundation model is handed a quantity whose true value we know from physics. First we show the model
holds that quantity as an identifiable, gauge-free coordinate — the actual angle on the torus, recovered up to
isometry on held-out data, not merely a subspace that reconstructs the activation (C1). Then we show the model
*uses* it: rotating the coordinate drives the model's own output around a full circle and back to baseline at
2*pi, a causal signature a single direction cannot fake (C2). Then the turn: that truth coordinate sits in a
different place from the model's own confidence about it — steer one and the other does not move, and on the cases
where the model is confident and wrong, the truth axis still knows (C3). We validate the phase feature against
structure-derived slips that positional encoding cannot invent (C4), map where in the computation edits survive
(C5), and show the whole triad recurs for a scalar thermodynamic variable in a second model family (C6) — so the
separation of what-is-true from how-sure-i-am is a property of the model class. Finally we turn the separation into
a monitor (the trust axis, 8.1) that reads the internal truth coordinate against the gameable confidence head —
the defensive payoff. Biology is not the application here; it is the one bench where both truth and confidence are
pinned from outside the model, letting us answer, causally and with ground truth, a question the LLM field is
currently asking without either.
