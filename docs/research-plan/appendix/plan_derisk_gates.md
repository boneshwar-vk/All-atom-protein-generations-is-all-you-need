# DE-RISKING PLAN — kill-gates, cheapest-decisive-first
Adversarial de-risking lead (skeptical senior reviewer). Date 2026-11-22. Research/planning only — NO repo changes.
Benign/defensive framing throughout. Source of truth: PROJECT_BRIEF.md + PITCH_VETTING_consolidated.md +
vet_pitch1..4 + FINAL_FINDINGS.md (all in this scratchpad).

## 0. STANCE AND THE ONE RULE
I assume this project dies. My job is to make it die **cheap** — sequence the tests so the most-likely-fatal,
cheapest-to-run check comes first, and every expensive step is gated behind a decisive cheap one. I treat
"curved geometry beats flat" as already dead (SASA/Dal26 owns capacity+remedy; a flat block of equal dim pays
O(1)); the only surviving contribution is **CAUSAL + IDENTIFIABILITY + TRUTH-vs-CONFIDENCE on exact physical
ground truth**. The whole plan is built to protect that kernel, and to convert each death into a publishable
calibration/benchmark result rather than a dead end.

**Critical-path ordering (by cost of the decisive test, cheapest first):**
G0 synthetic matched-bits + identifiability (CPU-hours, no model) →
[parallel: G-B1 Track-B leakage probe, cheap, no steering] →
G1 RoPE/positional control on ESM3 (probes only, no steering) →
G2 persistence/commitment on ESM3 (1 cell, steer+full-decode+refold) →
G-B2 ipSAE-collinearity (truth≠confidence, direction-angle only) →
G3 decodable≠steerable + geometry-lift map (ESM3, expensive) →
G4 wrap-around causal dose-response (ESM3, expensive) →
G-B3 ipSAE Goodhart + SKEMPI ΔΔG causal check (Track B, expensive).
Track A and Track B de-risk in parallel; neither blocks the other until the write-up.

A single **nuisance-only baseline** (defined in §Evolve) is threaded through every gate: the target must beat
"what you get from position-index ∪ training-set identity ∪ model self-confidence alone." Every GO threshold
below is *relative to that baseline*, not to zero.

---

## GATE G0 — "Curved earns its complexity, AND the angle is identifiable up to isometry" (SYNTHETIC)
**Cost:** ~1–2 CPU-hours. No model weights. FIRST, unconditionally. Kills death (i) + stress-tests the
identifiability pillar on the easiest possible case.

**Precise claims under test**
- G0a (headline-at-risk): a curvature-matched block strictly beats a **Euclidean block of equal dimension at
  matched bits/MDL** on reconstruction-to-ε, for ≥1 planted manifold.
- G0b (the kernel): the recovered angular coordinate θ̂ IS the planted θ **up to the isometry group** (rotation/
  reflection), i.e. a monotone, gauge-free alignment on a held-out split — not merely "a subspace exists."

**Cheapest falsifying experiment**
Plant circle→R², torus T²→R⁴, sphere S²→R³, swiss-roll→R³, (stretch) tree→low-dim, in R⁶⁴ via random
orthonormal frames + small Gaussian noise; 50–200k points; sweep ε. Fit three families on cached points:
(a) scalar TopK-SAE k=1 and k=2; (b) **Euclidean block** of ambient-subspace dim; (c) curvature-matched block.
Record atoms/bits to reach FVU≤ε, AND the matched-bits rate–distortion frontier, AND — for G0b — circular–
circular correlation of θ̂ to planted θ on held-out points after Procrustes/rotation alignment, with Betti
(b₀,b₁)=(1,1) for the circle.

**Null / baselines / controls**
- Matched **BITS/MDL**, never matched atom count (a flat block of equal dim already captures low-dim manifolds).
- Euclidean-block-of-equal-dim is the load-bearing control (isolates curvature from capacity).
- ≥5 seeds + a stability statistic (seed variance of θ̂ alignment and of any learned κ); Tournament Top-K.
- Regularizer-insensitivity sweep (if a learnable-κ block is tried at all): κ answer must not be a function of
  the κ-regularizer weight.

**GO / NO-GO**
- G0a GO only if curved strictly dominates the Euclidean block at matched bits, ≥5-seed separation, ≥1 manifold.
  **Predicted NO-GO (tie).**
- G0b GO if held-out circular-corr(θ̂,θ) ≥ 0.9 and Betti correct in ≥9/10 seeds for circle/torus.

**On NO-GO**
- G0a NO-GO (expected): **do NOT build the learnable-curved-dictionary machinery** (saves weeks; avoids the
  Sko19 ill-conditioning trap). Use a **fixed/assigned 2-D block** to carry the angle; demote "geometry
  discovery" entirely. This is a framing pivot, not a project kill.
- G0b NO-GO: **KILL the identifiability pillar at its cheapest test.** If you cannot recover θ up to isometry on
  clean synthetic data, you never will on ESM3. Fall back to the Minimum Publishable Result (benchmark + the
  honest statement that recovery is not identifiable even synthetically — itself a useful negative).

**Scoop & differentiation**
SASA (Dal26, 2606.06333) owns the covering-number theorem (Thm 4), the empirical atoms-vs-covering curve
(Fig 2c, circle/sphere/helix), and the O(1)-block remedy (Prop 10, Thm 11) — in **Euclidean** form. Skopek
(Sko19) owns learnable signed curvature and *documents its ill-conditioning* (κ drifted to ~0 on trees, no sign
recovery). **Differentiate by NOT competing on capacity/geometry at all**: G0 is run only to *confirm the kill
cheaply* and to validate identifiability (which SASA does not claim — it recovers the linear subspace, not the
intrinsic coordinate up to isometry). The contribution is recovery-of-the-coordinate + causality, not a block.

---

## GATE G-B1 — "Track-B Kd is learning, not leakage" (TRACK B, runs in PARALLEL with G0/G1)
**Cost:** cheap GPU; probes only, no steering, no decode. Decisive for the *entire* Track B. Kills death (iv).

**Precise claim** The recovered Boltz interface feature predicts **experimental Kd** out-of-distribution under
leakage-aware splits, i.e. it tracks physical binding strength, not memorized complexes or the model's own
predicted affinity.

**Cheapest falsifying experiment**
Cache Boltz pair/interface reps on PDBBind/PLINDER complexes. Fit a probe rep→Kd. Evaluate under **three splits**:
(1) random (optimistic), (2) **time split** (train pre-cutoff, test post-cutoff), (3) **scaffold/sequence-cluster
split** (PLINDER leakage-aware). Report held-out Spearman/RMSE per split.

**Null / baselines / controls (the leakage probe is mandatory)**
- **Explicit leakage probe:** a baseline that sees ONLY nuisance identifiers — training-set-membership flag,
  ligand/sequence cluster ID, ligand MW/heavy-atom count, protein family — and no learned interface geometry.
  This upper-bounds Kd predictable from leakage alone. The feature must beat it on the clustered/time split.
- **Target must be EXPERIMENTAL Kd, never Boltz-2's own predicted affinity** (that is the PairSAE/Mig26
  circularity, R²=0.528 on the model's own score).
- Random-init same-architecture Boltz control (feature must beat random-weights).
- ≥5 seeds; validity filter (pLDDT/ipTM/clash) so "predictive" features aren't reading garbage complexes.

**GO / NO-GO**
GO if held-out Spearman on the **clustered + time** split ≥ (leakage-probe + 0.15) AND > random-init, stable
across ≥5 seeds. NO-GO if performance collapses to the leakage probe on the clustered/time split (random-split
number is then leakage, not learning).

**On NO-GO** **KILL Track B's "predicts physical Kd" claim.** Pivot Track B to the *truth-vs-confidence
calibration* result only: report that Boltz's internal affinity feature is a memorization/confidence artifact,
not a physical-Kd representation (a publishable leakage-audit negative). Consider dropping Track B to an appendix
and making Track A load-bearing — consistent with FINAL_FINDINGS (Boltz lane crowded by PairSAE/Jed26b).

**Scoop & differentiation** PairSAE (Mig26, 2606.27440) predicted Boltz-2's **own** affinity (circular).
Jed26b (2608.11475) = decodable≠steerable on Boltz-1 trunk/diffusion. Differentiate by (a) experimental Kd
target, (b) leakage-aware splits + explicit leakage probe, (c) SKEMPI ΔΔG as a built-in causal check (G-B3).
PLINDER counts/licences are UNVERIFIED in the brief — verify before committing.

---

## GATE G1 — "The periodic feature is biology, not RoPE" (TRACK A, positional control FIRST)
**Cost:** activation caching + probes on ESM3-open; tens of GPU-h; **no steering, no decode** → cheaper than any
causal gate, so it runs before G2. Kills death (ii).

**Precise claim** The recovered helix-phase / (phi,psi) periodicity is a learned structural representation, not
an artifact of ESM3's rotary positional encoding (RoPE rotates 2-D channel pairs by an angle linear in index →
a false-positive machine for constant phase velocity).

**Cheapest falsifying experiment (sequence-only input, structure track OFF)**
Cache ~2–5k single-domain chains at ~3 mid/late layers (e.g. L18/L30/L40 of 48). Compute per-residue
**structure-derived** helical phase (screw-axis projection from experimental coordinates), DSSP labels. Discover
a 2-D cyclic subspace (size-2 BSF block or 2D probe to [cos,sin] of structure-phase), then test its phase
velocity.

**Null / baselines / controls (all mandatory, positional FIRST)**
1. **Helix vs coil at matched sequence indices** (primary): RoPE's phase advance is identical for helix and coil
   at the same index; only biology differs. A ~100°/res slope in helix but not coil at matched index = biology.
2. **Trained − untrained (random-init, SAME RoPE)**: the trained-minus-random gap at the helix frequency is the
   learned part (also the Heap–Aitchison table-stakes control).
3. **Local-sequence-window baseline** (index-mod-3.6): feature-phase must track **structure-derived phase slips**
   (kinks, 3₁₀ caps, bulges) via *partial* circular correlation after regressing out index-mod-period — the one
   signal RoPE/positional encoding structurally cannot fake.
4. Residue-shuffle and phase-randomized surrogates for the null distribution; matched-norm random 2-D direction.
5. ≥5 seeds; report seed variance.

**GO / NO-GO**
GO if (a) helix slope ≫ coil slope at matched index with CI excluding coil, (b) trained−untrained gap
significant at the helix frequency, (c) partial circular-corr(feature-phase, structure-phase | index-mod-3.6)
> 0 with helices showing matching local velocity deviations. NO-GO if the signal is equal in coil or in the
untrained model, or vanishes once index-mod-3.6 is regressed out.

**On NO-GO** **KILL the "RoPE-proof physics validator" pillar (pillar 3).** The periodicity is positional.
Pivot: keep the non-periodic SASA control variable as the honest headline (periodicity claim retracted), and
report the RoPE-artifact finding as a methodological caution (valuable: it tells the field its "rings" may be
rotary index). Do not proceed to wrap-around steering on a positional artifact.

**Scoop & differentiation** Two NeurIPS 2026 codon-frame papers own period-3/Z₃ + causal frameshift end-to-end
(genomics) — **do not compete on codon; cite as positive prior evidence**. Gup26b "Spokes, not a Ring" disputes
*between-state arc meaning*, a different question — do NOT advertise phase-velocity as "rebutting Spokes"
(category error a reviewer will catch). Set26 owns geometric *annotation* of ESM-2 SAE features (static torsion,
no phase/winding). Differentiate on the **structure-derived phase-slip** test (G1 control 3) — RoPE-proof,
unoccupied.

---

## GATE G2 — "Edits survive iterative-unmasking commitment" (TRACK A, PERSISTENCE, before any steering claim)
**Cost:** steer + **full iterative decode** + refold on ONE best cell (the single most-decodable layer from G1);
~1 GPU, days. This is the **most likely death of the causal half** (Hin26/Zha26g forecast it) — gate it before
building any steering sweep. Kills death (iii).

**Precise claim** A mid-computation edit to the recovered coordinate **survives ESM3's iterative unmasking /
discrete token-commitment** to the final decoded structure — i.e. ESM3 is *not* subject to the masked-pLM
wash-out of Hin26 (edits die in the activation→token-sample→re-embed cycle; foldability collapses >~15%
coverage) or the off-manifold collapse of Zha26g.

**Cheapest falsifying experiment**
On the single best (layer, variable) cell: (A) steer, re-run ONE forward pass, decode structure tokens at
targeted masked positions → backbone → Δ(phi,psi) = single-pass upper bound. (B) apply the *same* nudge, then
let the **full iterative unmasking schedule** run to a completed structure (two variants: steer-once-then-free-
decode; steer-at-every-step); recompute final (phi,psi). **Persistence ratio = final Δ / single-pass Δ.**

**Null / baselines / controls**
- Matched-norm **random-direction** control (|z|>2 band).
- **Coverage sweep** (import Hin26's axis): vary fraction of positions steered/rewritten; locate the coverage at
  which foldability (pLDDT/pTM) breaks.
- **Validity filter, 3 legs (mandatory):** Mahalanobis typicality vs random-AA reference (Zha26g), pLDDT/pTM,
  clash + ProteinMPNN→refold "designability." Report Δ only for candidates that pass — collapse-driven "effects"
  are discarded.
- ≥5 seeds on the cell.

**GO / NO-GO**
GO if persistence ratio clears the matched-random band (|z|>2), is monotone in α, at typicality **above** the
collapse threshold, for structures that pass the validity filter. NO-GO if final Δ is null everywhere, or only
"works" by driving activations off-manifold (collapse the filter catches).

**On NO-GO** **Do NOT kill the project — convert it.** This is the honest, publishable **commitment/persistence
negative**: naive mid-network edits are erased by iterative unmasking on ESM3 structure tokens (extends
Jed26b/Hin26/Zha26g from one-layer/ESM-2/Boltz to a layer map on ESM3). Reframe the causal half as a calibration
result; the logit-level/sampling control (Zho26h) becomes the "what actually works" baseline. Pillar 2
(wrap-around causal) is then reported as "structurally impossible under commitment," not claimed as a win.

**Scoop & differentiation** Hin26 (masked-pLM edit persistence), Zha26g (off-manifold collapse, ESM-2 only),
Jed26b (decodable≠steerable, Boltz-1). None uses ESM3 structure tokens or a layer-resolved persistence axis.
Differentiate on ESM3 + structure-token decode + the coverage/collapse joint report.

---

## GATE G-B2 — "The physical-Kd direction is DIFFERENT from the model's self-confidence" (TRACK B, truth≠confidence)
**Cost:** direction-angle measurement only; cheap. Pre-empts the ipSAE Goodhart death before any Track-B steering.
Kills death (v) at its root.

**Precise claim** The internal direction that carries experimental Kd is **not collinear** with the model's own
interface self-confidence (ipSAE/ipTM) direction — truth and confidence occupy different subspaces.

**Cheapest falsifying experiment**
Recover the experimental-Kd direction (from G-B1's leakage-clean probe) and the ipSAE/ipTM direction (probe
rep→ipSAE, Dunbrack-lab score, github.com/dunbracklab/IPSAE). Measure principal angles / cosine between the two
subspaces on a held-out split; partial correlation of each with the other removed.

**Null / baselines / controls**
- Matched-norm random-direction null for the angle distribution.
- ipSAE is a **CONTRAST, never a steering target** (steering toward it is circular Goodhart).
- Keep a **model-independent geometric interface descriptor** (buried surface area / contacts) as a third,
  distinct target — the physical-Kd direction should align better with it than the confidence direction does.

**GO / NO-GO**
GO if principal angle between Kd-direction and ipSAE-direction is large (e.g. cosine ≤ ~0.5) and Kd retains
predictive power with ipSAE partialed out. NO-GO if the two are collinear (the "Kd feature" is just the model's
confidence re-labelled).

**On NO-GO** **KILL the truth-vs-confidence claim for Track B.** Report the collinearity as the finding: the
model does not separate physical binding strength from its own confidence (a calibration/trust negative — still
publishable, and defensively useful). Do not run any ipSAE-adjacent steering.

**Scoop & differentiation** No prior work separates experimental-Kd from ipSAE direction on a co-folding model.
PairSAE used the model's own affinity (confidence-adjacent) — this gate is precisely the control PairSAE lacked.

---

## GATE G3 — "Decodable ≠ steerable, AND geometry beats scalar" (TRACK A, the positive contribution)
**Cost:** ESM3 layer sweep × 3 directions × α-sweep × seeds + decode + validity; expensive; only after G1+G2 GO.
Kills death (vi) and the "scalar already works" killer.

**Precise claims**
- G3a: the most-**decodable** layer is **not** the most-**steerable** layer (readable≠causal; a layer-resolved
  efficacy-ratio map on ESM3 — nobody has done this; Steenwyk steered one layer).
- G3b (geometry-lift): a geometry-matched (torus/curved-coordinate) direction beats a **scalar SAE direction** at
  **matched norm** on validity-filtered, wrap-aware realized Δ(phi,psi).

**Cheapest falsifying experiment**
Probe decodability by layer (free). At each layer inject three **matched-norm** directions — (i) geometry-matched
coordinate, (ii) Steenwyk-style scalar structure-SAE direction, (iii) matched-norm random — decode→backbone→
Δ(phi,psi), wrap-aware. Build the 2-axis (decodable, steerable) map.

**Null / baselines / controls**
- Matched-norm random-direction control at every layer (the Jed26b/Xun26 null band).
- Scalar-SAE direction is the **must-beat baseline** (Steenwyk Ste26, dose r=0.87 at L33 — so "scalar can't
  steer" is false; the claim must be the quantitative geometry lift).
- Validity filter (Mahalanobis ⊕ pLDDT/pTM ⊕ ProteinMPNN-refold) on every steered output.
- ≥5 seeds; report efficacy ratio with CI.

**GO / NO-GO**
G3a GO if argmax(decodable) layer is significantly weaker as a steering site than argmax(steerable), on
validity-filtered Δ. G3b GO if curved > scalar at matched norm with ≥5-seed separation. NO-GO(G3a→coincide):
downgrade to a "probe-and-steer" benchmark note. NO-GO(G3b→tie): **the geometry headline collapses** (Steenwyk's
scalar is enough) → keep G3a as the sole positive claim.

**On NO-GO** Not fatal: G3a alone (layer-resolved efficacy map + the G2 persistence negative) is a workshop-scale
paper. G3b tie folds "geometry-matched vs scalar vs matched-random" back to being a *control inside the paper*,
exactly as the brief instructs ("not the headline").

**Scoop & differentiation** Ste26 = scalar SAE steering of ESM3 L33 (beat it on matched-norm geometry lift +
layer map + validity filter it lacks). Xun26 = readable-before-causal (LLM) — port to ESM3 quantitatively.
Zho26h = logit/sampling control of ESM3 (non-interp baseline to beat). BioEmu+ESM3 NeurIPS camera-ready
(steering = future work) — UNVERIFIED scoop; check before writing.

---

## GATE G4 — "Wrap-around causal dose-response closes the loop" (TRACK A, crown jewel; only if G2+G3 GO)
**Cost:** expensive; last on Track A. The test a 1-D latent structurally cannot pass.

**Precise claim** Steering along the recovered circular coordinate moves the model's OWN output head monotonically
and **returns to baseline at 2π** (and SKEMPI ΔΔG agreement for Track B, see G-B3).

**Cheapest falsifying experiment** ≥8 targets; traverse the coordinate 0→2π in steps; readout = the model's own
output head (NOT a same-variable probe — readout-independence control); plot dose-response and the return-to-
baseline at 2π.

**Null / baselines / controls** Matched-norm random direction (no signed monotone response, no closure); scalar
direction (cannot close the loop by construction); validity filter on every point; ≥5 seeds.

**GO / NO-GO** GO if output follows the coordinate monotonically and returns to baseline at 2π above the random
band, on validity-passing structures. NO-GO if no closure or indistinguishable from random.

**On NO-GO** Report as a negative: the coordinate is decodable/identifiable but not causally used as a cyclic
variable (joins the decodable≠causal ledger). Pillar 1 (identifiability) can still stand alone.

**Scoop** Engels (Eng24) did modular-arithmetic circular causal tests on LLMs with hand-set targets (no physical
period); differentiate on exact physical ground truth + wrap-around closure on a bio model.

---

## GATE G-B3 — "Kd steering moves physical binding, not just ipSAE" (TRACK B, Goodhart + SKEMPI; last on Track B)
**Cost:** expensive; only after G-B1+G-B2 GO. Second line of defence against death (v).

**Precise claim** Steering the recovered Kd coordinate shifts a **model-independent** interface descriptor (buried
area/contacts) and agrees with **SKEMPI ΔΔG** on mutation (a built-in causal test), rather than only inflating
ipSAE/ipTM (Goodhart).

**Cheapest falsifying experiment** Steer toward higher/lower recovered-Kd; measure (a) model-independent
interface descriptor, (b) SKEMPI ΔΔG sign agreement on matched mutations, (c) ipSAE change. Goodhart = ipSAE
rises while (a)/(b) do not move.

**Null / baselines / controls** Matched-norm random; validity filter; ipSAE tracked as CONTRAST only; SKEMPI
ΔΔG as the physical causal anchor; ≥5 seeds.

**GO / NO-GO** GO if (a) and (b) move with the steer above random while the effect is not explained by ipSAE
alone. NO-GO if only ipSAE moves → Goodhart confirmed → report as the honest Goodhart calibration negative.

**On NO-GO** Publish the Goodhart finding (defensively valuable: shows affinity-steering a co-folding model games
its own confidence). **Ethics:** no released affinity-maximizing steering pipeline; benign complexes only
(exclude toxins/viral-fusion/select-agents); dual-use section needs a human/compliant pass.

---

## EVOLVE — the single change that most reduces wasted compute, + the MPR

**Single highest-value reordering + added control (do this):**
Introduce ONE shared **nuisance-only baseline** and make it the common null for *all* gates, computed before any
expensive step: a decoder that sees ONLY {position index / RoPE phase} ∪ {training-set-membership + sequence/
ligand cluster ID} ∪ {model self-confidence (pLDDT for A, ipSAE/ipTM for B)} and no learned feature. This single
control simultaneously pre-empts deaths (ii) RoPE, (iv) leakage, and (v) Goodhart — three of the six — with one
cheap artifact, and every gate's GO threshold becomes "beat the nuisance baseline by a pre-registered margin."
The accompanying reordering: **fold the identifiability-up-to-isometry test (G0b) into the first CPU run** so one
~2 CPU-hour job decides both the dead "curved" headline *and* whether the surviving kernel is even possible —
and **run the two pure-probe gates (G-B1 leakage, G-B2 ipSAE-collinearity) and G1 (RoPE) to completion before a
single steering/decode job is launched**, because every steering gate (G2/G3/G4/G-B3) is 10–100× costlier and is
wasted if a probe-level gate already fails. Net effect: the ~800–1500 GPU-h steering budget is never spent until
four CPU/probe-level gates have cleared.

**Minimum Publishable Result (survives if all ambitious causal claims fail):**
A **leakage-aware, exact-geometry ground-truth benchmark + calibration/negative paper** with four stand-alone
pieces, any subset of which publishes:
1. The ground-truth suite itself: (phi,psi) torus / helix-phase / non-periodic SASA computed by code from
   experimental structures (not DB labels), with the RoPE-proof / positional-control protocol and the
   nuisance-only baseline — a reusable benchmark the field lacks (OPEN gap #8).
2. **Identifiability-up-to-isometry** recovery of the coordinate (G0b + G1), reported honestly with or without
   the curved block (G0a expected to tie — stated as a *control result*, crediting SASA).
3. The **commitment/persistence negative** (G2): naive mid-network edits are erased by iterative unmasking on
   ESM3 structure tokens — extends Hin26/Zha26g/Jed26b to ESM3 structure tokens + a layer/coverage map.
4. The **truth≠confidence calibration** (G-B2): the angle between the experimental-Kd direction and the model's
   ipSAE self-confidence (or the honest finding that they are collinear) — a trust/audit result.
This MPR is defensive, scoop-robust (it concedes SASA/Steenwyk/Skopek up front and contributes the
exact-ground-truth + leakage-aware + persistence pieces they lack), and reachable even if G3/G4/G-B3 all NO-GO.

---

## RANKED TOP-3 GATES MOST LIKELY TO KILL, + EARLIEST TEST POINT
1. **G2 — Persistence/commitment wash-out (death iii).** Highest kill probability: two papers (Hin26, Zha26g)
   already show masked-pLM edits dying in decode / collapsing off-manifold; ESM3's iterative unmasking + discrete
   token commitment is the same mechanism. Kills the entire causal half if it fires. **Earliest test:** one GPU,
   days, on the single best (layer, variable) cell — immediately after G1 clears; does not need any full sweep.
2. **G-B1 — Kd is leakage, not learning (death iv).** Co-folding Kd prediction notoriously inflates on random
   splits; PairSAE's circularity shows how easy this is to get wrong. **Earliest test:** cheap probe on cached
   Boltz reps under time + scaffold-cluster splits vs the leakage probe — runnable in parallel with G0/G1, no
   steering, before any Track-B investment.
3. **G1 — RoPE manufactures the periodicity (death ii).** ESM3 uses RoPE; with hundreds of frequencies some pair
   sits near 100°/res by chance → a false-positive machine for the phase headline. **Earliest test:** probes on
   cached activations (helix-vs-coil at matched index; trained−random-init; structure-phase-slip partial
   correlation) — no steering, before G2.

Honorable mention (cheapest of all, run first regardless): **G0a** confirms the "curved geometry" headline is
dead at ~2 CPU-hours (SASA already owns it) — it reroutes the project rather than killing it, which is exactly
why it must be first: it stops weeks of curved-dictionary engineering before it starts.
