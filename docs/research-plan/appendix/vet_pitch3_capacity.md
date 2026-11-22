# Vetting PITCH 3 — "Measure the capacity wall instead of proving it"

Reviewer: senior sparse-coding / compressed-sensing + interpretability critic. Blunt.
Research/planning only; no repo changes. Date 2026-11-20.
Prior art via Undermind workspace 33899782-4f3e-48f1-9e4a-a3b6ac8a7af7 (full-text reads) + WebSearch.
Decision rule applied: prefer a small decisive test over a proof — UNLESS a concrete paper shows it already done / shown not to work. **It does.**

PITCH 3 as pitched: plant manifolds of known covering number into activations; count scalar-SAE atoms vs ONE
curved block needed to reconstruct to tolerance ε; headline = atoms-vs-covering-number curve (scalar climbs,
curved block flat). "The empirical face of: single-direction SAEs pay a covering-number penalty, a
curvature-matched block pays O(1)."

================================================================================
## 0. ONE-SCREEN VERDICT — **ALREADY DONE (the curve + the theorem) + CONFOUNDED (the "curved" part)**
================================================================================
The headline figure and its underlying theorem are **already published**, in a paper the earlier passes did not
read to the metal: **Dal26, "Subspace-Aware Sparse Autoencoders" (SASA), arXiv 2606.06333, NeurIPS 2026.**
SASA already:
1. **DEFINES the covering number** — Definition 2, the "k-sparse covering number"; Theorem 4 is literally titled
   "Covering number of feature subspace."
2. **PROVES the scalar lower bound** — Theorem 4: a single-direction (k=1) decoder needs
   `L_i^(1)(ε) ≥ C·(t/ε)^(d_i−1)` directions to reconstruct the radius-t sphere inside a d_i-dim feature
   subspace to tolerance ε (exponential in intrinsic dimension; Ω(1/ε) for a circle). This **is** your T1.
3. **PLOTS the empirical atoms-vs-covering curve** — Figure 2(c): empirical # decoder directions N vs 1/ε
   (log-log) for a **circle (d=2), sphere S² (d=3), helix (d=3)**, with empirical counts **tracking the
   Theorem-4 lower bound**. This **is** your "scalar SAE climbs" curve.
4. **SHOWS the block stays flat** — Figure 1: on planted circle/sphere/helix in R^64 + 5% noise, scalar SAEs
   use 10–44 atoms while **ONE SASA group of rank 2 (circle) / rank 3 (sphere, helix)** captures the whole
   feature. This **is** your "one block flat" side.
5. **PROVES the O(1) upper bound** — Proposition 10 + Theorem 11: a block of rank r≥d_i represents the whole
   slice with zero error, and under Top-1 gating every global minimizer activates a single block recovering the
   subspace. This **is** your T2 (constant-capacity block), on the nose.

So the "atoms-vs-covering-number relationship" **has been measured**, the "scalar SAE pays a covering penalty"
**has been proven**, and the "one block pays O(1)" **has been proven and measured.** PITCH 3's headline is
**scooped as a whole**, not in parts.

The ONLY thing SASA does not do is use a **curved** block — its blocks are **linear Euclidean subspaces**
(D_k ∈ R^{d×r}, nuclear-norm rank penalty). But — and this is the kill — **for every manifold PITCH 3 names
(circle→R², torus→R⁴, sphere→R³, swiss-roll→R³, tree→low-dim), a FLAT Euclidean block already pays O(1)**,
because these manifolds live inside a low-dimensional **linear** subspace and a free block code `x̂=D z, z∈R^r`
reconstructs any point in that subspace to ≈0 error. **Curvature buys nothing on the reconstruction/capacity
axis** that a flat block of the same dimension does not already buy. The covering-number penalty is a
**single-active-atom (k<d) linear-span** phenomenon, not a curvature phenomenon. Your "curvature-matched block"
qualifier therefore does not isolate anything in the capacity story — it reduces to SASA's Euclidean block.

**Bottom line:** measuring this is NOT enough for ICML, because it is already measured AND already proven by
SASA; and the one novel word ("curved") does not survive contact with the Euclidean-block control. Do not run
PITCH 3 as pitched. The survivable idea is a *different* claim (identifiability / monosemantic recovery up to
the isometry group — your T3), which is NOT what PITCH 3 measures.

================================================================================
## 1. PRIOR-ART KILL CHECK  (ALREADY DONE / PARTIAL / OPEN, with confidence)
================================================================================
Confidence: HIGH = read the relevant figures/theorems in full text this pass; MED = read abstract+targeted
Q&A; UNVERIFIED = snippet only.

### The decisive one
- **Dal26 "Subspace-Aware Sparse Autoencoders" (SASA), arXiv 2606.06333, NeurIPS 2026 — ALREADY DONE. HIGH.**
  Owns: covering-number definition; scalar lower bound Ω((t/ε)^{d−1}) PROVEN (Thm 4); empirical atoms-vs-(1/ε,
  intrinsic-dim) curve for circle/sphere/helix tracking the bound (Fig 2c); scalar-fragments-vs-one-block
  illustration (Fig 1); one-block O(1) consolidation PROVEN (Prop 10, Thm 11). Uses the word "covering number"
  explicitly. **Does NOT: use a curved block (linear subspaces only), test a torus, or sweep both models'
  atom-counts across a covering-number axis in one plot (Fig 2c is the scalar count; Fig 1 is a fixed-width
  SAE-vs-SASA snapshot — the two are not overlaid as a single scalar-climbs/block-flat figure).** That cosmetic
  gap (overlay the two into one figure, add a torus) is the ENTIRETY of what PITCH 3 would add over SASA, and it
  is not a contribution.

### The ones you already flagged
- **Zah26 "The Geometric Wall", arXiv 2605.09887 (ICML 2026) — PARTIAL, different quantity. HIGH.**
  Does NOT contain your curve. It correlates four geometric summaries (intrinsic dim, multi-scale curvature κ_ms,
  tangent variation, heterogeneity) with **released Gemma-Scope SAE scaling exponents** α_ℓ(k) and a fitted loss
  floor across 68 layers (Fig 10: geometry on x, exponent/floor on y). **No planted manifold of known covering
  number, no atom-count-at-fixed-ε, no covering number, no remedy** (local-frame-field / atlas-chart atoms are an
  Appendix-I wish-list, never implemented or measured). The "wall" is a second-order-residual *argument*, not a
  theorem or a bound. → Your earlier note "measured curvature→SAE width-scaling, no remedy" is correct; it does
  NOT pre-empt PITCH 3's curve, but SASA does.
- **Jer26 "A Deeper Analysis of BSF", arXiv 2608.27515 — PARTIAL. HIGH.**
  Manifold Zoo = segment×64, circle×10, {disk, sphere, torus, Möbius, swiss-roll, helix}×9, embedded in R^128 by
  random orthonormal frames, samples = sparse sum of 4 features. **No tree.** Reports mean recovery R² (SAE 0.36
  vs BSF 0.77, Table 1) and **# BSF blocks to reach R²>0.8** (Fig 4: 1, 2, or "never within 8") — it counts
  BLOCKS, never scalar atoms, and **never vs covering number**. Blocks are **Euclidean subspaces only; no curved
  blocks.** Mixed-dimension blocks "failed" = failed to self-sort features into right-sized blocks under vanilla
  TopK / normalized TopK / MDL / JumpReLU (Sec 5), not a training crash. → Hands you a zoo; does NOT contain your
  curve and does NOT build the curved block. But the SAE-vs-(Euclidean)block separation it measures is the same
  one SASA proves, so the "block beats scalar" headline is doubly owned.
- **Bha26 "Do SAEs Capture Concept Manifolds?", arXiv 2604.28119 — PARTIAL (strong). HIGH.**
  Planted 8-manifold zoo (circle, sphere, torus, Möbius, swiss-roll, helix, disk, segment) with known embeddings
  (Table 4). Global (compact atom group spans manifold) vs local (tiling) vs **dilution**. Theorem 1 = subspace
  capture by **k atoms where k = ambient-subspace dim** (NOT intrinsic dim / covering number). Appendix G gives
  the **qualitative** "# atoms ∝ volume of M in atom-units, not ambient dimension" — the *verbal* covering-number
  penalty, but **no tolerance, no constants, no curve.** Sweeps SAE sparsity budget k, not covering number.
  **No curved-block remedy; names manifold-aware featurizers as future work.** → This is the conceptual seed;
  SASA turned the seed into the theorem+curve. Between Bha26 (seed) and Dal26 (theorem+curve), PITCH 3 is boxed in.
- **Mic25b "Understanding SAE scaling in the presence of feature manifolds" (Michaud/Gorton/McGrath, 2025)
  — PARTIAL. HIGH.** Plots **total SAE loss vs # latents** for synthetic hyperspheres S^1..S^8 and radial shells
  (Fig 3, 5, 6); shows hollow spheres keep tiling while radial-variation manifolds plateau near ~2d_i latents;
  gives a feature-frequency **latent-allocation** scaling law (discovery efficiency D(N)/N), NOT atoms-vs-covering
  number at fixed ε. **No block/curved remedy** (the ~2d_i plateau is an observation about ordinary SAEs). → The
  "latents tile a circle" mechanism precedent; not your curve, not a remedy.

### Supporting / classical (confirms the math is standard, i.e. a theorem adds little)
- **Covering-number / nonlinear n-width:** the Ω((t/ε)^{d−1}) count is the classical ε-covering number of a
  d-manifold (metric entropy; Kolmogorov n-width / nonlinear m-term approximation). A monotone single-direction
  atom reconstructs along a ray, so tiling a curved compact manifold to ε is an interval/cap cover — textbook.
  SASA's Theorem 4 is a clean restatement of this for the SAE setting. A "new" theorem here would be re-deriving
  metric entropy. OPEN-but-trivial; not an ICML contribution. HIGH.
- Dor26 "How Optimality Structures Sparse Dictionaries" (arXiv 2606.02385): SAE optimality, feature-splitting,
  dense antipodal pairs — Euclidean, 1-D, no manifolds/covering. Source of the crux-rebuttal risk (2 scalar
  atoms span a plane). MED.
- Tan25f "Unified Theory of SDL" (arXiv 2512.05534): SDL landscape, **negative** identifiability for 1-D
  features, feature-anchoring remedy — no blocks/curvature/covering. MED.
- Hin25b "Projecting Assumptions: Duality Between SAEs and Concept Geometry" (2025, 69 cites) — PDF unreadable
  this pass; abstract/title duality-of-geometry, unlikely to contain the covering-atom curve. UNVERIFIED (low risk).

### Explicit answer to the posed question
> Does Zah26 or Jer26 already contain this exact curve, and if so what is left?
- **Zah26: NO** (it correlates curvature with released-SAE scaling exponents; no planted covering-number curve,
  no remedy).
- **Jer26: NO** (R² recovery + block-count-to-R²>0.8 on a Euclidean-block zoo; no scalar-atoms-vs-covering curve,
  no curved block).
- **BUT Dal26 (SASA) DOES** contain the curve (Fig 2c), the theorem (Thm 4), AND the O(1)-block remedy (Thm 11,
  Fig 1) — in Euclidean form. **What is left after SASA = only "make the block curved" and "add a torus," and
  neither isolates a new effect on the capacity axis (see §2/§4).** So the honest finding is not "Zah/Jer left it
  open" — it is "a THIRD paper you had not fully read closed it."

================================================================================
## 2. MAKE IT RIGOROUS  (definitions + the fair comparison + the controls that would expose it)
================================================================================
Even though the verdict is negative, here is the rigorous framing — because it is exactly the rigor that makes
the negative result obvious before you spend a GPU-hour.

### 2.1 Covering number, operationally (per planted manifold M embedded in R^D)
N(M, ε) = minimum # of ε-radius balls (ambient L2) to cover M. Known rates:
- **Circle S¹(radius t):** N = Θ(t/ε). (SASA's (t/ε)^{d−1} with d=2.)
- **Torus T²=S¹×S¹:** N = Θ(t²/ε²). (Not tested by SASA — the only genuine gap, and a trivial one.)
- **Sphere S²:** N = Θ(t²/ε²). (SASA d=3 case.)
- **Swiss-roll (2-manifold, area A):** N = Θ(A/ε²).
- **Tree (L δ-separated leaves, depth h):** N = Θ(L) at leaf resolution; a hyperbolic chart changes constants,
  not the O(L) count. (Not tested by anyone in this zoo; also not a covering-number-vs-curvature story.)
The covering number is a function of **intrinsic dimension and extent**, not of curvature per se — which is
precisely why the "curved block" framing is mis-aimed (§4, Killer 2).

### 2.2 Reconstruction tolerance ε — state ALL of these or the claim is undefined
- Metric: relative L2 of the M-component (FVU) **averaged over M** (or sup over M for a worst-case cover).
- **Per-point active-atom budget k (THE load-bearing knob):** the penalty Ω((t/ε)^{d−1}) holds **only for
  k < d**. For a circle, k=1 gives Ω(1/ε); **k=2 kills it** (2 atoms span the plane, reconstruct to ~0). Real
  TopK SAEs run k≫1, so you MUST report the curve as a function of k, and concede the penalty is a small-k regime.
  This is the Dor26/crux rebuttal ("SAEs already encode a circle in ~2 correlated features"), and SASA already
  formalizes the escape (k-sparse covering): the penalty is about k-dim spans, not single directions in isolation.

### 2.3 The FAIR comparison — matched BITS/MDL, not matched atoms
- Quantize all codes; compute description length (bits); plot reconstruction-R² (or circular-R²) **vs bits**.
- A geometry wins only if it **dominates the rate–distortion frontier**. Matched-unit or matched-atom comparisons
  are capacity artifacts (a b-dim block trivially has more parameters than a scalar latent).

### 2.4 Controls — and which one is decisive
- random-init dictionary + random-init model (Heap–Aitchison trap).
- ≥5 seeds + Tournament Top-K (Jer26) for seed stability; report seed variance of the recovered geometry.
- rate/bits-matched scalar SAE alongside every block.
- **THE KEY CONTROL (isolates CURVATURE from CAPACITY): a Euclidean block of EQUAL dimension b, at matched
  bits.** Curved block vs Euclidean block of the same b and the same MDL. **Predicted result from SASA's own
  theory: they tie** on reconstruction-to-ε for every low-dim-embedded manifold (both O(1)), because a flat block
  already spans the subspace the manifold lives in. If they tie, "curvature-matched" adds nothing to the capacity
  claim and only the (already-scooped) block-vs-scalar separation remains. **This single control is why PITCH 3's
  thesis cannot stand: it is designed to show the curved block beats scalar, but the honest control shows the
  curved block does NOT beat the Euclidean block, and the Euclidean-block-beats-scalar result is SASA's.**

================================================================================
## 3. SMALL DECISIVE TEST  (hours, CPU — but note it is now a DISCONFIRMING test, not a headline-maker)
================================================================================
Run it ONLY to confirm the kill cheaply before abandoning the capacity framing (≈1–2 CPU-hours; no GPU, no model).

**Protocol.**
1. Plant M ∈ {circle→R², torus T²→R⁴, sphere S²→R³, swiss-roll→R³} embedded in R^D (D=64) via a random
   orthonormal frame; additive N(0, σ²), σ small; sample 50–200k points. Vary covering number by sweeping
   tolerance ε (equivalently radius t / density).
2. Fit three families on cached points (no model):
   (a) scalar TopK SAE, per-point k=1 then k=2;
   (b) **Euclidean** block, b = ambient-subspace dim of M (2/4/3/3);
   (c) **curved** block matched to M (S¹ / T² / S² code via magnitude–direction split).
3. For each, record **# atoms (and # bits) to reach FVU ≤ ε**, as ε shrinks → the atoms-vs-covering-number curve,
   AND the matched-bits rate–distortion frontier.

**Go/No-go — and it is pre-registered to fail the curvature claim:**
- The scalar-k=1 curve WILL climb ~N(M,ε) (reproduces SASA Fig 2c — not a contribution).
- **GO (curvature is real) iff:** curved block strictly dominates the **Euclidean block of equal b at matched
  bits** on reconstruction-to-ε, for at least one manifold, with ≥5-seed separation. **Predicted: NO-GO** — they
  tie, because the manifolds live in low-dim linear subspaces. If NO-GO, the capacity/covering framing of PITCH 3
  is dead (curvature buys no reconstruction capacity a flat block lacks) → **do not proceed to any real model.**
- The ONLY way to a GO is to plant a manifold whose **minimal linear embedding dimension ≫ intrinsic dim + 1**
  (high extrinsic curvature, e.g. a Veronese-type / highly-wound embedding where a flat chart must be large but a
  curved low-dim chart suffices). That is a contrived synthetic regime with no demonstrated biological instance,
  and it changes the paper from "measure the known wall" to "construct an adversarial embedding" — a different,
  weaker pitch. Flag honestly.

================================================================================
## 4. VERDICT
================================================================================
**Is measuring this (vs proving) enough for ICML?** — **No, on two independent grounds.**
1. It is **already measured** (SASA Fig 2c) **and already proven** (SASA Thm 4 + Thm 11), with a circle/sphere/
   helix planted zoo and the word "covering number." PITCH 3 would add a torus and overlay two existing figures —
   cosmetic. Per the user's decision rule, a concrete paper shows it done → **do not run it as the headline.**
2. Even ignoring SASA, the **"curved block" is a confound, not a contribution**: a flat Euclidean block of equal
   dimension pays O(1) on every manifold in the zoo, so the headline figure would show curved ≈ Euclidean ≈ flat
   ≫ scalar — i.e. it measures the block-vs-scalar capacity gap (owned by SASA/Jer26), not a curvature effect.

**Should the theorem be attempted at all, given the small-test preference?** — **No.** The theorem exists
(SASA Thm 4 is your T1; Thm 11 is your T2). Re-deriving it = re-deriving classical metric entropy. There is no
theorem to win here. The small test is worth running ONCE, purely to confirm the kill cheaply (§3), not to chase
a headline.

**Two most likely killers (ranked):**
1. **SASA scoop (fatal, already materialized).** Dal26 proves the covering-number lower bound, plots the
   empirical scalar-climbs curve (circle/sphere/helix), and proves + shows the one-block O(1) remedy. Published,
   NeurIPS 2026. PITCH 3's headline and its theorem are both inside this paper.
2. **Curvature ≠ capacity confound (fatal, structural).** The covering-number penalty is a small-k linear-span
   phenomenon; a flat Euclidean block already removes it for low-dim-embedded manifolds. Curvature only matters
   for **identifiability / monosemantic coordinate recovery** (recovering θ up to the isometry group — your T3),
   which PITCH 3 does not measure. So PITCH 3 measures the one axis where curvature is irrelevant.

**Scoop exposure (ranked):** (1) **Dal26/SASA — critical, total** (covering-number theorem + curve + block
remedy, Euclidean). (2) **Bha26 — high** (planted zoo + qualitative "tiles ∝ volume"). (3) **Mic25b — med**
(loss-vs-latents tiling, Goodfire-adjacent, actively extensible). (4) **Jer26 — med** (Euclidean-block zoo,
block-vs-scalar). (5) **Zah26 — low for this curve** (different quantity, but owns "curvature predicts SAE
scaling"). Goodfire/Arc + the SASA group (Penn State) are both actively in this exact lane.

**What survives (the pivot, NOT PITCH 3):** the only un-scooped, curvature-dependent claim is **identifiability
up to the manifold's isometry group** — a flat block spans the circle's plane but gives no monosemantic angle;
a curved (spherical/toroidal) block's code *is* the angle, enabling the **wrap-around causal dose-response** on a
bio model's own output head (plan_task1_geometry §3–4; plan_icml_novelty T3). That is a RECOVERY/CAUSAL claim,
not a covering-number CAPACITY claim, and SASA's negative-space (it only recovers the linear subspace, not the
intrinsic coordinate) is the springboard. **Redirect effort from "measure the capacity wall" (done) to "curved
blocks recover the intrinsic coordinate + pass the wrap-around causal test where flat blocks and scalars
cannot."** Do not spend the flagship's headline on PITCH 3's capacity curve.

================================================================================
## 5. KEY REFERENCES (workspace cite keys + arXiv)
================================================================================
- **Dal26** SASA, arXiv 2606.06333 (NeurIPS 2026) — covering-number theorem (Thm 4), empirical curve (Fig 2c),
  one-block remedy (Prop 10, Thm 11). THE SCOOP.
- Zah26 Geometric Wall, arXiv 2605.09887 (ICML 2026) — curvature↔SAE-scaling correlation; no curve, no remedy.
- Jer26 Deeper BSF, arXiv 2608.27515 — Euclidean-block manifold zoo; block-count, not covering curve.
- Bha26 Do SAEs Capture Concept Manifolds?, arXiv 2604.28119 — planted zoo, global/local/dilution, "tiles ∝
  volume" (qualitative).
- Mic25b Understanding SAE scaling with feature manifolds (Michaud/Gorton/McGrath, 2025) — loss-vs-latents,
  tiling, allocation law.
- Dor26 arXiv 2606.02385 (antipodal-pair / splitting — crux rebuttal); Tan25f arXiv 2512.05534 (SDL
  identifiability); Fel26c BSF arXiv 2606.25234 (Euclidean blocks, MDL — PDF unread this pass).
- Classical: ε-covering number / metric entropy / Kolmogorov n-width — the Ω((t/ε)^{d−1}) count is textbook.

NOTE: Dal26/SASA, Zah26, Jer26 are all 2026 and fast-moving; a human should re-scan arXiv + goodfire.ai/research
immediately before any pivot, especially for a curved-block or identifiability follow-up to SASA.
