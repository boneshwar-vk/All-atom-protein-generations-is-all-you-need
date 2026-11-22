# Vetting PITCH 1 — Learnable-curvature sparse dictionary (per-block signed kappa)

Reviewer stance: senior researcher in representation geometry + sparse coding + mech-interp, blunt critic.
Research/planning only; no repo changes. Date 2026-11-20. Prior art verified via Undermind workspace
33899782-4f3e-48f1-9e4a-a3b6ac8a7af7 + WebSearch. "UNVERIFIED" = not confirmed from a primary source this pass.

PITCH 1 restated: each feature-block of a block-sparse dictionary carries a differentiable curvature
scalar kappa (kappa>0 spherical / =0 Euclidean / <0 hyperbolic, stereographic/gyrovector parameterization);
the dictionary "discovers each concept's intrinsic geometry." Headline figure = learned kappa recovers
known geometry of structural variables unsupervised (positive for dihedral/phase, ~0 for SASA, negative
for fold hierarchy).

---

## 0. VERDICT (read first)

**Status of the core atom: OPEN but NARROW, and the headline claim is PARTIALLY PRE-EMPTIED / AT HIGH RISK.**

- The **mechanism** PITCH 1 proposes — a *differentiable per-component signed curvature* in a product of
  constant-curvature model spaces, optimized by gradient descent, able to **change sign** between
  spherical/Euclidean/hyperbolic via the projected-sphere + Poincare parameterization — is **ALREADY DONE**
  and is 5–7 years old: `Gu18` (product spaces, learned per-factor curvature, 2018) and especially
  **`Sko19` Mixed-curvature VAE (2019), which has a "universal curvature" model whose components learn
  their curvature AND are explicitly allowed to change sign.** PITCH 1's parameterization is essentially
  Sko19's, moved from a VAE latent to a sparse dictionary. The "stereographic/gyrovector signed-kappa"
  trick is not novel.
- The **exact headline experiment** — "learned kappa recovers the known geometry (sign) of the data
  unsupervised" — was **effectively attempted in `Sko19` and did not cleanly work.** On their tree-like
  synthetic data (branching diffusion), the learnable-curvature components **did not converge to a definite
  curvature**; they drifted to roughly (-0.1, +0.1), i.e. near-Euclidean, with no sign recovery and no
  identifiability analysis. The authors state in plain text that **"learning curvature directly is badly
  conditioned."** This is the single most important finding of this review: **the thing PITCH 1 wants its
  money-shot figure to show is the thing the closest prior work failed to show.** That is not quite a
  published "shown not to work" kill (nobody set out to do sign-recovery-as-the-result and reported failure),
  but it is a concrete, primary-source caution that the headline is fragile.
- The **combination** "learnable per-block signed curvature *inside a sparse dictionary/SAE* on
  *foundation-model activations*, with *per-concept geometry discovery*" is **not found** in Undermind's
  corpus or on the web — **OPEN** (medium-high confidence). But it reads as "**mixed-curvature VAE
  (Sko19) + block sparsity (BSF/SASA)**," which is exactly the framing the user asked me to stress-test,
  and a geometry-literate reviewer will say so in one line.
- **Scoop-adjacent:** `Gro26` "Text Has Curvature" is NOT a scoop (it *measures* a per-slot inference
  curvature from a frozen MLM at inference time; **no dictionary, no learned kappa, no geometry recovery**).
  HyperSAE (hyperbolic SAE on LLMs, PyPI) remains **UNVERIFIED** and is a *fixed*-curvature, single-sign
  (hyperbolic-only) method — it does not do learnable signed per-block kappa, so it is adjacent, not a
  direct scoop of PITCH 1, but it owns "non-Euclidean SAE on LLM activations."

**Bottom line for the decision rule:** there is **no single paper that fully does PITCH 1**, so by the
user's rule we do NOT kill on prior art alone — we design the cheap decisive test. BUT the cheapest
decisive test is *pointed straight at the Sko19 failure mode* (is kappa identifiable/recoverable at all?),
because if curvature is not recoverable on planted synthetic data, the whole pitch is dead before any bio.

---

## 1. PRIOR-ART KILL CHECK (per the user's explicit checklist)

Confidence scale: HIGH / MED / LOW. All arXiv IDs are as returned by the workspace; treat unconfirmed
ones as UNVERIFIED.

### 1.1 Learnable per-component signed curvature (the mechanism) — ALREADY DONE (not in an SAE). HIGH.
- **`Gu18` — Learning Mixed-Curvature Representations in Product Spaces (ICLR 2019).** Embeds data in a
  product of spherical/Euclidean/hyperbolic model spaces; **learns the curvature of each component** via
  Riemannian optimization; uses a **discrete sectional-curvature heuristic** to pick the signature
  (how many components, which sign). This is the canonical "let the data pick the geometry" paper. Not a
  sparse dictionary, not interpretability, not a pretrained model's activations.
- **`Sko19` — Mixed-curvature Variational Autoencoders (2019).** THE closest prior art for PITCH 1's
  mechanism. Abstract: product of constant-curvature manifolds "where the per-component curvature is
  **fixed or learnable**." Their **"universal curvature" VAE** uses projected-sphere D and Poincare ball P
  precisely because the hypersphere/hyperboloid "behave poorly as K->0," and lets non-Euclidean components
  **change sign** spherical<->hyperbolic during training. **This is PITCH 1's parameterization, minus the
  sparsity, minus the bio, minus the interpretability framing.** Confirmed by full-text read.
- **"Switch Spaces: Learning Product Spaces with Sparse Gating" (arXiv 2102.08688, 2021).** Learns product
  spaces with a **sparse gating** mechanism selecting components — i.e. **sparsity + curved-component
  selection already co-exist** in the embedding literature (for KG/graph tasks). Adjacent; weakens the
  "sparse + curvature is new" claim. UNVERIFIED in detail (web snippet only).
- Lineage you must cite or be called derivative: `Tab21/Tab21b` (classifiers in product/mixed-curvature
  spaces), `Cru20` (tractable Riemannian manifolds for embeddings), `Bor22b` (product-manifold latent
  graph inference), `Li17` (curvature-aware manifold learning).

### 1.2 Non-Euclidean / Riemannian SPARSE dictionary learning — ALREADY DONE (on DATA, not activations). HIGH.
- `Xie12` Dictionary Learning on Riemannian Manifolds; `Har13/Har15` Grassmann + kernel Riemannian coding;
  `Li13` Log-Euclidean SPD dictionary learning; `Xue21` hierarchical Riemannian pursuit. "Constrain
  dictionary atoms/codes to a (known, fixed) Riemannian manifold" is a mature >10-yr field. **None learns a
  signed curvature; the manifold is fixed a priori.** PITCH 1's twist (learn the sign) is the delta, and
  that delta is owned by the mixed-curvature-embedding line (1.1), not the sparse-coding line.

### 1.3 Learnable curvature INSIDE an SAE / sparse dictionary on NN activations — OPEN. MED-HIGH.
- Direct searches (Undermind semantic + two WebSearches) returned **no** SAE/sparse-dictionary with a
  learnable per-feature/per-block curvature. Nearest misses are all **Euclidean**: `Muc25`/H-SAE and
  `Luo26` (hierarchy by gating, flat), `Fra26` SMIXAE (mixture-of-SAEs for multidim features, flat),
  `Bha26` (do SAEs capture manifolds — analysis, flat atoms), `Cha26e` AEN-SAE and the "coherence-penalty
  curvature phase transition" NeurIPS'26 poster (these use *loss-landscape* curvature, **not** a feature
  geometry), KAN-SAE (per-latent learnable spline activations, not curvature). **So the specific
  combination is unoccupied.** Confidence capped at MED-HIGH because a brand-new arXiv preprint could exist
  below Undermind's indexing; a human should run one `launch_deep_search` before committing.
- **`Gro26` "Text Has Curvature" (2026) — NOT a scoop.** Full-text read: it estimates a signed per-slot
  score kappa in [-1,1] (focus vs fan-out) from a **frozen** masked LM's two-sided belief field **at
  inference time** via Sinkhorn transport divergence. **No sparse dictionary, no learned parameter, no
  recovery of a known latent geometry.** It shares the word "curvature" and the symbol kappa and nothing
  else. Cite it as "curvature-as-a-measured-signal" prior art; it does not threaten novelty.

### 1.4 Non-Euclidean SAE on LLM activations — PARTIALLY DONE (fixed hyperbolic), UNVERIFIED scoop.
- **HyperSAE** (PyPI `hypersae`; self-listed papers "Escaping Flatland..." and "Hyperbolic Sparse
  Autoencoders: Empirical Validation of Poincare Manifold Geometry on LLM Activations"; repo
  github.com/vishal-dehurdle/hypersae). Design per the page: **Euclidean forward pass + hyperbolic
  weight-space optimization**, **fixed hyperbolic (single sign)**, Gemma-2-2B example. **UNVERIFIED** — no
  arXiv text, authors, date, or peer review confirmable this pass. **Relevance to PITCH 1:** it owns
  "hyperbolic SAE on LLM activations," but it is **fixed, single-sign, not learnable-signed-kappa, not
  per-block geometry discovery, not bio.** So it is adjacent and a reason to NOT frame PITCH 1 as "first
  curved SAE," but it does not do the learnable-kappa geometry-discovery thing. A human MUST verify
  HyperSAE before any writeup.

### 1.5 Geometry-of-concepts / manifold-in-SAE — ADJACENT, crowded, Euclidean. HIGH.
- `Par24c` (Geometry of Categorical & Hierarchical Concepts — simplices + orthogonal subspaces, **flat**),
  `Mod25` (origins of representation manifolds — cosine-as-geodesic, **flat metric**, analysis),
  `Zah26` (Geometric Wall — curvature predicts SAE scaling, **diagnoses** the problem, **no curved-code
  remedy**), `Li24i`, `Hin25b`, `Bus25` (Matryoshka). This cluster establishes that "activation manifolds
  have curvature and flat SAEs struggle" is **already the accepted framing** — which *motivates* PITCH 1
  but also means the gap is "provide the curved remedy + show geometry discovery," not "notice curvature."

### 1.6 Bio foundation-model SAEs — ALL EUCLIDEAN SCALAR. HIGH.
- `Set26` (geometric *annotation* of ESM-2 SAE latents — post-hoc labels, closest bio prior art, flat),
  `Ada25`, `Guj25`, `Par25`, `Liu26`, `Ste26`, `Hak25` (isotropy/geometry of protein LMs — analysis, flat).
  **No non-Euclidean / learnable-curvature sparse dictionary on any biological model.** This is the real,
  defensible open lane (same conclusion as plan_task1_geometry.md).

### 1.7 KILL-CHECK SUMMARY TABLE
| Component of PITCH 1 | Status | Key refs | Conf |
|---|---|---|---|
| Learnable signed per-component curvature (sign-changing, stereographic/Poincare) | **ALREADY DONE** (embeddings/VAE) | `Gu18`, `Sko19`, Switch Spaces | HIGH |
| "Learned kappa recovers known geometry unsupervised" (the headline) | **ATTEMPTED & DID NOT CLEANLY WORK** in VAE; badly conditioned; no sign recovery | `Sko19` | HIGH |
| Riemannian/curved sparse dictionary (fixed manifold) | **ALREADY DONE** (data manifolds) | `Xie12`,`Har13/15`,`Li13`,`Xue21` | HIGH |
| Learnable curvature INSIDE an SAE on NN activations | **OPEN** | (none found) | MED-HIGH |
| Non-Euclidean SAE on LLM activations (fixed hyperbolic) | **PARTIALLY DONE** | HyperSAE (UNVERIFIED) | MED |
| "Text has curvature" measured signal | DONE but ORTHOGONAL (not a scoop) | `Gro26` | HIGH |
| Non-Euclidean/curvature sparse dict on a BIOLOGICAL model | **OPEN** | (none found) | HIGH |

---

## 2. METHOD SPEC (minimal math for learnable per-block kappa)

All training on **cached activations only** (no grad through ESM3/the big model), block-sparse across G
blocks, decode into Euclidean activation space. Baseline BSF: code z_g in R^b, xhat = sum_g D_g z_g,
keep l0 blocks by ||z_g|| (block-TopK); b=1 => TopK SAE.

### 2.1 The signed-curvature block (stereographic / kappa-stereographic model)
Use the **kappa-stereographic model** (one chart that unifies sphere and hyperboloid; the Sko19/geoopt
"universal" trick), NOT a raw hypersphere+hyperboloid pair, because the latter blow up as kappa->0.

Per block g, a single learnable scalar kappa_g in R (unconstrained; sign free). Let the block ambient dim
be b (b=2 for a circle/1-D curved feature; b=3 for a 2-surface). Define the kappa-dependent operations:
```
tan_k(x)  = (1/sqrt|k|) tan( sqrt|k| x)   if k>0 ;  x if k=0 ;  (1/sqrt|k|) tanh(sqrt|k| x)  if k<0
artan_k(x)= (1/sqrt|k|) atan(sqrt|k| x)   if k>0 ;  x if k=0 ;  (1/sqrt|k|) artanh(sqrt|k| x) if k<0
```
Encode (exp-map at origin of the kappa-stereographic ball/sphere), decode (log-map), gate, select:
```
v_g   = W_enc,g x                              # raw pre-activation in R^b (tangent at origin)
z_g   = exp0_k(v_g) = tan_k(||v_g||) * v_g/||v_g||    # point on the kappa-manifold (stereographic coords)
t_g   = log0_k(z_g) = artan_k(||z_g||) * z_g/||z_g||  # back to tangent R^b  (t_g = v_g up to the radial warp)
r_g   = softplus(w_g^T x + beta_g) >= 0        # SEPARATE scalar gain = "how strongly present"
u_g   = z_g / (radius factor)                  # unit-direction readout (angle on the circle/sphere)
xhat  = sum_{g in A} r_g * D_g * u_g            # A = active set; D_g in R^{d x b}
```
- **Code / direction:** the *angular* part of z_g (its direction) is the curved feature coordinate; for a
  circle (b=2) that is theta = atan2. For the Euclidean limit kappa->0 this degrades continuously to a flat
  block (exp0 = identity), so the Euclidean SAE is nested as kappa=0 — good for honest ablation.
- **Sparsity gate:** block-TopK on the **gain r_g** (NOT on code norm), so selection is decoupled from the
  curved geometry. (If you instead TopK on hyperbolic distance from origin = monotone in ||z_g||, selection
  collapses to Euclidean-norm TopK — see pitfall (iii).)
- **Loss:** L = sum ||x - xhat||^2 + lambda * (block-sparsity on r_g) + mu * R_kappa.
  - `R_kappa` is a **weak** regularizer on kappa (e.g. small L2 toward 0) ONLY to stabilize, tuned so it
    does not dictate the answer. Report sensitivity to mu — the headline is worthless if kappa is a
    function of the regularizer, not the data.
  - **DO NOT** add a supervised entailment/Poincare-distance loss if you want the "unsupervised geometry
    discovery" headline; adding it makes kappa<0 a near-tautology and imports circularity. (See killer #2.)

### 2.2 Retraction / projection / optimizer
- Projected-gradient is sufficient and simplest: after each step, project z_g to ||z_g|| <= (1-1e-5)/sqrt|kappa|
  for kappa<0 (inside the ball). kappa itself is an ordinary unconstrained parameter with normal SGD/Adam.
- If projected-grad underfits, use `geoopt` RAdam on the kappa-stereographic manifold. geoopt already
  implements the kappa-stereographic (`Stereographic`) manifold with a learnable curvature — **use it;
  do not hand-roll exp/log maps.**

### 2.3 Numerical-stability pitfalls (these are the real engineering cost)
1. **kappa crossing 0:** tan_k/artan_k must be computed with the **series expansion near |kappa|~0**
   (tan_k(x) ~ x + k x^3/3 ...), else 1/sqrt|kappa| * tan(sqrt|kappa| x) is 0/0 and the gradient is NaN.
   This is the exact issue Sko19 flagged ("hypersphere/hyperboloid behave poorly as K->0"); the
   kappa-stereographic chart + series is the fix. Budget time here.
2. **Poincare boundary (kappa<0, ||z||->1/sqrt|k|):** artanh and its gradient explode. Fixes: hard-project
   to radius <= 1-1e-5; clip tangent norms; compute artanh in **float64**; Mathieu-style feature clipping.
3. **Sphere wrap/antipode (kappa>0):** tan(sqrt k x) has a pole at sqrt k x = pi/2; clip ||v_g|| so you stay
   within the first "hemisphere" chart, or the exp-map is multivalued and gradients are garbage.
4. **kappa identifiability (the deepest one, NOT purely numerical):** for a *bounded* code reconstructing a
   finite activation patch, many (kappa, D_g, scale) triples reconstruct equally well — curvature trades off
   against decoder-column norm and the gain r_g. This is why Sko19's kappa drifted and did not converge.
   **Mitigation is experimental, not algorithmic:** you must *measure* whether kappa is recoverable on
   planted data (Pilot 0) before believing any bio kappa. There is no parameterization that fixes
   non-identifiability; only a signal strong enough (a genuinely periodic or genuinely tree-like variable)
   can pin the sign.

### 2.4 Compute order vs plain SAE
- **Same order of magnitude.** Trains on cached activations; per-block overhead is O(b) extra ops for the
  exp/log maps + one scalar kappa gradient. Sphere/kappa-stereographic ~= free over BSF; the float64
  boundary handling for kappa<0 adds a small constant factor and real debugging time, not a complexity
  class. Dominant cost is activation caching + (later) causal interventions, which is geometry-independent.
  **Confirmed: cached-activations-only, ~1x–1.5x a plain SAE of equal width.**

---

## 3. SMALL DECISIVE TEST (1 GPU / days)

The decision rule says run the cheap decisive test. The cheapest test is aimed squarely at the Sko19
failure mode: **is kappa identifiable/recoverable AT ALL?** If not, stop — no bio needed.

### Pilot 0 (A) — PLANTED SYNTHETIC MANIFOLD, sign recovery. ~hours on 1 GPU, maybe CPU.
**Purpose:** falsify/support "learned kappa recovers planted geometry," in a setting with NO model, NO bio,
NO confounds — the maximally favorable case. If it fails here it cannot possibly work on ESM3.
- **Data:** synthesize activations x = D_true * phi(t) + noise, for three planted blocks embedded in
  R^d (d~256) with random orthonormal D_true:
  - **circle** (kappa_true>0): t ~ U(0,2pi), phi = (cos t, sin t).
  - **line/interval** (kappa_true=0): t ~ U(0,1), phi = (t, 0) or a flat 2-patch.
  - **tree** (kappa_true<0): sample points from a synthetic branching/hierarchy (e.g. noisy BDP or a
    balanced tree embedded by Sarkar/hyperbolic embedding) — the honest hard case.
  - Mix all three blocks additively per sample + Gaussian noise at a few SNRs; this is also the
    block-sparse recovery test.
- **Fit:** the learnable-kappa block-sparse dictionary (G blocks, b=2 and b=3), multiple inits, >=5 seeds.
- **EXACT METRIC (pre-register):**
  - **Primary = sign-recovery accuracy:** fraction of seeds for which sign(kappa_learned) matches
    sign(kappa_true) on the block that aligns (by matched subspace / Hungarian) to each planted block.
  - **Secondary:** |kappa_learned - kappa_true| and its seed variance; and a **flat-baseline reconstruction
    gap** (does the curved block reconstruct the circle/tree at lower FVU than a matched flat block, at
    matched active real coordinates).
  - **Topology sanity:** persistent homology Betti of the recovered circle block = (b0,b1)=(1,1).
- **GO gate:** sign recovery >= ~90% across seeds for circle AND tree AND flat, with seed variance of kappa
  small (pre-register, e.g. sign agreement in >=9/10 seeds), AND the curved block beats the flat block on
  its matched planted manifold at matched rate. Also: kappa answer must be **insensitive to the R_kappa
  regularizer weight mu** over a reasonable range.
- **NO-GO:** if, as in `Sko19`, kappa drifts to ~0 for the tree (no negative-sign recovery) or is
  seed-unstable or regularizer-determined => **the pitch's headline is not deliverable; stop or pivot to
  fixed-geometry blocks (plan_task1) where you *assign* the geometry rather than *discover* it.**
  This single cheap experiment is the whole go/no-go for PITCH 1's novelty claim.

### Pilot 0 (B) — ONE ESM3-open test, sequence-only. ~tens of GPU-h. Only if (A) passes.
**Purpose:** does the recovered sign track real structural-variable geometry on a real model.
- **Model/data:** ESM3-open, one mid layer, **sequence-only input** (structure track OFF — else dihedrals
  are trivially present and the claim is void), few-thousand-protein subset, cache activations once.
- **Matched variables:** (periodic) **helix phase** = residue index mod 3.6 within DSSP H segments, or
  omega (~180 deg); (flat) **relative SASA** (monotone scalar). [Tree/hyperbolic fold-hierarchy is a
  STRETCH: it is a database annotation the model likely trained on => circularity; demote to appendix,
  consistent with plan_task1.]
- **Fit:** learnable-kappa block-sparse dictionary; read kappa on the blocks that best decode helix-phase
  vs SASA.
- **EXACT METRIC (pre-register the DOUBLE DISSOCIATION):**
  - The block that best captures **helix phase** learns **kappa>0** (and its angle has
    circular-circular correlation with true phase above a pre-set threshold, with Betti b1=1);
  - the block that best captures **SASA** learns **kappa ~= 0** (CI overlaps 0);
  - **sign is an INTERACTION, not a main effect:** kappa>0 must NOT also appear on the SASA block. If
    everything learns kappa>0, that is normalization acting as a generic regularizer = confound = fail.
  - **Positional-confound guard (mandatory, first):** a local sequence-index window baseline must NOT
    predict helix phase as well as the block code; else the "circle" is a positional artifact (cf.
    "Spokes, not a Ring").
- **GO gate:** double dissociation holds (periodic->kappa>0, SASA->kappa~0) at matched rate, trained-minus-
  random-init gap passes, positional baseline does not explain phase. **NO-GO:** kappa does not separate the
  periodic from the monotone variable => curvature is cosmetic on the real model.

**Why this ordering is decisive and cheap:** (A) isolates the *identifiability* question (the Sko19 risk)
with zero model cost; (B) isolates the *geometry-matches-biology* question. Either can kill the pitch for
<1 GPU-week total. Do NOT build the full bio sweep before (A) passes.

---

## 4. NOVELTY + KILLERS

### 4.1 Is this ICML-grade, or "mixed-curvature VAE with sparsity"?
**Honestly: as a METHOD, it is close to "mixed-curvature VAE (`Sko19`) + block sparsity (BSF/`SASA`)."**
The learnable-signed-kappa machinery is not new; the sparsity is not new; combining them is incremental on
the method axis and a reviewer who knows `Sko19`/`Gu18` will say exactly that. It is **NOT** ICML-grade as
"a new geometric SAE."

It **can** be a strong, distinct contribution if reframed away from the method and onto a **result**:

**Single strongest framing (use this, drop the rest):**
> *"Does a foundation model's own representation tell us the intrinsic curvature of a concept? A
> learnable-curvature sparse dictionary, applied to a biological model's activations, recovers the KNOWN
> geometry of structural variables unsupervised — positive curvature for periodic backbone phase, flat for
> monotone solvent exposure — as a geometry x variable DOUBLE DISSOCIATION, where the ground-truth geometry
> is mathematically exact (a circle/torus from coordinates), not a database label."*

That is publishable because (a) it is the **first curved/curvature-learning sparse dictionary on a
biological model** (open lane, 1.6), (b) it uses **exact geometric ground truth** that LLM work lacks, and
(c) the **double dissociation + unsupervised sign recovery against exact ground truth** is a result nobody
has shown — `Sko19` *tried the mechanism and could not even recover sign on synthetic tree data*, so
cleanly recovering sign on a real model's features would be a genuine advance. The contribution is
**"geometry is recoverable and matches known biology,"** not "we invented learnable kappa."

### 4.2 The two things most likely to KILL it
1. **kappa is unidentifiable from a bounded code (the `Sko19` failure).** For a bounded block reconstructing
   a finite activation patch, curvature trades off against decoder-norm and gain; many (kappa, D, r) give
   the same reconstruction, so kappa drifts, is seed-unstable, or is set by the regularizer rather than the
   data. `Sko19` observed exactly this ("badly conditioned"; tree-data curvature drifted to ~0, no sign
   recovery). **If Pilot 0(A) reproduces this, the headline figure cannot exist.** This is the #1 risk and
   the reason Pilot 0(A) is the whole ballgame. Mitigation: pre-register sign-recovery on planted data,
   regularizer-insensitivity, seed stability; if it fails, pivot to **assigned** (fixed) geometry per block
   (plan_task1) and abandon "discovery."
2. **Curvature does nothing without a hierarchy/entailment loss (the hyperbolic-side tautology).** On the
   kappa<0 side, block SELECTION reduces to Euclidean-norm TopK and a tangent-space decode is linear in
   Euclidean tangent coords, so **a Poincare block is just a bounded Euclidean code and curvature is inert**
   unless you add a hierarchy/entailment objective — but adding that makes kappa<0 a near-tautology and
   turns the "unsupervised discovery" claim into a supervised one (circularity; also the exact HyperSAE
   design, so you'd be re-doing HyperSAE's trick on bio). So the hyperbolic/tree half is caught in a vise:
   inert without a hierarchy loss, tautological/scooped with one. **Mitigation:** make the **periodic
   (kappa>0) side the load-bearing result** (there curvature is genuinely active: wrap-around/angle is
   real and a flat code structurally cannot close the loop), and demote the tree/hyperbolic side to a
   conditional appendix — the same conclusion plan_task1 reached for fixed geometries.

### 4.3 Scoop exposure (ranked)
1. **HyperSAE (UNVERIFIED)** — owns "hyperbolic SAE on LLM activations." Not learnable-signed-kappa, not
   bio, but removes your right to say "first curved SAE." **A human must verify its two papers before any
   writeup.** MED risk.
2. **`Sko19`/`Gu18`/Switch Spaces** — own "learnable mixed/signed curvature" and "sparse gating over curved
   components." Not a scoop of the bio result, but they make the *method* look derivative; you must cite
   them up front and position on the result. HIGH certainty they'll be raised.
3. **Goodfire+Arc (BSF) / Penn State (`SASA`)** — own block-sparse dictionaries; "curved block beats flat
   block" is adjacent to their turf. MED.
4. **`Zah26` (Geometric Wall)** — owns "curvature predicts SAE scaling"; if someone adds a curved-code
   remedy, that's your pitch. MED, rising.
5. **`Set26`** — nearest bio prior art (geometric annotation of protein-SAE latents); differentiate on
   *geometry discovery/recovery*, not post-hoc labels. LOW-MED.

### 4.4 Relation to the sibling pitches (so PITCH 1 isn't redundant)
- PITCH 1 (learn kappa) vs plan_task1 / fixed-geometry blocks (assign geometry): these are **competitors,
  not complements.** Pilot 0(A) decides which: if kappa is recoverable, PITCH 1's "discovery" framing is
  the stronger, more surprising paper; if not, fall back to **assigned** geometry (you lose the "model
  tells you the geometry" headline but keep a working method). **Run Pilot 0(A) before choosing.**
- The wrap-around causal dose-response (PITCH 2) is the independent validator that makes the kappa>0 side
  non-cosmetic; keep it as the companion test regardless.

---

## 5. ONE-LINE ANSWERS TO THE USER'S 4 TASKS
1. **Prior art:** mechanism ALREADY DONE (`Gu18`,`Sko19` learnable signed curvature; Riemannian sparse
   coding old); the SAE-on-activations combination OPEN; bio OPEN; `Gro26` orthogonal (not a scoop);
   HyperSAE PARTIAL+UNVERIFIED. **Headline ("kappa recovers geometry unsupervised") was attempted in
   `Sko19` and did not cleanly work — the strongest concrete caution.**
2. **Method:** kappa-stereographic block, learnable scalar kappa per block, gain-based sparsity gate,
   recon + weak-kappa-reg loss, series-expansion at kappa~0 + float64 boundary handling; cached activations
   only; ~1–1.5x a plain SAE.
3. **Test:** Pilot 0(A) planted circle/line/tree sign-recovery (go/no-go on identifiability) FIRST, then
   0(B) ESM3 helix-phase vs SASA double dissociation; metric = kappa sign-recovery accuracy + double
   dissociation + Betti/circular-corr, with positional-baseline and random-init guards.
4. **Novelty/killers:** reframe as "first curvature-learning sparse dictionary on a biological model
   recovering exact structural-variable geometry unsupervised (double dissociation)," NOT a new SAE method;
   killers = (1) kappa unidentifiable from a bounded code (the `Sko19` risk), (2) curvature inert without a
   hierarchy loss / tautological with one; scoop = HyperSAE (verify) + Sko19/Gu18 (cite, position on result).

### Key references (workspace cite keys)
Mechanism (learnable signed curvature): `Gu18`, `Sko19`, Switch Spaces (arXiv 2102.08688, UNVERIFIED).
Riemannian sparse coding (fixed manifold): `Xie12`, `Har13`, `Har15`, `Li13`, `Xue21`.
Curvature-as-signal / orthogonal: `Gro26` (Text Has Curvature).
SAE geometry / manifolds (flat): `Bha26`, `Mod25`, `Zah26`, `Par24c`, `Muc25`, `Fra26`, `Li24i`, `Hin25b`.
Non-Euclidean SAE on LLMs: HyperSAE (PyPI, UNVERIFIED).
Bio SAEs (all Euclidean): `Set26`, `Ada25`, `Guj25`, `Par25`, `Liu26`, `Ste26`, `Hak25`.

Benign-scope note: all interventions restricted to benign structural variables (dihedrals, phase, SASA,
fold geometry). No genomic/ligand capability steering.
