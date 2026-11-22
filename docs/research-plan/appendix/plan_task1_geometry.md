# Task 1 — Critical research plan: alternative within-block GEOMETRIES for block-sparse dictionaries on biological foundation models

Reviewer stance: blunt critic, not cheerleader. Research/planning only; no repo changes.
Date: 2026-11-19. Prior art verified via Undermind (workspace 33899782-4f3e-48f1-9e4a-a3b6ac8a7af7) + WebSearch.
"UNVERIFIED" = not confirmed from a primary source this pass. arXiv IDs without a confirming hit are flagged.

---

## 0. ONE-PARAGRAPH VERDICT (read this first)

The idea is **partly novel and partly already taken.** "Geometry-aware sparse coding" is a 10+ year-old
field (Riemannian/Grassmann/SPD dictionary learning), hyperbolic-for-biological-hierarchy is a solved
*embedding* problem, circular/toroidal features in neural-net activations were already found and
published on LLMs (Engels et al.), the Euclidean block-subspace dictionary is now **doubly** owned
(Goodfire BSF **and** an independent NeurIPS 2026 reinvention, SASA), and a **hyperbolic SAE on LLM
activations already appears to exist** (HyperSAE, unverified). What is **genuinely unoccupied** is narrow
and specific: (a) *any* non-Euclidean sparse dictionary on a **biological** model, and (b) a
**matched-geometry double-dissociation** design that tests each geometry against the **exact** biological
ground-truth manifold it should fit (torus for backbone dihedrals; circle for phase; hyperbolic for fold
taxonomy) **and a mismatched control**, at capacity/rate-matched budget, with the circular **wrap-around
causal dose-response** as the signature. The strongest, least-scoopable version is **toroidal/circular
blocks on ESM3 dihedrals/phase**. **Hyperbolic should be a stretch goal, not a pillar** — highest
implementation cost, highest scoop risk, weakest biological match, and most exposed to the "it's just a
bounded Euclidean code + a regulariser" objection. The naive "does a sphere beat a cube overall" framing
is wrong and will be rejected; the science is the geometry×variable **interaction**, not a main effect.

---

## 1. PRIOR ART (verified vs genuinely new)

### 1.1 What is ALREADY DONE (and threatens novelty)

**A. Circular / multi-dimensional features in NN activations — ALREADY PUBLISHED (LLMs).**
- **Engels et al., "Not All Language Model Features Are One-Dimensionally Linear"** (ICLR 2024; workspace
  `Eng24`). Days-of-week / months-of-year represented as **circles**, discovered by clustering SAE
  features; introduces the irreducible-multi-dimensional-feature test. **This owns "circular manifold
  feature found via sparse coding."** Your circular claim is not novel *as a phenomenon*; it is novel only
  *on a bio model with exact geometric ground truth and a code whose geometry is the circle itself.*
- `Li24i` "The Geometry of Concepts: SAE Feature Structure" (Entropy 2024); `Mod25` "The Origins of
  Representation Manifolds in LLMs" (2025); `Fra26` "SMIXAE: Unsupervised Manifold Discovery in LLMs"
  (2026) — all establish that activation manifolds + SAE-manifold interaction is an active, crowded LLM topic.
- **`Gup26b` "Spokes, not a Ring: Revisiting the Weekday Manifold in Language Models" (2026)** — a direct
  **cautionary counter-result**: the claimed weekday "circle" is actually spokes, not a smooth ring. This
  is the single most important paper for your *critique* section: a claimed circle can be a sampling/topology
  artifact. You must pre-empt it (see §4 Betti pre-registration, §6 killer #2).

**B. Euclidean block / subspace sparse dictionaries — DOUBLY OWNED.**
- `Fel26c` BSF (Goodfire, arXiv 2606.25234) with Vanilla / **Grassmannian** / Group-Lasso variants;
  `Jer26` critique (2608.27515, Tournament Top-K); `Bha26` "Do SAEs Capture Concept Manifolds?" (2604.28119).
- **NEW/CONFIRMED THIS PASS — `SASA` "Subspace-Aware Sparse Autoencoders" (Dalili & Mahdavi, Penn State,
  arXiv 2606.06333, NeurIPS 2026 poster).** Independently reinvents the block idea: learned decoder
  **subspaces**, Top-s group gating, **nuclear-norm rank regulariser**; argues a scalar SAE needs
  exponentially many directions to cover a d_i≥2 feature. **Implication: the Euclidean-block-beats-scalar
  headline is now a two-group race and is closing fast. Do not make that your headline.**
- Note: BSF's **Grassmannian** variant already constrains the block's decoder basis to an **orthonormal
  frame**. A reviewer will immediately ask how your "sphere code" differs. Answer (pre-empt in the paper):
  Grassmannian constrains the *decoder basis* `D_g` and leaves the code `z_g∈R^b` **flat**; a spherical
  code constrains `z_g∈S^{b-1}`. They are **orthogonal design axes** and compose (orthonormal frame +
  spherical code = the cleanest angular feature). This distinction is load-bearing for your novelty.

**C. Hyperbolic SAE — APPARENTLY ALREADY EXISTS on LLMs (scoop risk, UNVERIFIED).**
- **HyperSAE** (PyPI `hypersae`): "High-Performance Hyperbolic Sparse Autoencoders for Mechanistic
  Interpretability." Self-lists two papers — *"Escaping Flatland: Weight-Space Regularization and
  Hyperbolic Geometry in Mechanistic Interpretability"* (theory) and *"Hyperbolic Sparse Autoencoders:
  Empirical Validation of Poincaré Manifold Geometry on LLM Activations"* (empirical). Design per the
  page: **Euclidean forward pass + hyperbolic weight optimisation + a Poincaré entailment loss.**
  **UNVERIFIED** — no arXiv ID, authors, or date confirmable from here; PyPI self-description is the only
  source. **Treat as a live scoop of the hyperbolic-SAE idea itself (on LLMs, not bio).** Also adjacent:
  `HELM` hyperbolic LLM (arXiv 2505.24722); "Hesitation Has a Geometry" hyperbolic steering probe
  (arXiv 2610.02391). Your hyperbolic differentiator can only be **bio + hierarchical biological ground
  truth**, and even that is thin (see B/E).

**D. Riemannian / manifold-constrained sparse coding — a MATURE pre-interpretability field.**
- `Che15` Riemannian dict learning for SPD matrices; `Har13`/`Har14` Grassmann dict learning;
  `Xie13`/`Xie12` nonlinear dict learning on manifolds; `Cha18`/`Cha16` dict learning on statistical
  manifolds; `Har12` SPD kernel sparse coding; `Xue21` hierarchical Riemannian pursuit. **"Constrain
  dictionary atoms/codes to a Riemannian manifold" is not novel in general.** Novelty must come from the
  *interpretability framing + bio model + matched biological ground truth*, and you must cite this lineage
  or a methods reviewer will call the method derivative.

**E. Hyperbolic embedding of biological hierarchies — a SOLVED EMBEDDING problem (not interpretability).**
- `Nic17` Poincaré embeddings (Nickel & Kiela); `Mat19b` Poincaré VAE; `Gan18` entailment cones;
  `Kli19` Poincaré maps for single-cell hierarchies (Nature Comms); `Yu20` hierarchical **drug** embedding
  in hyperbolic space; `Jia22`/`Wil21` phylogenetic trees in hyperbolic space; `Ali25` hyperboloid
  **biological sequence** analysis; `Pog23`/`Pog25` hyperbolic biological-interaction/tissue networks;
  `Sko19`/`Gu18` mixed-curvature VAE / product-space embeddings. **These embed data *de novo* into curved
  space. None interpret the *internal activations of a pretrained biological foundation model.*** That is
  your only open lane on the curved side — and it is narrow.

**F. Normalised / directional SAE scoring — adjacent.**
- `Nai26` "Cosine-Scored Sparse Autoencoders" (directional scoring); `Qui25` "Binary Sparse Coding for
  Interpretability"; `Muc25` hierarchical-semantics SAE (Euclidean); `Jed26d` stable/steerable SAEs
  (weight reg, Crook group — a controls antecedent). Cosine scoring is **sphere-adjacent for the scalar
  case** and must be cited as the nearest "directional SAE."

**G. Protein/bio SAE lineage — ALL EUCLIDEAN SCALAR.**
- `Sim24` InterPLM, `Ada25` AlQuraishi, `Guj25` Gujral/Berger (PNAS), `Par25` Matryoshka-on-ESMFold,
  `Gar25` steering, `Set26` geometric **annotations** of ESM-2 SAE features (closest prior art — labels
  scalar latents post-hoc, does NOT test manifold recovery), `Mig26` PairSAE, `Liu26` ESMFold trunk,
  `Ste26` structure-token vocabulary. **Confirmed: no non-Euclidean / geometric sparse dictionary exists
  on any biological model.** This is the real gap.

### 1.2 What is GENUINELY NEW (the defensible atoms)

1. **Any non-Euclidean sparse dictionary on a biological foundation model.** HIGH confidence open.
2. **Matched-geometry double-dissociation against EXACT biological ground-truth manifolds** (torus =
   (φ,ψ) dihedrals is a *mathematically exact* torus, not a database annotation), with a **mismatched
   control** per geometry, at **capacity/rate-matched** budget, random-init + seed controls. HIGH.
3. **The circular/toroidal wrap-around causal dose-response** on a bio model's **own output head** — a
   test a scalar or flat-R^b code structurally cannot pass. HIGH (inherited from the main project; the
   geometry twist is that the *code itself* lives on the circle).
4. (Weaker) Hyperbolic block tied to a **tree-like biological variable actually present in the model** —
   open only if HyperSAE doesn't generalise first AND the representation is genuinely tree-like (§2, §5).

**Net:** the project is novel as a *bio-model + matched-geometry-to-ground-truth* study, NOT as a
"new geometric SAE method." Frame it as the former or it reads as incremental.

---

## 2. MATCHING GEOMETRY TO BIOLOGY (with mismatched controls)

**Design principle (the critique of the naive framing):** "Does a sphere beat a cube?" is the **wrong
question** and will be desk-rejected by a geometry-literate reviewer. The correct claim is a
**geometry × variable interaction**: each geometry should win **only on its matched variable** and tie/lose
on mismatched ones. The experimental signature is a **diagonal win pattern** (a double/multiple
dissociation), not a main effect of geometry. If spherical blocks beat Euclidean on *everything*
(including a monotone scalar like SASA), that is evidence of a **confound** — normalisation acting as a
generic regulariser — **not** of geometric matching, and it *weakens* the paper.

| Geometry (block) | b / effective dim | MATCHED biological variable (true manifold) | MISMATCHED control (should NOT win) | Notes |
|---|---|---|---|---|
| Scalar (SAE, b=1) | ℝ¹ | monotone scalars: relative **SASA**, contact number, hydrophobicity; GC content (DNA) | **dihedral / phase** (periodic) — scalar must fragment | the incumbent baseline |
| Euclidean block (BSF/SASA, b=2–3) | ℝᵇ (flat) | a small **non-periodic** linear patch of property space | a **pure circle** — flat block can't close the loop; endpoints don't meet | your other baseline |
| **Circle S¹ (b=2)** | periodic 1-D | **helix phase** (res index mod 3.6 in DSSP helices), **ω** (≈180°), **codon phase** (ℤ/3 → circle), **~10.5 bp nucleosome phasing** (DNA) | monotone **SASA** — circle wastes/forces spurious wrap | cleanest cheap win |
| **Torus T²=S¹×S¹ (b=4)** | 2-periodic | **(φ,ψ) backbone dihedral pair = EXACT Ramachandran torus** | **fold class** (tree) — torus can't embed hierarchy | **strongest matched case in the whole study** |
| Sphere S² (b=3) | unit 3-vector | backbone **orientation** / Cα pseudo-vector direction; contact direction | periodic 1-D phase — S² over-parameterises | needs a genuine directional variable |
| **Poincaré ball Bᵇ (curv −c)** | tree / hierarchy | **CATH/SCOP fold taxonomy**, **GO** hierarchy, **organism taxonomy/phylogeny** of source, **EC** number | **dihedral/phase** (periodic) — hyperbolic can't close a loop | STRETCH; database-annotation variables → circularity risk |

**Biological-matching caveats (blunt):**
- The torus case is the jewel because (φ,ψ) is an *exact* manifold computed from coordinates, not a
  label the model trained on → it dodges the annotation-circularity trap the Orlov review flags.
- The hyperbolic matched variables (fold class, GO, taxonomy) are **database annotations the model very
  likely trained on** → maximal circularity exposure, exactly what `Ali26`/`Che26`/Orlov warn about. This
  is a second reason to demote hyperbolic.
- **Positional-confound warning (inherited, critical):** helix phase and codon phase may be **trivially
  positional** — the model could read them off sequence index, making any "circle" an index artifact
  (cf. `Gup26b` spokes). The matched-variable pilot MUST run a **local-sequence-window baseline first**; if
  a 1-hot position window predicts the phase as well as the block code, the geometric claim is dead.
- **Input modality:** use **sequence-only** input for ESM3 dihedral/phase experiments. Structure-track
  input makes dihedrals trivially present and invalidates the "model internally represents geometry" claim.

---

## 3. METHOD SPEC (minimal math per dictionary)

All variants: decode into the Euclidean activation space, train on **cached activations only** (no grad
through the big model), block-sparse across G blocks. Baseline BSF: code `z_g∈ℝᵇ`,
`x̂ = Σ_g D_g z_g`, keep the `l0` blocks with largest `‖z_g‖` (block-TopK); `b=1` ⇒ TopK SAE.

### 3.1 Hypersphere S^{b-1} code (circle b=2, sphere b=3)
**Construction — magnitude/direction split (the substantive, non-cosmetic version):**
```
v_g = W_enc,g x                     (raw block pre-activation, ℝᵇ)
u_g = v_g / (‖v_g‖ + ε)             (direction on S^{b-1})
r_g = softplus(w_gᵀx + β_g) ≥ 0     (separate scalar gain = "how strongly present")
x̂   = Σ_{g∈A} r_g · D_g u_g         (A = active set)
```
- **Sparsity is on the gain** `r_g` (block-TopK on `r_g`), **not** on code norm (‖direction‖≡1).
- **Retraction/projection:** `u = v/‖v‖` (projection). For Riemannian steps, exp-map on the sphere:
  `exp_u(ξ) = cos‖ξ‖·u + sin‖ξ‖·ξ/‖ξ‖`. Projected-gradient (renormalise after each step) is simpler and
  usually sufficient; use `geoopt` only if projected-grad underfits.
- **Loss:** `Σ ‖x − x̂‖² + λ·(block-sparsity on r_g)`; optionally add an **angular-uniformity** term so
  directions spread on the sphere (discourages collapse to a cap).
- **Why it's not "BSF with a different norm":** (i) the *code* is intrinsically angular, (ii) the readout,
  interpolation and **causal traversal are geodesic + periodic** (wrap-around), (iii) comparison is
  rate-matched. If you only normalise and still read/traverse a flat decoder, it IS cosmetic — do not do that.
- **Numerical pitfalls:** `v_g→0` ⇒ `u_g` undefined/unstable → gate blocks with small `r_g` off *before*
  normalising; keep `ε~1e-6`. Dead-direction collapse → periodic re-init + uniformity term.
- **Compute:** identical to BSF (+O(b) per block). Cached activations only. Essentially free.

### 3.2 Torus T^k = (S¹)^k (b = 2k; k=2 for (φ,ψ))
Split the block into k pairs, normalise each pair to its own circle; one shared gain (or per-circle gain):
```
for j=1..k:  (c_j,s_j) = v_g^{(j)} / ‖v_g^{(j)}‖     # θ_j = atan2(s_j,c_j)
x̂ = Σ_g r_g · D_g · [c_1,s_1,…,c_k,s_k]
```
- Retraction = per-pair normalisation. This is the **product-of-circles** manifold = the true Ramachandran
  geometry. **Highest-value, lowest-ambiguity variant.** Same cost/stability profile as §3.1.

### 3.3 Hyperbolic Poincaré ball Bᵇ (curvature −c)
**Tangent-space-decode formulation (clean, and matches HyperSAE's "Euclidean forward pass"):**
```
encode:  z_g = exp_0^c(W_enc,g x) = tanh(√c‖v‖)·v/(√c‖v‖)          # into the ball, ‖z_g‖<1
decode:  x̂ = Σ_g D_g · log_0^c(z_g),  log_0^c(z)=(1/√c)·artanh(√c‖z‖)·z/‖z‖
sparsity: block-TopK on d_hyp(0,z_g)=(2/√c)artanh(√c‖z_g‖)  ≡ TopK on ‖z_g‖ (monotone)
```
- **THE HYPERBOLIC TRAP (state this plainly in the paper):** block *selection* reduces to Euclidean-norm
  TopK, and a tangent-space decode is **linear in Euclidean tangent coordinates**. So **without a genuine
  hyperbolic objective, a Poincaré code is just a bounded Euclidean code and the curvature does nothing.**
  To make curvature load-bearing you MUST add a **hierarchy/entailment term**: e.g. a (weakly) supervised
  Poincaré distance / entailment-cone loss (`Gan18`) that places parent concepts near the origin and
  children near the boundary, OR an unsupervised tree-likeness reward (low Gromov δ-hyperbolicity). This
  makes the method partly **Direction-A (supervised)** and raises circularity — a real cost.
- **Numerical pitfalls (serious):** near the boundary `‖z‖→1`, `artanh` and its gradient **explode**.
  Standard fixes: project to `‖z‖≤1−10⁻⁵`; clip code norms; compute `artanh` in **float64**; Mathieu-style
  feature clipping; Riemannian optimiser (RSGD/RAdam, `geoopt`) or projected gradient. Budget real
  engineering time here.
- **Compute:** ~1.5–3× an SAE of equal width (exp/log maps + Riemannian opt), still on cached activations.
  **Order of magnitude unchanged** vs plain SAE; caching + interventions dominate.

### 3.4 What is shared / what differs in cost
- **All three need only cached activations.** The dominant cost (activation caching + causal interventions
  through ESM3/AlphaGenome) is **geometry-independent**. Geometry choice does **not** change the compute
  order of magnitude. Sphere/torus ≈ free over BSF; hyperbolic adds a constant factor and real debugging.

---

## 4. EVALUATION ("recovers the manifold better" — against ground truth, fairly)

**Recovery metrics (matched to each geometry):**
- **Circular–circular correlation** (Jammalamadaka–Sarma) between recovered block angle and true angle;
  **toroidal correlation** for T². For hyperbolic, **distortion / mean-average-precision of the hierarchy**
  (Nickel-style) and Gromov δ.
- **Geodesic R²:** variance of the true coordinate explained by **geodesic regression** from the block code
  on the *matched manifold metric* (circular/toroidal/spherical/hyperbolic regression, not linear).
- **Topology / Betti pre-registration (anti-`Gup26b` guard):** persistent homology of the recovered block
  point cloud; **pre-commit** expected Betti numbers — circle (b0,b1,b2)=(1,1,0); torus (1,2,1);
  sphere (1,0,1); tree/hyperbolic b1=0 + low δ. **Require (i) a match to the true manifold's Betti numbers
  AND (ii) a trained − random-init gap** (threshold pre-registered, e.g. circular-R² gap > 0.1). A "ring"
  that is actually spokes fails the b1/persistence test — report it honestly if so.

**Fair capacity comparison across different effective dimension (the hard, decisive part):**
- **Do NOT compare at matched units.** A sphere b=2 code (1 angular dof + 1 gain) carries fewer effective
  bits than a free ℝ³ block. Report **all** of: (i) matched **parameters** (G·d·b); (ii) matched **active
  real coordinates/token** (l0·b); (iii) matched **reconstruction FVU** (compare interpretability at equal
  FVU); and the parameterisation-free gold standard:
- **Rate–distortion / MDL frontier:** quantise codes, compute description length (bits), plot
  **reconstruction (or circular-R²) vs bits**. A geometry **wins only if it dominates the frontier on its
  MATCHED variable** and does NOT dominate on the mismatched control. This is the honest test and the one a
  reviewer will demand (enforces H3 at matched **bits**, per the earlier red-team).

**Controls (non-negotiable, from the earlier review):**
- **Random-init dictionary + random-init model** (Heap–Aitchison trap).
- **≥5 seeds + Tournament Top-K** (`Jer26`) for seed stability; report seed variance of recovered geometry.
- **Capacity/rate-matched** scalar SAE and Euclidean BSF/SASA alongside every geometry.
- **Local-sequence-window baseline FIRST** (positional confound) — if it explains the phase, stop.
- **Leakage/holdout** on proteins/sequences; structure-free input for dihedral experiments.

**Causal (the crown jewel):**
- **Geodesic traversal** of the recovered coordinate; for circle/torus the **wrap-around dose-response**
  (output returns to baseline at 2π) — a scalar/flat code **structurally cannot** produce it.
- **Readout independence:** measure effect on the **model's OWN output head** (masked-AA / track logits),
  NOT a same-variable probe.
- **Matched-norm random-direction controls**; must clear the **`Jed26b` "decodable ≠ causal" bar**
  (their strand-steering null is the protocol antecedent to beat/cite).

---

## 5. STAGING + FEASIBILITY + COMPUTE

**Pilot 0 — cheapest decisive test (go/no-go, ~24 GB GPU).**
- ESM3-open, **one layer**, **sequence-only**, few-thousand-protein subset, cache activations **once**.
- Fit: scalar SAE (b=1) · Euclidean BSF (b=2, Tournament Top-K) · **circular BSF S¹ (b=2)**.
- Matched variable = **helix phase** or **ω** (clean periodic 1-D); mismatched control = **relative SASA**.
- **GO iff** (double dissociation): circular BSF beats scalar+Euclidean on circular-R²/Betti for the
  **periodic** variable **at matched rate**, does **NOT** beat them on **SASA**, trained−random gap passes,
  and the **local-window baseline** does not explain the phase. **NO-GO ⇒ the geometry is cosmetic; stop or
  pivot to the Datasets&Benchmarks fallback.** Cost: ~tens of GPU-h (one geometry, one layer, one var pair).

**Gate 1 — torus + causal.** Add **T² for (φ,ψ) Ramachandran** (strongest matched case) + the **wrap-around
causal dose-response** on ESM3's output head with matched-norm random controls. GO iff the output follows
the toroidal coordinate and returns at 2π where a flat code cannot. ~100–250 GPU-h.

**Gate 2 — hyperbolic (CONDITIONAL, stretch).** Proceed **only if**: (a) a hierarchical variable
(CATH/SCOP fold class OR source-organism taxonomy) is **linearly decodable** from the layer at all (probe
first — if the hierarchy isn't present, hyperbolic is moot); **and** (b) the activation subspace for that
variable is **actually tree-like** (low Gromov δ). If not tree-like, **hyperbolic BSF cannot win — drop
it.** Even on GO, expect the circularity objection (annotation variable) and the HyperSAE scoop. ~200–400
GPU-h incl. debugging boundary instability.

**Gate 3 — genomic generality + full sweep.** One genomic panel: **AlphaGenome v1 codon phase (S¹ / ℤ3)**
(off Goodfire's turf; train your own baseline) or **Evo 2** for the cheap pilot (public SAE baseline).
Full ESM3 sweep: all variants × ≥5 seeds × Tournament Top-K × rate-matched frontier. ~800–1500 GPU-h on
1–2×80 GB; ~4–8k GPU-h for full seed ensembles on a small cluster. (Orders of magnitude; UNVERIFIED.)

**Honest verdict on hyperbolic:** **not worth co-equal investment.** Highest implementation cost
(Riemannian opt, boundary blow-ups), highest scoop risk (HyperSAE claims it on LLMs), weakest biological
match for the load-bearing *geometry* story (dihedrals are the clean win; fold hierarchy is a
trained-on annotation → circularity), and most exposed to "bounded-Euclidean-code + regulariser."
**Make it one appendix panel conditional on Gate 2, not a pillar.** The circle/torus is the publishable core.

---

## 6. FINAL VERDICT

**Novel and publishable as a distinct contribution?** — **Qualified yes, if narrowed.** NOT as a "new
geometric SAE method" (Riemannian dict learning is old; Euclidean block is doubly owned by BSF+SASA;
circular LLM features are Engels's; a hyperbolic SAE on LLMs apparently already exists). **Yes** as:
*the first non-Euclidean sparse dictionary on a biological foundation model, tested by a matched-geometry
double dissociation against exact biological ground-truth manifolds, at a rate-matched frontier with
random-init/seed controls and a wrap-around causal dose-response.* That specific atom is unoccupied.

**Is it "just BSF with a different norm"?**
- **Sphere/torus on a periodic variable: NO** — *provided* (i) the win is a geometry×variable **interaction**,
  not a main effect; (ii) readout/traversal are intrinsically angular with the **wrap-around causal test**;
  (iii) it is **rate/bits-matched**. Drop any one and it collapses to "a different norm."
- **Hyperbolic: at serious risk of YES** (a bounded Euclidean code + a regulariser) unless a genuine
  hierarchy/entailment objective is used AND the representation is actually tree-like — and HyperSAE may
  already own it.

**Single strongest version:** *"Toroidal/circular block-sparse dictionaries recover the Ramachandran (φ,ψ)
torus and helix/codon phase in ESM3-open that scalar SAEs fragment and flat BSF splits — shown at a
capacity/rate-matched rate–distortion frontier with random-init, ≥5-seed/Tournament-Top-K, and
local-window controls, and confirmed by a wrap-around causal dose-response on the model's own output head;
one genomic panel (codon phase, S¹) for generality."* **Sphere/torus only; hyperbolic demoted to a
conditional appendix.**

**Two things most likely to sink it:**
1. **The geometry is cosmetic.** No double dissociation (the sphere also "wins" on SASA), or the win
   **vanishes at matched rate/bits** → the contribution is "BSF with a different norm," and BSF-vs-SAE is
   itself the likely-negative H3. Mitigation: rate–distortion frontier + the interaction design are the
   whole paper; pre-register the dissociation.
2. **The periodicity is a positional/dataset artifact** (`Gup26b` "spokes not a ring" + the helix/codon
   positional confound): the recovered "circle" is an index artifact the model reads off position, not a
   learned geometric feature — killing both existence and causal claims. Mitigation: local-window baseline
   **first**, Betti/persistence pre-registration with trained−random gap, sequence-only input.

**Scoop exposure (ranked):**
1. **Goodfire+Arc** — owns BSF **and** the Evo 2 SAE; maximal on any "block/geometry beats SAE on bio."
2. **Penn State (SASA, NeurIPS 2026)** — the Euclidean-subspace lane is already closing; stay off it.
3. **HyperSAE (unverified)** — live scoop on the hyperbolic-SAE idea (LLMs); only "bio+hierarchy" is left,
   and it's weak. A human should verify HyperSAE's papers before committing to the hyperbolic half.
4. **Engels/Tegmark** — owns circular multi-dimensional LLM features; cite as the phenomenon antecedent.
5. **Setlur** (geometric annotation of protein SAE features) — nearest bio prior art; differentiate on
   *manifold recovery + matched code geometry*, not post-hoc scalar labels.
The defensible, un-scooped atom is specifically **matched-code-geometry ↔ exact-biological-ground-truth-
manifold on a bio model, with the double dissociation + wrap-around causal test.** Narrow to that or be scooped.

**Benign-scope note:** all interventions restricted to benign structural variables (dihedrals, phase,
SASA, fold geometry). No genomic/ligand capability steering.

---

### Key references (workspace cite keys)
Circular/manifold LLM features: `Eng24`, `Li24i`, `Mod25`, `Fra26`, **`Gup26b` (critique)**.
Euclidean block: `Fel26c` (BSF), **SASA arXiv 2606.06333**, `Jer26`, `Bha26`, `Muc25`.
Hyperbolic SAE (scoop, UNVERIFIED): HyperSAE / "Escaping Flatland" / "Hyperbolic SAEs … LLM Activations".
Riemannian dict learning: `Che15`, `Har13`, `Har14`, `Xie13`, `Cha18`, `Xue21`.
Hyperbolic bio embedding: `Nic17`, `Mat19b`, `Gan18`, `Kli19`, `Yu20`, `Jia22`, `Wil21`, `Ali25`, `Pog23`.
Mixed-curvature: `Gu18`, `Sko19`. Directional SAE: `Nai26`, `Qui25`, `Jed26d`.
Bio SAE (all Euclidean): `Sim24`, `Ada25`, `Guj25`, `Par25`, `Set26`, `Mig26`, `Liu26`, `Ste26`, `Gar25`.
Causal bar / controls: `Jed26b`, `Ali26`, `Che26`, `Orl26`, `Sha26`.
