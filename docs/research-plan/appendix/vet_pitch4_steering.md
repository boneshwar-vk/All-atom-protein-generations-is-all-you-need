# Vet — PITCH 4: Causal-efficacy map on ESM3-open (geometry-matched (phi,psi) steering)

Senior reviewer, blunt. Date 2026-11-20. Planning/literature only — no code, no repo changes, no
operational design guidance. Benign/defensive framing (benign structural variable = backbone dihedral
geometry; NO binder/affinity/design pipeline; see §4 ethics).

Verification tags: CONFIRMED-PDF (read full text this session via Undermind), CONFIRMED-REC (resolved to real
record/abstract), CONFIRMED-WEB (web source this session), PRIOR (carried from this session's
plan_task3_steering.md / review_proteins.md, already verified), UNVERIFIED.

--------------------------------------------------------------------------------
## 0. BOTTOM LINE (read first)
--------------------------------------------------------------------------------
**OPEN — but narrow, incremental, and flanked by two papers that forecast how it dies.** The exact conjunction
(ESM3-open layer-wise *causal-efficacy map* + geometry-matched **(phi,psi) torus** steering of structure tokens,
with decodable != steerable as the headline) is **not done**. But every ingredient is occupied, and one
sub-claim as written is **already contradicted**:

- **The premise "ESM3 avoids wash-out because it is masked, not diffusion" is HALF-FALSE.** Masked iterative
  decoding has its **own** persistence/commitment pathology, and it is **already demonstrated** on a masked
  pLM: [Hin26] (CONFIRMED-PDF) shows activation edits on ESM-2 do **not** survive the
  activation→token-sample→re-embed cycle (foldability collapses once >~15% of positions are rewritten), and
  [Zha26g] (CONFIRMED-PDF) shows strong steering of masked ESM-2 drives activations **off-manifold toward
  random-amino-acid statistics**. Neither uses ESM3 or structure tokens, so neither *kills* the pitch — but
  together they say: **run the persistence/validity gate FIRST; it is the most likely cause of death.**
- **The sub-claim "scalar SAE directions do NOT steer structure tokens" is likely FALSE.** [Ste26] (Steenwyk,
  CONFIRMED-PDF) already steers ESM3 along a **scalar** averaged SAE structure-feature direction at layer 33 and
  it **works** (dose r=0.87, 4% of AA tokens flip, random/shuffled/orthogonal controls, causal confirmation via
  attribution patching + sparse feature circuits + L0H7 head ablation). So the geometry claim cannot be
  "scalar fails, curved works"; it must be the **quantitative** "curved-block beats scalar at matched norm on
  *realized, validity-filtered* (phi,psi) change."
- **The genuinely novel atoms (all OPEN, MED-HIGH confidence):** (i) a **layer-resolved** decodable-vs-steerable
  *efficacy-ratio* map on ESM3 (nobody has swept layers for steerability; Ste26 steers at one layer only);
  (ii) steering along a **geometry-matched (phi,psi) torus coordinate with wrap-around**, curved-block vs
  scalar-SAE vs matched-random (no protein-representation-steering paper uses circular/toroidal geometry);
  (iii) a **persistence check under iterative-unmasking commitment** as a first-class result.
- **DECISION-RULE call: RUN THE SMALL TEST.** No paper shows this specific thing done, and no paper shows it
  concretely does-not-work for ESM3 structure tokens. Two papers show *related* masked-pLM steering failing,
  which is exactly why a cheap decisive test (persistence gate first) is the right move rather than a proof.
  **Verdict: viable workshop paper IF (a) the persistence gate clears and (b) curved beats scalar; otherwise it
  degrades to a calibration/negative result (still publishable, less exciting).** Sell it as an
  **interpretability finding**, never as a steering tool.

--------------------------------------------------------------------------------
## 1. PRIOR-ART KILL CHECK (Task 1) — ALREADY DONE / PARTIAL / OPEN
--------------------------------------------------------------------------------
Workspace 33899782-4f3e-48f1-9e4a-a3b6ac8a7af7. Undermind account bs20b012@smail.iitm.ac.in.

### A. "Off-Manifold Collapse in Guided Protein Language Models" [Zha26g] — does it kill/pre-empt?
**CONFIRMED-PDF. Verdict: does NOT kill the pitch; PARTIALLY pre-empts the validity-filter component; reframes
the premise.** arXiv 2608.18597, 2026-08-19. (Earlier docs guessed "Zha26g" as the caution cite — correct.)
- **Models:** ESM-2 650M **only**, under iterative masked decoding. **No ESM3, no structure tokens, no phi/psi,
  no secondary-structure target.**
- **Steering:** (i) mean-difference contrastive steering vector, norm-rescaled, added to the residual stream
  (all layers in main runs; single-layer diagnostic at layer 17); (ii) predictor-gradient guidance at layer 17.
  **Not a geometry-matched direction.**
- **The collapse signature:** strong guidance contracts activations **inward** toward the per-channel natural
  mean. A **diagonal Mahalanobis statistic** (standardize each of D=1280 channels by UniRef50 stats, sum of
  squares, average over positions) falls from ~1232 toward the **random-AA reference ~604** as pLDDT drops.
  The property oracle can still *reward* collapse (glycine homopolymer scored solubility 0.99). Remedy =
  **"Mahalanobis filtering,"** a training-free **post-hoc output selector** (keep candidates with
  Mahal^2 >= D - k*sqrt(2D), k=1 → threshold 1229); raises mean pLDDT at matched guidance, transfers across
  guidance methods. It is a **selector, not a projector** (cannot fix a bad candidate; targets inward collapse
  only, misses outward/composition failures).
- **What it does NOT do:** no ESM3, no geometry-matched (phi,psi) steering, no realized structure-token change,
  **no layer-wise causal-efficacy map**. Its filter scores activation *typicality*, not whether a candidate hit
  a geometric target.
- **Use it:** cite as the collapse-failure caution **and** adopt Mahalanobis typicality as one leg of the
  validity filter. Differentiate hard (the pitch's filter must also check *realized geometric target*, not just
  typicality). Confidence HIGH.

### B. SAE / feature steering of ESM / ESM3 structure tokens — ALREADY DONE (and it works)
- **[Ste26] "Structure tokens sharpen the feature vocabulary of pLMs"** (Steenwyk, bioRxiv 10.64898/
  2026.05.12.724593, 2026-05-14). **CONFIRMED-PDF. THE closest ESM3 prior art.** SAE on **ESM3 layer 33/48**
  (and ESM-2 layer 24/33); steering = normalized average of top-20 structure-enhanced SAE feature decoder
  directions added at layer 33 all positions, alpha in {±5…±50}, 500 proteins, **random/shuffled/orthogonal
  controls**; dose r=0.87, 4% token flips; causality confirmed by **attribution patching (layerwise)**, **sparse
  feature circuits** (ESM3 16→33, 33→42), and **L0H7 attention-head ablation** (zeroing one layer-0 head changes
  secondary-structure prediction at 40% of residues). **Does NOT:** compare decodability vs steerability across
  layers (steers at 33 only); use phi/psi/torus/wrap-around geometry; apply any validity filter
  (pLDDT/clash/refold) to steered outputs; steer the structure-token track itself. **This is the paper to beat
  and the reason the "scalar can't steer structure" claim is dead.** Confidence HIGH.
- **[Gar25]** ESM-2 8M SAE steering to zinc-finger (arXiv 2502.09135) — ALREADY DONE for ESM-2. HIGH.
- **[Par25]** ESM2-3B Matryoshka SAE, steers **ESMFold** to raise solvent accessibility at fixed sequence
  (arXiv 2503.08764) — ALREADY DONE for the folding model. HIGH (CONFIRMED-WEB/REC).
- **[Hua25]** "Steering Protein Language Models" (arXiv 2509.07983) — activation steering + editing-site ID on
  auto-encoding and autoregressive pLMs (lysozyme). General PLM steering DONE. HIGH.
- **[Sim24] InterPLM**, **[Ada25]**, **[Guj25]**, **[Tsu25]** — SAE feature steering of ESM-2 generation / low-N
  design. DONE for ESM-2. HIGH (PRIOR).
- **[Zho26h] "Unlocking Multimodal pLMs at Inference Time"** (arXiv 2608.25855, 2026-08-26). **CONFIRMED-PDF.**
  Studies **ESM3**, DPLM-2, DPLM-2.1. Inference-time control via **vanilla sampling, classifier-free guidance
  (logit-level), reward-guided beam search (trajectory/pTM)** — **NOT activation/residual-stream steering, NO
  layer analysis, NO phi/psi.** Establishes that ESM3 structure-token generation is controllable at
  logit/sampling level. **Must-cite, must-differentiate** (the pitch is residual-stream + layer-resolved +
  geometry, which this does not touch), and its CFG/beam are the **non-interpretability baselines to beat** on
  realized structural change. HIGH.
- **Scoop-watch:** "Extracting and Steering Ensemble and Structural Properties in BioEmu" (NeurIPS 2026) —
  SAEs on **BioEmu and ESM3** for structural/thermodynamic properties, **steering framed as future work.**
  CONFIRMED-WEB (title/abstract only). Closest live scoop on ESM3 SAE structural steering; check its camera-ready.

### C. Decodable-vs-steerable in protein models — ALREADY DONE off-ESM3; OPEN on ESM3
- **[Jed26b]** real title **"Probing and steering biology across Boltz-1's trunk-diffusion boundary"** (arXiv
  2608.11475). **CONFIRMED-PDF/PRIOR. The protocol antecedent and the "which layer/boundary" version of the
  claim.** Decodable != causal on Boltz-1 single rep *s*: β-strand F1 0.82 but steering −0.001 [CI
  −0.003,+0.001] (null) while helix +0.039 / coil +0.061, matched-norm random controls, dose curves. The pitch
  **extends this to ESM3 (masked generative, not diffusion), to structure tokens, and to a geometry-matched
  direction** — a real but incremental extension. HIGH.
- **[Xun26] "Lagged Coupling: Internal Representations Become Readable Before They Become Causal"** (arXiv
  2609.01048). **CONFIRMED-REC/PRIOR. The framing twin.** Probe AUROC ≥0.99 early while steering sits in the
  matched-random null band (43/48 cells). LLM, not bio, no timestep/geometry, no site-selection rule. The
  pitch's "causal-efficacy ratio = realized change / decodability" is **this made quantitative on ESM3**. Must
  cite + differentiate hard — nearest conceptual scoop. HIGH.
- **On ESM3 specifically: OPEN.** No paper maps decodability vs steerability across ESM3 layers. Ste26 steers at
  one layer; Zho26h has no layer axis; Whi26 (below) has decodability-by-layer but only partial interventions.
  MED-HIGH.

### D. Layer-wise causal-efficacy / "where to intervene" — PARTIAL
- **[Whi26] "High-resolution dissection of concept acquisition…"** (bioRxiv 10.64898/2026.07.20.739599).
  CONFIRMED-REC. Layer-by-layer concept **decodability** on ESM2/AMPLIFY (physicochemical/motifs early,
  secondary structure mid, domains middle) + molecular-biology interventions (phosphomimic). **Where concepts
  are readable by layer is done; a steerability/efficacy-ratio axis is not.** MED-HIGH.
- **[Nai25] "motif-gated domain recognition…"** (bioRxiv 10.1101/2025.08.22.671739). CONFIRMED-PDF. **Causal
  activation patching in SAE latent space** on ESM-2; minimal circuit for contact prediction (0.022% of
  latent-token pairs), early motif → mid/late domain gating, path ablations. First circuits-style causal pLM
  analysis. Causal "where" is done for contact prediction; **not a steerability-vs-decodability efficacy ratio,
  not generation, not ESM3.** HIGH.
- General SAE causal-validation methodology to borrow: **[Bal26b]** "Geometric Recovery to Causal Validation /
  Causal Inertness," **[Doa26]** "Causal Activation Steering via Sparse Mediation," **[Guo26]** "What Do SAE
  Feature Discovery Scores Identify? From Activations to Interventions," **[Rie26b]** "Steering grids for SAE
  features," **[May24]** "Can SAEs decompose steering vectors," **[Jia26c]** "Refit the Probe: Single-Direction
  Ablation Is Not a Necessity Test." CONFIRMED-REC. These sharpen controls; none is on ESM3 or (phi,psi).

### E. Geometry-matched (phi,psi) torus steering, curved vs scalar — OPEN
- **No paper recovers the (phi,psi) toroidal/Ramachandran geometry in a protein model's *representation* space
  and steers along the circular coordinate with wrap-around.** Searches returned only *structural-biology*
  torus/Ramachandran papers (FoldDoF [Zhu26g], internal-coordinate density [Art23], etc.), not
  representation-geometry steering. The concept "multidimensional/subspace steering > scalar" and "circular
  variables are rings in hidden states" exist in **LLM** land (manifold steering; Engels circular day/month),
  so the idea is **not conceptually new** — but the **(phi,psi)-torus-on-ESM3-structure-tokens instantiation is
  open.** This is the atom shared with flagship Pitches 1-3 (dihedral=circle, (phi,psi)=torus). MED-HIGH
  (caveat: workspace corpus may lag 2026-Q4; BioEmu/ESM3 SAE work is moving — UNVERIFIED that no Nov-2026
  preprint scoops it).

**NET:** ALREADY DONE = {scalar SAE steering of ESM3 structure features (Ste26); decodable≠steerable on bio
diffusion (Jed26b) and LLMs (Xun26); logit/sampling control of ESM3 structure tokens (Zho26h); masked-pLM
steering off-manifold collapse (Zha26g) and decode-persistence failure (Hin26)}. OPEN = {the layer-resolved
*efficacy-ratio* map on ESM3; the geometry-matched (phi,psi) torus direction with wrap-around; the persistence
check under iterative-unmasking commitment as a first-class result}. **The pitch is a synthesis whose only sellable
core is (OPEN atoms) conditional on surviving (ALREADY-DONE failure modes).**

--------------------------------------------------------------------------------
## 2. CONFIRM THE PREMISE (Task 2): does ESM3 escape wash-out? — HALF-FALSE
--------------------------------------------------------------------------------
**Claim in the pitch:** ESM3 is masked-generative, so you steer and re-run the forward pass — no denoising
trajectory, no wash-out.

**Honest critique:**
- **True part:** ESM3 has no continuous reverse-diffusion trajectory that re-projects toward the data manifold
  at *every* step. A single steered forward pass is read out directly; you are not fighting a denoiser that
  undoes your edit over 50–1000 steps. So the *specific* diffusion wash-out of [Jed26b]/image-diffusion does not
  apply in the same mechanical form. Good instinct, partially.
- **False/naive part — ESM3 has its OWN commitment/re-projection problem, and it is already observed on masked
  pLMs:**
  1. **Iterative unmasking is still a trajectory.** ESM3 generation unmasks tokens over multiple steps.
     Committing a structure token (categorical sample / argmax) is a **discrete, irreversible projection** onto
     the codebook; committed tokens then condition later steps. A mid-network nudge that is not *committed* at a
     masked position in the step where it is applied leaves **no trace** — the discrete bottleneck is the
     masked-model analog of re-projection.
  2. **[Hin26] already demonstrates the masked-pLM persistence failure (CONFIRMED-PDF).** On ESM-2, an
     activation edit (DiffMean / SAE clamp) is corrected on-manifold *in isolation* (one GLP pass recovers >90%
     of excess loss), **but does not survive the activation→token-sample→re-embed cycle**: after decoding, edited
     sequences are no more on-manifold than the naive arm, and **foldability (pTM) collapses once rewriting
     coverage exceeds ~15%** (their 30-round, 10%-mask schedule touches ~96% of positions → pTM 0.37–0.65 →
     0.23–0.32). Helix clamp is the only clear structural success; β-strand/coiled-coil/zinc-finger fail. This
     is the **ESM-equivalent of wash-out**, driven by *decode coverage*, not a denoiser.
  3. **[Zha26g] adds the second pathology (CONFIRMED-PDF):** strong steering of masked ESM-2 → **off-manifold
     collapse** to random-AA activation statistics, with the oracle still scoring "success." So "stronger
     steering realizes more (phi,psi) change" risks being a **validity artifact.**
- **Conclusion:** The premise cannot be asserted; it must be **tested**. ESM3 may be *better* than diffusion on
  persistence (no denoiser), but the discrete-commitment + decode-coverage + off-manifold-collapse triad is a
  live risk that two papers already realize on masked pLMs.

**Persistence check for ESM3 (design — the analog of the diffusion persistence test, Task 2 deliverable):**
- **Single-pass realized change (upper bound):** steer at layer ℓ, re-run ONE forward pass, decode structure
  tokens at the targeted masked positions, map tokens→backbone→(phi,psi). Measure Δ(phi,psi) vs matched-random.
- **Commitment-survival:** apply the *same* nudge but then let the **full iterative unmasking schedule** run to
  a fully-decoded structure (two variants: steer-once-then-free-decode; steer-at-every-step). Recompute final
  (phi,psi) from the completed structure (ESM3 structure tokens → coordinates, or re-fold the decoded sequence).
  **Persistence ratio = final realized Δ / single-pass Δ.** Wash-out analog ⇔ ratio → 0.
- **Coverage sweep (import Hin26's axis):** vary fraction of positions steered/rewritten; locate the coverage at
  which foldability (pLDDT/pTM) breaks. Report Δ(phi,psi) and validity *jointly* vs coverage.
- **Collapse monitor (import Zha26g):** track the diagonal Mahalanobis typicality of steered activations vs the
  random-AA reference across the alpha sweep; flag the alpha where "effect" becomes collapse.
- **G0 gate:** a layer-ℓ nudge must produce a final-structure Δ(phi,psi) that clears the matched-random band
  (|z|>2), monotone in alpha, **at typicality above the collapse threshold.** If null everywhere → commitment
  erases edits → pivot to logit-level/single-step framing (and the pitch becomes a negative/calibration result).

--------------------------------------------------------------------------------
## 3. SMALL DECISIVE TEST (Task 3) — 1 GPU / days
--------------------------------------------------------------------------------
**Model:** **ESM3-open (esm3-open, 1.4B)** — the model [Ste26] used; structure tokens are VQ-VAE codes over
backbone geometry, so (phi,psi) is recoverable from decoded tokens. Fits on a single 48–80 GB GPU. (If weights
access is a problem, ESM-2+ESMFold is the fallback, but then you lose the "masked generative, no wash-out"
premise and collide with Par25/Hin26 — keep ESM3.)

**Target variable (benign):** per-residue backbone dihedral **(phi,psi)** encoded on the torus as
(sinφ,cosφ,sinψ,cosψ), and/or the discrete **structure token**. Model-independent, geometric, continuous,
circular — exactly the flagship thesis's known-geometry asset. Secondary readout: secondary-structure fraction.

**Protocol (cheapest version):**
1. **Cache + decodability map.** N≈100–300 benign, non-hazard monomers (length 50–300; exclude toxins/viral
   fusion/immune-evasion/select-agent classes, §4). Cache residual-stream activations at all 48 layers (or a
   coarse 8–12-layer grid first). Fit a **circular ridge probe** act(ℓ)→(sinφ,cosφ,sinψ,cosψ) per residue;
   report held-out R²/angular error by layer → **(ℓ)_dec = argmax**. (Probes are free.)
2. **Steerability map.** At each ℓ, inject three matched-norm directions and re-decode:
   (i) **geometry-matched curved-block direction** — a 2–4-D subspace aligned to the (sinφ,cosφ,sinψ,cosψ) probe
   / torus tangent, moved along the **circular coordinate with wrap-around** (the headline direction, reusing
   Pitch 1 machinery); (ii) **scalar SAE feature direction** — a Ste26-style structure-enhanced SAE feature or a
   single probe direction; (iii) **matched-norm random**. Decode structure tokens → backbone → recompute
   (phi,psi). Realized Δ = decoded − unsteered, **wrap-aware** (circular distance). → **(ℓ)_steer = argmax**.
3. **Validity filter (mandatory, 3 legs):** (a) **Mahalanobis typicality** [Zha26g]; (b) **pLDDT/pTM** of the
   decoded/re-folded structure; (c) **clash check + ProteinMPNN→refold "designability."** Report realized Δ
   **only for candidates that pass**; collapse-driven "effects" are discarded.
4. **Persistence gate (§2, run FIRST on the single best cell).** Single-pass Δ vs full-decode Δ; coverage sweep.

**Core result (the paper):** a 2-axis map showing **(ℓ)_dec ≠ (ℓ)_steer** (decodable != steerable on a masked
generative protein model), with **curved-block > scalar-SAE > matched-random** in *validity-filtered, wrap-aware*
realized Δ(phi,psi) at matched norm.

**Go/no-go gates:**
- **G0 (persistence) — run first.** Final-structure Δ(phi,psi) clears matched-random (|z|>2), monotone in alpha,
  above collapse threshold. **Null everywhere → STOP/pivot** (commitment erases edits; reframe as negative
  result + the Zho26h logit-level baseline wins). *Most likely failure (Hin26/Zha26g forecast it).*
- **G1 (the finding).** (ℓ)_dec ≠ (ℓ)_steer, decodability-argmax significantly weaker as a steering site than
  the causal-argmax. **This is the paper.** If they coincide → downgrade to "probe-and-steer" benchmark note.
- **G2 (geometry).** curved-block > scalar-SAE at matched norm on validity-filtered realized Δ. **If equal →
  the geometry headline collapses** (Ste26 already shows scalar works) → fall back to G1 as the sole claim.
- **G3 (validity).** The realized Δ is achieved by **on-manifold, foldable** structures, not collapse; report
  the fraction surviving the filter and the alpha at which collapse begins.

**Compute:** single GPU, days. ESM3-open 1.4B inference only; N=100–300 × {8–12 layers} × {3 directions} ×
{4–5 alpha} × {2–3 seeds} forward passes + decode + a ProteinMPNN/ESMFold validity pass. Probes free. Pilot
clearly fits "1 GPU / days." Decisive: G0+G1+G2 either produce the headline or kill it cheaply.

--------------------------------------------------------------------------------
## 4. NOVELTY + ETHICS (Task 4)
--------------------------------------------------------------------------------
### Genuinely new vs incremental?
**Incremental, but genuine — if framed as interpretability, not steering.** The *causal-efficacy ratio* is
[Xun26]'s readable-before-causal and [Jed26b]'s decodable≠causal made quantitative and ported to a **masked
generative** protein model; geometry-matched circular steering is a known LLM idea (manifold/subspace steering,
circular features) in a **new (phi,psi)-torus-on-structure-tokens** instantiation. None of {Ste26, Zho26h,
Jed26b, Zha26g, Hin26, Par25, Gar25} does the **layer-resolved efficacy-ratio map**, the **torus/wrap-around
direction**, or the **iterative-commitment persistence check**. That triad is the contribution. It is a
workshop-scale synthesis finding, not a new mechanism — be honest about that.

### Strongest framing (raises odds, lowers hazard)
"**Readable != steerable in a masked generative protein model: a layer-resolved causal-efficacy map of backbone
(phi,psi) geometry in ESM3, showing the most-decodable layer is not the most-steerable, that geometry-matched
(torus, wrap-around) directions are more causally effective than scalar SAE directions at matched norm, and that
realized control must survive iterative-unmasking commitment — which erases naive edits and drives off-manifold
collapse.**" Lead with the interpretability finding + the persistence/collapse *calibration* result; the
validity filter (Mahalanobis typicality ⊕ fold-check ⊕ refold) is a headline control, not a footnote. Venue: a
mech-interp / ML-for-structural-biology workshop first.

### Two killers
1. **Commitment/persistence wash-out (G0).** Iterative unmasking + discrete token commitment + decode coverage
   erase the mid-network edit before it reaches the final decoded structure. **Already realized on masked pLMs**
   by [Hin26] (correction dies in the decode→re-embed cycle; foldability breaks >~15% coverage). If ESM3
   behaves the same, there is no "steer-and-re-run beats diffusion" story. *Most likely cause of death; gate it
   first.*
2. **Scalar already works / geometry gives no lift (G2).** [Ste26] shows scalar SAE-direction steering of ESM3
   structure features works (r=0.87). If the curved-block direction does not beat scalar on validity-filtered
   realized (phi,psi) change, the geometry headline collapses to "steer-and-measure" — a re-skin of Ste26 ⊕
   Jed26b. *(Secondary killer: off-manifold collapse — [Zha26g] — any alpha large enough to move (phi,psi) may
   push activations to random-AA statistics, so the "effect" is a validity artifact and the filter eats it.)*

### Dual-use note (benign/defensive)
- **Benign variable:** backbone **dihedral geometry** (phi,psi) / secondary-structure composition — a geometric
  descriptor, **not** affinity, interface, or binder optimization. **No binder-design pipeline**; do not steer
  toward interface/affinity; do not release a structure-optimization tool tuned for binding.
- **Data hygiene:** benign, open monomers only; **exclude** toxins, viral entry/fusion, immune-evasion, and
  select-agent-adjacent classes.
- **Posture:** frame the deliverable as an **interpretability + calibration/negative finding** (where structural
  control is causally encoded, and why naive steering fails), whose value is **defensive** — auditing/monitoring
  generative protein models and detecting off-manifold collapse — explicitly over any generative use. Adopt the
  biosecurity-statement precedent (VFUSE [Yu26b], and Zha26g's own collapse-detection framing). The mechanistic
  knowledge "where a structural property is causally encoded" is inherently dual-use; argue the auditing value
  and withhold operational steering recipes.

--------------------------------------------------------------------------------
## 5. Must-cite / must-differentiate ledger
--------------------------------------------------------------------------------
Closest ESM3 prior art (beat it): **[Ste26]** (scalar SAE steering of ESM3 structure features, layer 33).
Framing twin: **[Xun26]** (readable before causal). Protocol antecedent: **[Jed26b]** (decodable≠steerable,
Boltz-1 trunk↔diffusion). Collapse caution + validity filter: **[Zha26g]**. Masked-pLM persistence failure
(premise test): **[Hin26]**. ESM3 inference-time baselines (logit/sampling/beam — beat on realized structural
change): **[Zho26h]**. Layer-wise decodability / causal "where": **[Whi26]**, **[Nai25]**. Other pLM/ESM
steering: **[Gar25]**, **[Par25]**, **[Hua25]**, **[Sim24]**, **[Ada25]**, **[Guj25]**, **[Tsu25]**. SAE
causal-validation methodology (controls): **[Bal26b]**, **[Doa26]**, **[Guo26]**, **[Rie26b]**, **[May24]**,
**[Jia26c]**. Scoop-watch (verify camera-ready): "Extracting and Steering Ensemble and Structural Properties in
BioEmu" (NeurIPS 2026, SAE on BioEmu+ESM3, steering = future work).

**Open UNVERIFIEDs to close before committing:** (a) confirm no 2026-Q4/Nov preprint already does an ESM3
layer-wise steerability map or (phi,psi)-geometry steering (corpus may lag; check the BioEmu+ESM3 NeurIPS
camera-ready and arXiv 2026-10/11); (b) confirm ESM3-open weight access for structure-token decode; (c) verify
ESM3 structure-token→backbone decode gives usable per-residue (phi,psi) at the N/length regime chosen.
