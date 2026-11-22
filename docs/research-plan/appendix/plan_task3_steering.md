# Task 3 — Critical research plan: interpretability-guided localization + intermediate-timestep steering of a structure-diffusion model

Senior-reviewer plan + blunt critique. Written 2026-11-19. Planning/literature only — no code, no repo
changes, no operational design guidance. Benign/defensive framing throughout (see §8 Ethics).

Verification status tags: CONFIRMED-WEB (primary web source this session), CONFIRMED-PDF (read full text this
session), CONFIRMED-REC (resolved to real paper record), PRIOR (carried from this session's FINAL_FINDINGS /
review_proteins, already verified), UNVERIFIED.

--------------------------------------------------------------------------------
## 0. BOTTOM LINE (read this first)
--------------------------------------------------------------------------------
- **"ipSAE" is NOT a sparse autoencoder.** It is the Dunbrack lab's **interface confidence score** for
  AlphaFold/Boltz-style predictions — a PAE-derived scalar that replaces **ipTM** for judging protein–protein /
  protein–ligand **interface quality**. (CONFIRMED-WEB.) In this project it is a candidate **TARGET PROPERTY**
  to localize/steer toward, not a method. It is also a **trap** as the primary target (it is the model's own
  self-confidence → Goodhart/circularity; see §4).
- **Verdict on the idea: RESHAPE toward one causal finding; it is at high risk of being a re-skin.** "Steer at
  an intermediate denoising step instead of at the end" is **already established** in image diffusion
  (limited-interval guidance [Kyn24], selective guidance [Gol23], critical windows [Li24h], commitment-timing
  [Khi26b]). "Use interpretability/SAE to pick where in a folding model to intervene" is **already done**
  (FoldSAE on RFdiffusion [Zar25]; decodable≠causal on Boltz-1 [Jed26b]). The conjunction is thin.
- **The one defensible novel atom:** a **2-D (layer × denoising-timestep) decodable-vs-steerable map** for a
  **continuous interface property** on an open co-folding/structure-diffusion model, showing that the
  most-**decodable** (layer,step) cell is **not** the most-**steerable** cell, and that choosing the steering
  site by a **causal** criterion beats (a) choosing it by probe accuracy, (b) steering at the output, and
  (c) a matched-norm random direction — i.e. **extending Jedryszek's "decodable≠steerable" from "which layer"
  to "which (layer, denoising-step)", and turning it into a usable site-selection rule.**
- **Target for the first study:** NOT ipSAE/ipTM and NOT experimental affinity. Use a **geometric,
  model-independent, continuous** interface property computable by code from the generated structure —
  **interface buried surface area (ΔSASA/BSA)** or **interface contact count** — and use ipSAE/ipTM only as a
  *secondary* readout to expose Goodhart.

--------------------------------------------------------------------------------
## 1. ipSAE — clarification (Task 1)
--------------------------------------------------------------------------------
**What it is.** ipSAE = a confidence/quality score for the **interface** of a predicted complex, from
**Roland Dunbrack's lab**. Preprint: *"Rēs ipSAE loquuntur: What's wrong with AlphaFold's ipTM score and how to
fix it"*, bioRxiv **10.1101/2025.02.10.637595** (2025). Code: **github.com/dunbracklab/IPSAE**; PyPI `ipsae`.
A 2026 follow-on, *"Calibrating AlphaFold confidence scores for interface quality"*, **arXiv 2610.17108**,
positions ipSAE as a **PAE-based interface score** that is less sensitive than ipTM to long non-interacting or
poorly-predicted regions. (All CONFIRMED-WEB.)

**Mechanics (for design, not operational use).** It is computed from AlphaFold/Boltz **PAE** (predicted aligned
error) restricted to inter-chain residue pairs, with a PAE cutoff; it is **directional**, so the AFDB benchmark
reports `ipSAEmax = max{ipSAE(A→B), ipSAE(B→A)}`. Variants include a `d0chn` form. The acronym is expanded
inconsistently across sources ("interaction prediction Score from Aligned Errors" in the preprint; "interprotein
Score from Aligned Errors" on a community skills page) — a minor flag, not a substantive issue.

**Why the name collides with "SAE".** Pure coincidence of letters. ipSAE's "SAE" = *Score from Aligned Errors*;
it has nothing to do with *sparse autoencoders*. The user's instinct was correct; treat any mention of "ipSAE"
in this project strictly as the **interface-quality target**, and say so explicitly in the paper to avoid
reader confusion (a reviewer in this subfield will otherwise assume SAE=autoencoder).

**How it serves the project.** ipSAE is a **scalar interface-quality readout** of a generated/predicted
complex. It could be the property you localize ("where in the network is interface quality represented?") and
steer toward ("raise interface quality mid-generation"). **BUT** ipSAE is a transform of the model's own PAE
head — so for a Boltz/AF3-family model, steering toward ipSAE is steering toward the model's **own
self-assessment**, which (i) can be inflated without the structure actually improving (Goodhart), and (ii)
repeats PairSAE's exact failure mode (PairSAE "predicts affinity" but really predicts **Boltz-2's own predicted
affinity**, test R²=0.528, [Mig26]/PRIOR). **Recommendation:** demote ipSAE/ipTM to a *secondary/held-out*
readout used to demonstrate the Goodhart gap, and make the *primary* target a geometric property the model
does not emit (§4).

--------------------------------------------------------------------------------
## 2. Decompose the method into its two real parts (Task 2)
--------------------------------------------------------------------------------
The idea is one sentence but two separable research problems. Plan and gate each; the **binding contract**
between them is the decodable≠steerable caveat.

### Part A — LOCALISATION ("where does the target property live?")
Candidate localizers, from cheapest/weakest to strongest:
- **A1 Linear probing per site.** Train a linear/ridge probe `activation(layer ℓ, timestep t) → property y`.
  Produce a **decodability heatmap over (ℓ, t)**. Cheap, but **decodability ≠ usability** — this is exactly the
  quantity Jed26b and Xun26 warn against trusting.
- **A2 Attribution/activation patching.** Patch activations from a high-property run into a low-property run at
  (ℓ, t) and measure the realized shift in the final property. This is a **causal** localizer and is the right
  gold standard for "where to steer"; it is more faithful than a probe but ~10–50× more forward passes.
- **A3 SAE / BSF feature selection.** Train a dictionary (TopK SAE baseline; BSF if the property is
  continuous/periodic) on cached activations at promising (ℓ, t); select features by probe coefficient
  ([Zar25]-style) or by causal effect. Adds a discovery/auditing layer and interpretable directions, at the
  cost of a second model to train and seed-stability concerns ([Jer26]/PRIOR).
- **A4 Jacobian/transport-style readout.** *De-emphasize.* On a diffusion/flow model the "read an intermediate
  state out through a token unembedding" construction does not apply (no token head); the useful remnant is a
  linearized "where is this state heading" map, which is heavy and ill-posed here. Keep out of the main plan
  (consistent with FINAL_FINDINGS H5 "drop the Jacobian half").

**Design rule (the whole point):** localisation must be reported on **both** axes — a probe/decodability map
**and** an intervention/steerability map — and the headline is their **disagreement**, not the probe map alone.

### Part B — EARLY/INTERMEDIATE STEERING ("intervene there during generation, measure realized change")
- **B1 Where to inject.** At a chosen (ℓ, t): add `α·d` to the residual/track activation, where `d` is the
  unit probe direction (or SAE feature direction), `α` swept. Diffusion/flow-specific: the injection happens
  **inside one reverse step** and then the trajectory continues to denoise — so you must measure whether the
  change **persists to the final structure** or is **washed out** by later steps (critical failure mode, §5).
- **B2 Mandatory controls** (table-stakes, all from this session's prior art):
  - **matched-norm random-direction** control at the same (ℓ, t) ([Jed26b], [Xun26] `z_e` null band).
  - **readout independence**: the evaluation metric must **not** be the probe's own training target nor the
    model's own head — use an independent geometric recomputation (§4), else the "effect" is circular.
  - **output-only baseline**: steer only at the last step(s) / or guide the output — to show intermediate
    beats output (the actual claim).
  - **strong external baseline**: **Feynman–Kac particle steering** [Har25c] at **matched reward-evaluation
    budget** — this is the method to beat; if interpretability-guided single-site steering cannot match FK's
    0.32→0.60-type gains at equal compute, the contribution is academic, not practical (be honest about this).
  - **validity filter**: steric-clash check + ProteinMPNN→refold "designability", so a realized ΔBSA is not
    achieved by garbage/clashing structures.
- **B3 What "success" means.** A **monotone dose–response** of the realized property in `α`, that **persists to
  the final sample**, **exceeds the matched-random band**, and is **larger at the interpretability-chosen
  (ℓ,t) than at the most-decodable cell and at the output** — under readout independence.

--------------------------------------------------------------------------------
## 3. Prior art — what is already done (Task 3). Do not reinvent.
--------------------------------------------------------------------------------
**Steering at intermediate denoising steps is ESTABLISHED (images):**
- [Kyn24] *Applying Guidance in a Limited Interval Improves Sample and Distribution Quality* — 309 cites;
  guidance restricted to a **middle** noise interval beats whole-trajectory guidance. (CONFIRMED-REC.) This is
  the single biggest "you did not invent intermediate steering" citation.
- [Gol23] *Selective Guidance: Are All the Denoising Steps … Important?*; [Jin25] *Stage-wise Dynamics of CFG*;
  [Wan23d] *Diffusion Models Generate Images Like Painters (outline first, details later)*. (CONFIRMED-REC.)
- [Li24h] *Critical windows: non-asymptotic theory for feature emergence in diffusion*; [Khi26b] *When Does a
  Diffusion Model Decide What to Draw?* — localizes commitment **by noise level** via restart/committor
  analysis (coarse content settles ~σ≈2.4, exact class ~σ≈1.1); explicitly **tries and fails to pin the CIFAR
  guidance window** (CONFIRMED-PDF). ⇒ the *timestep-localization* idea exists; its clean instantiation on a
  **biological interface property** is open.

**Readable-before-causal is ESTABLISHED (language) — the conceptual twin of your thesis:**
- [Xun26] *Lagged Coupling: Internal Representations Become Readable Before They Become Causal* — Pythia/OLMo;
  probe AUROC ≥0.99 from step 1k while steering sits in the matched-random **null band in 43/48 cells**
  (CONFIRMED-PDF). **Not diffusion, not bio, not a timestep axis, and gives no site-selection rule** → your
  diffusion/bio/(layer×timestep) instantiation with a *usable* rule is the gap. **Must cite as the framing
  antecedent and differentiate hard — this is your nearest scoop.**
- [Jed26b] Boltz-1 trunk+diffusion: β-strand decodable (F1 0.82) but strand-steering ≈ −0.001 vs helix +0.039 /
  coil +0.061, matched-norm random controls, dose curves (CONFIRMED-PDF/PRIOR). **The protocol antecedent and
  the "which layer" version of your claim.** You extend it to "(layer × denoising-step)" and to a *continuous*
  target with a site-selection rule.
- [Shi26d] *When Does Activation Steering Change What a Model Computes From?*; [Bea26] *Toward universal steering
  and monitoring* (Science); [Wur26] *Manifold Steering* — general steering theory to position against.

**Guided protein/structure diffusion toward properties is ESTABLISHED (output-level):**
- [Har25c] *Controllable protein design with particle-based Feynman–Kac steering* — RFdiffusion, reward =
  favorable interface ΔG, resampling (not gradient), designability **0.32→0.60**, +>89.5% (CONFIRMED-PDF). The
  SOTA inference-time steerer and your must-beat baseline.
- [Jia24c] *General Binding Affinity Guidance for Diffusion Models in SBDD*; [Wan24k] *Force-Guided SE(3)
  Diffusion*; [Ric25]/[Lam26b] inference-time landscape / meta-energy biasing; [Vil23] guided antibody
  co-design; [Lai26b] force-field guidance; [Aya25] unified geometry-conditioned guidance. (CONFIRMED-REC.)
  **None of these choose the injection site by interpretability; they guide the output/structure proxy.**

**SAE-based steering of structure/bio models is ESTABLISHED:**
- [Zar25] **FoldSAE** — TopK SAE on **RFdiffusion block `main_04`** (chosen by an **ablation screen**), steers
  helix/strand via a **selected feature set**, **time-agnostic** probes, **no matched-random control**, does
  **not** localize the best timestep, does not isolate individual features (CONFIRMED-PDF). **The closest bio
  prior art; your differentiators: continuous interface target, explicit (layer×timestep) steerability map,
  matched-random + readout-independence controls, and a site-selection rule validated against FK.**
- [Mig26] PairSAE (Boltz-2 pair; predicts the model's **own** affinity; no causal/seed/leakage controls);
  [Yu26b] VFUSE (SAE on all-atom RFDiffusion3; biosecurity framing); [Gar25] ESM SAE steering; [Set26] geometric
  annotations of ESM-2 SAE latents; [Sim24] InterPLM; [Ada25]; [Guj25]; [Par25]; [Liu26] ESMFold-trunk SAEs.
  (PRIOR / CONFIRMED-REC.)

**Net:** every ingredient has a paper. The project is a **synthesis**, and must be sold as a **causal finding +
a rule**, not a new mechanism.

--------------------------------------------------------------------------------
## 4. Target property + ground truth (Task 4)
--------------------------------------------------------------------------------
**Three distinct "targets" that the user's brief conflates — keep them separate:**
1. **Model self-confidence** (ipSAE, ipTM, PAE): emitted by the model. Cheap, differentiable-ish, but a
   **Goodhart/circular** target — steering raises it almost by construction; PairSAE already shows a model's
   own predicted score is the thing that gets learned. Use only as a **secondary/held-out** readout.
2. **Geometric interface quality** (ΔSASA/buried surface area; inter-chain contact count ≤5 Å; interface shape
   complementarity Sc): **computed by external code** (FreeSASA/DSSP-style) from coordinates, **independent of
   any model head**, continuous, deterministic, cheap. **← Primary target for the first study.**
3. **Experimental binding affinity** (PDBBind, PLINDER, BindingDB): the real prize, but **do not use first.**
   Reasons: (a) severe **leakage / time-split** problems — most structure models trained on the PDB; naive
   random splits inflate everything; PLINDER's ligand/pocket-similarity splits exist precisely because of this.
   (b) affinity is a **different target** from confidence — a model can be confident about a wrong-affinity
   pose. (c) it drags in dual-use (binder optimization). Park it for a follow-up with strict time/similarity
   splits.

**Chosen first target: interface buried surface area (BSA/ΔSASA).** Justification: model-independent (clean
readout independence vs §2-B2), continuous (supports dose–response), cheap to compute at scale, well-defined for
any complex, and **benign** (a geometric descriptor, not an optimization objective for a real binder). Contact
count is an equally acceptable backup; Sc is a nice-to-have third.

**Ground-truth dataset.** A fixed panel of **~300–1000 open, benign protein–protein complexes** (e.g. a curated
slice of the PDB / an existing docking benchmark), **time/similarity-split** from any training data, deliberately
**excluding** hazard-adjacent classes (toxins, viral entry/fusion, immune-evasion, select-agent organisms; see
§8). Property values precomputed by external code. ipSAE/ipTM recorded alongside as the secondary readout.

--------------------------------------------------------------------------------
## 5. Minimal decisive experiment + go/no-go gates (Task 5)
--------------------------------------------------------------------------------
**Model (pick one, open weights):** **Boltz-1** — preferred, because [Jed26b] already released trunk+diffusion
SAEs, per-residue activations and intervention code, so the injection plumbing and the protocol antecedent are
in hand and directly citable; its diffusion module is the exact place the "intermediate-step" claim lives.
*Alternative:* **RFdiffusion** (reuse FoldSAE `main_04` plumbing + FK baseline), cleaner backbone-only geometry
but no pair/interface head (BSA still computable from coordinates). Do **not** use a non-diffusion model (ESM3,
AlphaGenome) — the intermediate-timestep claim requires a diffusion/flow sampler (make this switch explicit:
this task leaves the ESM3/AlphaGenome family).

**The smallest test that can kill or confirm the thesis:**
- **Caching.** For N≈200 complexes, run the sampler and cache activations at a grid of **L layers × T denoising
  timesteps** (e.g. 4–6 layers × 6–8 timesteps). (This per-timestep caching is what makes diffusion more
  expensive than a sequence model.)
- **Step 1 — Decodability map.** Ridge probe `act(ℓ,t) → BSA`; report R²/heatmap over (ℓ,t). Identify
  `(ℓ,t)_dec = argmax`.
- **Step 2 — Steerability map.** At each (ℓ,t), inject `α·d(ℓ,t)` (probe direction) across an α-sweep during
  sampling; recompute **final-structure** BSA; subtract the **matched-norm random-direction** band; identify
  `(ℓ,t)_steer = argmax realized ΔBSA`.
- **Step 3 — The rule.** Compare realized ΔBSA (at matched compute, with validity filter) across:
  (i) steer at `(ℓ,t)_steer`; (ii) steer at `(ℓ,t)_dec`; (iii) steer at the **output/last step**; (iv)
  matched-norm **random**; (v) **Feynman–Kac** output steering [Har25c] at matched reward-eval budget.
- **Step 4 — Goodhart control.** Does steering toward BSA also move the **held-out** readouts (ipSAE/ipTM, a
  physics/clash score, designability), or **only** BSA? Report both; a target-only move is the Goodhart warning.

**Go/no-go gates:**
- **G0 (persistence).** At the single best cell, a mid-trajectory injection produces a realized final-structure
  ΔBSA that **clears the matched-random band** (|z|>2) with a monotone α dose–response. **If null everywhere →
  STOP**: the trajectory washes out intermediate edits; pivot to output-only steering or to a non-diffusion
  framing. *(This is the most likely way the project dies — see §6.)*
- **G1 (the finding).** `(ℓ,t)_dec ≠ (ℓ,t)_steer`, i.e. decodability argmax ≠ steerability argmax, with the
  probe-chosen site significantly weaker than the causal-chosen site. **This is the paper.** If they coincide,
  the "interpretability-guided" angle collapses to "probe and steer" → downgrade to a benchmark note.
- **G2 (beats output).** Intermediate causal-chosen steering > output-step steering at matched compute.
- **G3 (useful).** Causal-chosen single-site steering is **competitive with FK** at matched reward-eval budget,
  OR cheaper for equal effect. If FK dominates at every budget, report honestly as "interpretability explains
  *where* the property lives, but output resampling is the practical tool" — still publishable, less exciting.

**Compute (order-of-magnitude, UNVERIFIED).** Dominated by per-timestep activation caching + the α×(ℓ,t)
intervention sweep (each cell = N full samplings). Pilot (N=200, 4×6 grid, 5 α, 3 seeds, Boltz-1 on 1–2×80 GB
GPUs): ~**300–700 GPU-h**. Full (N≈1000, 6×8 grid, random+FK baselines, seeds): ~**1.5–3k GPU-h**. Probes are
free; the sampler forward passes are the cost.

--------------------------------------------------------------------------------
## 6. Verdict, risks, scoop, ethics (Task 6)
--------------------------------------------------------------------------------
### Novel vs re-skin
**Borderline — reshape required.** As literally stated ("interpretability to pick a layer/timestep, then steer
mid-generation") it is a **re-skin**: [Kyn24]/[Gol23]/[Khi26b] own intermediate-step guidance, [Zar25] owns
SAE-localized steering of a folding diffusion model, [Jed26b] owns decodable≠steerable on a co-folding model,
[Xun26] owns readable-before-causal. **The one genuinely novel, defensible contribution is the (layer ×
denoising-timestep) decodable-vs-steerable MAP for a continuous interface property on an open structure-diffusion
model, plus a causally-validated site-selection RULE that beats probe-selection and output-steering.** That is
"interpretability-guided *choice of where to steer*, validated causally" — real but **incremental**.

### Two biggest risks
1. **Wash-out (kills the thesis).** A diffusion/flow sampler re-projects toward its learned data manifold at
   every subsequent step, so a mid-trajectory edit may leave **no** trace in the final structure (Jed26b already
   saw strand-steering ≈ 0). If G0 fails broadly, there is no "intermediate steering" story. **Mitigation:**
   test persistence first; include output-step steering as the fallback result.
2. **Goodhart / circularity.** Steering toward a scalar that is the model's own head (ipSAE/ipTM) inflates it
   trivially; steering toward BSA can be "won" by clashing/unphysical structures. **Mitigation:** geometric,
   model-independent primary target + validity filter + held-out readout. (This is also why ipSAE must be
   secondary.)
   *Secondary risks:* seed instability of SAE directions ([Jer26]); probe directions entangling BSA with size/
   length confounds (control for chain length and total SASA); the (ℓ,t) grid being too coarse to resolve a
   real window.

### Scoop exposure (ranked)
1. **The [Xun26] authors / general mech-interp steering groups** porting "lagged coupling" to diffusion/bio —
   highest conceptual scoop; the framing is published and one instantiation away.
2. **Oxford/Crook (Jedryszek)** — already own the co-folding decodable≠causal protocol; extending to the
   timestep axis is their natural next step.
3. **FoldSAE authors (Deja et al.) / Goodfire** — own SAE-steering-of-folding-diffusion plumbing.
4. **The [Khi26b] / critical-windows community** adding a biological target.
LIVE UNKNOWN: check goodfire.ai/research and arXiv listings 2026-10/11 before committing.

### Dual-use (a paper MUST address this explicitly)
- **This is capability-adjacent.** "Steer a structure-diffusion model toward better interface quality" is one
  short step from "design a tighter binder." Treat interface/affinity steering as **sensitive**.
- **Keep benign and defensive:** primary target a **geometric descriptor** (BSA/contacts), not affinity; benign,
  open, non-hazard complexes only (exclude toxins, viral fusion/entry, immune-evasion, select agents); **do not
  release** a steering pipeline tuned to maximize binding/affinity, and do not report per-step recipes that
  function as a binder-optimization how-to. Frame the deliverable as the **interpretability finding**
  (decodable≠steerable in diffusion time) and a **negative/calibration result**, not a generation tool.
- **Precedent to cite:** VFUSE [Yu26b] frames all-atom-diffusion SAE work around biosecurity; adopt a similar
  explicit statement. Note that the mechanistic knowledge "where a property is causally encoded" is inherently
  dual-use and argue the **defensive** value (auditing/monitoring generative design models) over the offensive.

### Recommended framing (raises odds, lowers hazard)
"**Decodable ≠ steerable along diffusion time**: a (layer × timestep) causal map of a benign geometric interface
descriptor in an open structure-diffusion model, showing probe-optimal sites mis-predict steerable sites, and a
causally-selected site-rule that beats probe-selection and output guidance." Venue: a mech-interp / ML-for-
structural-biology workshop first (fast stake in the ground), then a main track if G1+G2 are clean.

--------------------------------------------------------------------------------
## 7. Must-cite / must-differentiate ledger
--------------------------------------------------------------------------------
Framing twin: [Xun26]. Protocol antecedent: [Jed26b]. Closest bio method: [Zar25] FoldSAE. Must-beat baseline:
[Har25c] FK steering. Intermediate-guidance priors: [Kyn24], [Gol23], [Khi26b], [Li24h], [Jin25], [Wan23d].
Target-definition trap precedent: [Mig26] PairSAE. ipSAE: Dunbrack bioRxiv 10.1101/2025.02.10.637595 +
arXiv 2610.17108. Steering theory: [Shi26d], [Bea26], [Wur26]. Property-guided bio diffusion: [Jia24c],
[Wan24k], [Ric25], [Lam26b], [Vil23], [Aya25]. Biosecurity precedent: [Yu26b] VFUSE.
Open unverifieds to close before submission: confirm no 2026-Q4 Goodfire/Oxford preprint already does the
(layer×timestep) map; confirm Boltz-1 vs RFdiffusion plumbing choice; verify ipSAE arXiv 2610.17108 authorship.
