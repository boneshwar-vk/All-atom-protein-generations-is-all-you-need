# MLIP Physics Steering Head — Design & Data Plan

**Status:** planning (no code yet) · **Compute target:** ≤ 2× A100 + a multicore CPU node
**Companion docs:** `steering-head-atom-interaction-literature.md` (field survey), confidence-head plan (chat)
**Date:** 2026-11-03

---

## 1. Why add an MLIP head at all

The **confidence head** (ipSAE / PAE / ipTM / pDockQ / ΔG, predicted re-folding-free from a
backbone) answers *"does a folding model believe this interface is real and well-bound?"*. It does
**not** answer *"do the atoms actually obey chemistry?"* — a structure can have a confident-looking
interface while containing steric clashes, stretched peptide bonds, inverted chirality, non-planar
rings, or energetically absurd packing. Those are exactly the failures that make a generated design
die in the wet lab.

An **MLIP head** (machine-learned interatomic potential) supplies the missing *physics* verdict: a
differentiable energy — and its gradient, a **force field** — scoring bonded geometry, steric
clashes, and non-covalent/interface interaction energetics (van der Waals, electrostatics,
H-bonding, desolvation). The two heads are **complementary steering signals**, not substitutes:

| | Confidence head | MLIP / physics head |
|---|---|---|
| Question | Is the interface *plausible / bound*? | Do the atoms *obey physics*? |
| Signal | ipSAE, PAE, ΔG proxy | Energy + per-atom forces + validity energy |
| Trained against | Co-folder PAE, Rosetta ΔG, experimental Kd | QM/MLIP reference energies & forces |
| Catches | Non-binders, bad interfaces | Clashes, strain, chirality, bad packing |

**Thesis (unchanged, now extended):** a *learned, generator-agnostic, re-folding-free, backbone/
x̂₀-level* scorer that scores the denoised estimate and steers diffusion generators — now with a
**physics head** added beside the confidence head. Agents 5 (Baker-lab sweep) and 6 (trunk survey)
both independently confirm the key gap: **no existing protein model emits both a geometric/confidence
property and a learned interatomic energy/force from one trunk, and none is used as a fast,
differentiable, in-loop physics steering signal.** RFdiffusion's hand-crafted guiding potentials
(`monomer_ROG`, attractive/repulsive `contacts`, `substrate_contacts`) occupy exactly the
architectural slot this head would upgrade from hand-crafted geometry to real energetics; Rosetta
ref2015 only ever enters *off-loop* as post-hoc FastRelax/filtering. Notably, RFdiffusionAA's learned
all-atom denoiser already beat the earlier hand-crafted attractive/repulsive potential on
binding-energy evaluation — direct evidence that a *learned* physics signal can outperform
hand-crafted terms.

---

## 2. The two-head steering stack (where each signal enters)

The single most important design axis (Agent 4) is **where a scalar score E(x) is injected into the
reverse-diffusion step**. Three composable injection points:

- **(A) Gradient on the noisy state x_t** — classic classifier guidance. **Rejected for the MLIP
  head**: an interatomic potential is meaningless on a noisy, off-manifold x_t.
- **(B) Gradient through the denoised estimate x̂₀ (Tweedie / DPS / reconstruction guidance)** —
  evaluate E on the model's predicted *clean* structure x̂₀(x_t) and backprop. **This is the correct
  entry point for physics**, because x̂₀ is an approximately clean, near-physical geometry even when
  x_t is pure noise.
- **(C) No gradient — particle level** — evaluate E on x̂₀ (or final x₀) and resample/select whole
  trajectories (best-of-N, beam, MCTS, Feynman–Kac / SMC). E can be an expensive non-differentiable
  black box (full MLIP relax, true Rosetta ΔG, ipSAE).

The stack uses **(B)** for cheap differentiable signals (distilled MLIP + validity energy +
confidence head) and **(C)** for expensive non-differentiable ones (true MLIP, ipSAE, Rosetta),
in the same sampler.

---

## 3. What the MLIP head outputs

A single all-atom trunk feeds three output heads:

1. **Physical-validity energy E_val(x)** — a differentiable scalar = weighted sum of smooth, local,
   reference-free terms (Agent 7's recommended training targets):
   - bond-length deviation (harmonic vs. Engh–Huber / CCD ideals)
   - bond-angle deviation (harmonic)
   - van der Waals **clash / overlap energy** (soft-core; intra-chain, inter-chain, protein–ligand)
   - **ligand/side-chain strain energy** (single-point internal energy / per-atom force norm)
   - **chirality** penalty (signed volume / improper dihedral at Cα and stereocentres)
   - **planarity** (out-of-plane distance for aromatic rings, sp² centres, peptide planes)
   - **omega / peptide-bond torsion** (trans preference, proline exception)
   - **Cβ deviation**
   - (low weight, optional) Ramachandran / rotamer favorability as a smooth KDE prior — *never* its
     discrete outlier count.
2. **Interaction / interface energy E_int(x)** — the physical binding energetics at the interface
   (vdW + electrostatics + H-bond geometry + desolvation), trained against QM/MLIP interaction
   energies. Optionally decomposed into SAPT-style components (electrostatics / exchange / induction /
   dispersion) following Splinter / AP-Net — this is the most informative steering signal.
3. **Per-atom forces F(x) = −∇E(x)** — via autograd on a **conservative** energy (see §4). Forces are
   the actual quantity injected into the sampler drift; Agent 4 found raw energies fluctuate too
   violently across the trajectory to guide stably, whereas forces are smoother.

**Hard categorical checks stay OUT of the head** as a non-differentiable **outer acceptance gate**:
PoseBusters PB-valid, MolProbity clashscore / Rama / rotamer thresholds, non-proline cis-peptide
detection, chirality pass/fail, PoseCheck strain percentile cutoff. The smooth head optimizes the
*surrogates*; the gate confirms the surrogates actually yield categorical validity (the recurring
caution: never trust a learned score without a hard physics check).

---

## 4. Architecture

**Design tension (Agent 6):** the MLIP head is the *binding constraint* — it requires all-atom
resolution + E(3)/SE(3)-equivariance + a scalar-energy output whose autograd gradient gives forces.
That rules out residue-level and invariant-only encoders (e.g. GearNet). The confidence head is
forgiving and can hang off pooled trunk embeddings. **So: choose the trunk to satisfy the MLIP head,
bolt the confidence/interface head on top.**

**Conservative, not direct forces (Agents 1 & 4):** energy-based guidance needs a gradient that is a
true potential's derivative (curl = 0). Use **conservative (autograd-force)** models; avoid
direct-force variants (Orb-v3-direct, UMA-L) — acceptable only as a crude one-step nudge, not as
principled energy guidance. Budget for **double-backward** (~2–3×) if you differentiate *through* the
force to train the steering coupling.

**Trunk options (ranked):**

1. **Small EquiformerV2 / MACE-style e3nn trunk (L_max 1–2, ~10–31M params)** — the only family with
   proven accurate all-atom **energy and force** (OC20/OC22; MACE strong on biomolecular PES). Scalar
   irrep head → E_val + graph-level ΔG/ipSAE; L=1 equivariant head → forces. Cap L_max and blocks low
   for diffusion-loop latency. MACE/Allegro are the loop-friendly members if force throughput
   dominates (Allegro's strict locality → O(N), 100M-atom demonstrated).
2. **ATOMICA** (Zitnik lab) — interface-native, pretrained on ~2M complexes with denoising+masking
   (physics-adjacent: denoising ≈ learning a force field), open weights (MIT). Strongest *interface
   prior*; emits atom/block/graph embeddings that naturally feed both an atom-level force head and a
   graph-level confidence head. Caveat: energy/force head not yet demonstrated — you'd validate it.
3. **GVP-GNN (all-atom, vector-gated)** — budget fallback: tiny, lowest latency on 2× A100, scalar+
   vector channels give invariant (energy/ΔG) + equivariant (force) outputs from one trunk. Lower
   force-accuracy ceiling.

**Dual-head wiring template:** GeoMFormer's invariant-stream (energy) + equivariant-stream (force)
pattern, applied to whichever trunk is chosen. Keep the Pairformer/AF3-style trunk and PeSTo only as
*confidence-head references* — the Pairformer is too heavy for an inner diffusion loop, PeSTo has no
force capability.

**Recommended starting pair:** prototype **#1 (small EquiformerV2/MACE)** for force fidelity and
**#2 (ATOMICA)** for the interface prior; benchmark force accuracy head-to-head before committing.

---

## 5. The noisy-x̂₀ problem (the crux)

MLIPs are trained near equilibrium; mid-diffusion x_t is far off-manifold, where MLIP energies/forces
are out-of-distribution and unreliable. Five stacking mitigations (Agents 4 & 8):

1. **Evaluate only on x̂₀, never on x_t** — the single most important mitigation (injection point B).
2. **Time-gated / scheduled guidance** — zero the physics term at high noise, ramp it in over the last
   ~20–30% of denoising steps as x̂₀ approaches equilibrium (DeltaDiff-style).
3. **Use forces, not energies** (or a smoothed surrogate) — energies fluctuate too much to guide
   stably; a learned force field is the smoother signal.
4. **Train the surrogate on diffusion-noised geometries** — sample x_t, compute x̂₀, label with the
   true MLIP on x̂₀, and train the cheap surrogate to be accurate *exactly where it will be queried*.
   This directly fixes the OOD problem and is the heart of the distillation plan (§6, §7).
5. **Uncertainty-gated guidance** — have the surrogate emit uncertainty u(x) and down-weight the term
   where it flags OOD (evidential/Bayesian MLIP UQ).

---

## 6. Integration recipe (4 tiers)

Base generator (RFdiffusion3 / Boltz-design / Chroma / Proteína) stays **frozen** — this is
inference-time steering, no retraining of the generator.

**Tier 1 — Distill; do not call the raw MLIP in-loop.**
Train a small SE(3)/E(3)-equivariant **surrogate** S_E(x) regressing a foundation MLIP's energy and
forces, trained on **diffusion-noised geometries** (§5.4), emitting uncertainty u(x). The confidence
head S_C(x) is already cheap/differentiable — keep as-is.

**Tier 2 — Gradient guidance on x̂₀, time-gated.**
In the last ~20–30% of steps, add to the score:
`g(x_t) = ∇_{x_t} [ λ_E · S_E(x̂₀) + λ_C · S_C(x̂₀) ]`
with λ ramped 0→target as t→0, each term down-weighted by its uncertainty, and **per-term gradient
norms z-scored/clipped** so the kcal/mol-scale energy doesn't swamp the bounded confidence term.

**Tier 3 — Particle selection with the true signals (Feynman–Kac / SMC), gradient-free.**
Run N particles; at a few checkpoints compute x̂₀, score each with a **rank-combined** reward of
{true MLIP energy, confidence, affinity} and resample by reward (Hartman et al. "difference"
potential on RFdiffusion is the template). Rank/z-score combination handles non-commensurable scales
far more robustly than raw weighted sums.

**Tier 4 — Final rerank / relax.**
On the N final x₀ candidates run the full MLIP relaxation + confidence + (optional) docking affinity;
select best-of-N or the Pareto front. Zero OOD risk (clean geometry). Apply the PoseBusters/MolProbity
gate here.

**Build order:** start with **Tier 1 + Tier 4** (best-of-N with true-MLIP rerank) — a strong,
low-risk baseline with no in-loop OOD risk. Add **Tier 3** (FK) for sample efficiency, then **Tier 2**
(gradient guidance) only if finer in-loop control is needed.

---

## 7. Training data (reuse / generate / label)

The signal the head needs is **per-atom forces on a consistent PES**, plus total/interface energy and
(optionally) SAPT components. That reshapes the plan: MM-GBSA gives an interface ΔG *scalar* but not
QM-quality forces, so it is a complementary label, not a force source.

### Tier R — REUSE (the bulk; download, ~0 GPU)

**Covalent / bonded / peptide physics (energies + forces):**
- **OMol25** (Meta FAIR; >100M DFT single-points, ≤350-atom systems, ωB97M-V/def2-TZVPD, biomolecules
  + electrolytes + metals + charges) — **the single most important base**; natural set to pretrain/
  distill from. (Data CC-BY; models under separate Meta license — verify for any product use.)
- **SPICE-2** (~2M conformers, ωB97M-D3BJ/def2-TZVPPD, 17 elements, dipeptides + ions + water, **CC0**)
  — best CC0 peptide/ion coverage.
- **ANI-2x** (HCNOFSCl, forces) and **Transition1x / nablaDFT** for strained/off-equilibrium diversity.

**Non-covalent / interaction energy (the interface core):**
- **DES370K** (CCSD(T)/CBS gold standard dimers incl. amino-acid sidechain analogs + ions; **CC0**) —
  highest-fidelity reusable non-covalent reference.
- **Splinter** (~1.7M dimers, **SAPT0 with electrostatics/exchange/induction/dispersion components**,
  protein + ligand fragment sets) — most directly interface-relevant; SAPT components are exactly the
  steering signal.
- **PANIP / PDB-FRAGID** (2026; NequIP MLIP on **PDB-derived protein fragment dimers**, two-fidelity
  r2SCAN-3c + ωB97X-D3BJ/def2-TZVPP) — closest existing thing; exploit its data/code release.
- **DES5M** (bulk, SNS-MP2 quality) for pretraining scale.

**Interface *geometries* (MM-level labels only — reuse for sampling, not QM labels):**
- **MISATO** (~19,443 protein–ligand complexes, MD + ligand QM) — best public source of realistic
  interface geometries to sample and relabel.
- **mdCATH** (coords + **MM forces**, CHARMM22*; note CC-BY-**ND** — check redistribution), **ATLAS**
  (coords, CHARMM36m) — monomer conformational diversity.

*Use **OpenQDC** as a single loader that bundles SPICE/ANI/DES/Splinter/QMugs to avoid per-dataset
plumbing.*

### Tier G — GENERATE yourself on 2× A100 + CPU (close the one real gap)

The gap no public set fills: **real protein–protein/ligand interface interaction energies on realistic
patches** — many-body, backbone-involving, charged (salt bridges), near-clash geometries the generator
actually produces. Strictly-2-body small-molecule dimer sets (DES/Splinter) and MM-only complex sets
(MISATO/mdCATH) do not cover this. Plan:
1. **Sample interface geometries** — short OpenMM MD (hundreds of ns/day/GPU on A100) seeded from
   MISATO/PDBbind complexes **and from your own generator outputs + decoys**; extract interface
   residue–residue patches / clusters, *including perturbed and near-clash configs*.
2. **Mass foundation-MLIP labels (GPU, all 10k–100k):** relabel every patch/complex with **MACE-OFF-
   large** (clean license, protein-validated, ωB97M-V level) or **AIMNet2** (charged-capable; truncate
   to ≤~1,500-atom clusters — it OOMs above that). Cost: **minutes–few hours per 100k per A100.** This
   is the bulk force/energy signal. Ceiling = teacher accuracy, so validate on the anchors below.
3. **GFN2-xTB second opinion (CPU):** single-point + gradient on truncated interface clusters
   (~200–500 atoms), ~13 CPU-hours per 100k on a 64-core node; catches foundation-MLIP failures on
   charged/polar interfaces. Use `dxtb` for GPU-batched differentiable baselines; **avoid
   `tblite`+`tfep` — no double-backprop, can't force-match.**
4. **Scalar interface ΔG (near-free):** single-structure **MM-GBSA** (OpenMM/AmberTools) on all
   complexes — a *ranking* label (r≈−0.65 on PPIs), not accurate absolute ΔG.

### Tier L — LABEL with QM + delta-learning (small, targeted)

- **GPU-DFT top-up** (GPU4PySCF / TeraChem) on **10⁴–10⁵** ≤~50-atom interface fragment dimers at
  ωB97X-D/def2-SVP(D) — GPU-weeks; corrects the distribution shift (real interface motifs, charged
  pairs, backbone H-bonds) that Splinter/DES miss. Optionally match Splinter's SAPT level on a subset.
- **Delta-learning** (GFN2-xTB baseline → DFT correction) to QM-calibrate labels at xTB cost and
  de-bias the foundation-MLIP teacher; keep the DFT anchor set to ~1k–10k.
- Reserve a few hundred **QM/MM** spot-checks for the hardest charged/metal interfaces.

### SKIP (infeasible or wasteful on 2× A100)
Regenerating a SPICE/OMol-scale QM set (QM is CPU-bound; OMol25 ≈ 6.6B CPU-core-hours); full-complex
QM geometry optimization at scale; MD-sampled MM-PBSA at scale (use single-structure MM-GBSA);
GFN-FF on charged interfaces (it fails there); automated QM/MM as a bulk labeler.

---

## 8. Training curriculum

1. **Pretrain the trunk** with masked-atom modeling + coordinate denoising (≈ learning a force field)
   on reused structures — or start from ATOMICA's pretrained weights.
2. **Supervise energy + forces** on the reused QM/MLIP sets (OMol25 / SPICE-2 / DES370K / Splinter /
   PANIP) + the Tier-G foundation-MLIP-labeled interface patches. **Train on diffusion-noised
   geometries** so the head is accurate on x̂₀ (§5.4). Loss: energy MSE + force MSE (+ SAPT-component
   loss where available) + the validity-energy terms (§3).
3. **Delta-learning calibration** against the Tier-L DFT anchors; de-bias the teacher.
4. **Confidence/interface head** fine-tuned on the existing engine-distilled + experimental labels
   (from the confidence-head plan): ipSAE/PAE (free from any co-folder), Rosetta ΔG, and experimental
   affinity (SKEMPI/PPB-Affinity/PDBbind) with **leakage-aware splits** and a **ranking** objective.
5. **Validate** against the hard gate (PoseBusters / MolProbity / cis-peptide / chirality) and on
   held-out targets; confirm the smooth validity energy actually tracks categorical validity.

---

## 9. Compute budget on 2× A100 (realistic)

| Task | Where | Rough cost |
|---|---|---|
| Download + preprocess reused QM/interaction sets (OMol25, SPICE-2, DES, Splinter, PANIP) | CPU/IO | days of IO, ~0 GPU |
| Foundation-MLIP labeling of 10k–100k interface patches | GPU | minutes–few hours / 100k / A100 |
| Short MD interface sampling (OpenMM) | GPU | hundreds of ns/day/GPU |
| GFN2-xTB anchors (10–30k clusters) | CPU | ~13 CPU-h / 100k on 64 cores |
| MM + single-structure MM-GBSA (all) | GPU/CPU | hours |
| GPU-DFT delta anchor (10⁴–10⁵ small dimers) | GPU | GPU-weeks |
| Train trunk + heads (10–31M params) | GPU | days on 2× A100 |

All fit comfortably within 2× A100 + a multicore CPU node. The expensive-but-optional piece is the
GPU-DFT anchor; everything else is hours-to-days.

---

## 10. Where this sits vs. prior art (the gap being targeted)

- **Closest learned energy head:** DSMBind (SE(3) denoising score-matching energy, unsupervised on
  unlabeled complexes) — but it targets *binding energy*, not atomistic physical validity, and isn't
  demonstrated as generator-agnostic in-loop steering.
- **Closest physics steering:** RFdiffusion's hand-crafted guiding potentials and FK-steered
  RFdiffusion (Hartman 2025) with ProteinMPNN + Rosetta-relaxation potentials — both **hand-crafted /
  non-differentiable**, not a learned MLIP.
- **Closest data/model:** PANIP/PDB-FRAGID (NequIP on PDB fragment dimers) — but pairwise, 2-body,
  weak on clashing geometries; not a steering head.
- **Empty quadrant:** a **generator-agnostic, differentiable, learned interatomic potential** that
  consumes x̂₀ each denoising step and emits a physical-validity + interface-interaction energy with
  conservative forces, fast enough for FK/SMC steering, trained to a QM/MLIP reference. Neither the
  Baker ecosystem nor the broader field has this.

---

## 11. Risks

1. **OOD / unreliable MLIP forces on noisy states** — the central failure mode; mitigated by x̂₀-only
   evaluation, time-gating, training the surrogate on noised geometries, uncertainty-gating.
2. **Biased Tweedie/DPS guidance** at high noise — never guide at high noise; prefer particle
   resampling for the heavy signals.
3. **Distillation error propagation** — a wrong surrogate confidently steers off-target; keep the true
   MLIP in the final rerank as a backstop, monitor surrogate-vs-true divergence.
4. **Objective scale mismatch & reward hacking** — energy/confidence/affinity are non-commensurable;
   use rank/z-scored combination and an independent held-out validator (high confidence ≠ specificity).
5. **Long-range electrostatics** — short-range-only MLIPs (MACE-OFF base, Egret, ANI) over-collapse
   solvated/complex structures; use long-range-augmented variants (MACELES-OFF) or charge-aware models
   (AIMNet2, SO3LR) for interface energetics.
6. **Teacher-accuracy ceiling & licensing** — distillation caps you at the teacher's accuracy/domain;
   verify the UMA / FAIR Chemistry and mdCATH CC-BY-ND licenses before any redistribution/product use.
7. **Compute blow-up in-loop** — cost ≈ #steps × #particles × (denoiser pass + score + gradient);
   late-only guidance, few FK checkpoints, and the cheap surrogate keep it bounded.
8. **No ground-truth interface QM set** — the Tier-G/L generation is the mitigation; validate against
   GFN2-xTB/DFT anchors on charged and metal-containing interfaces before trusting the head.

---

## 12. Phased build plan

- **Phase 0 (de-risk, ~1–2 weeks):** reuse OMol25 + SPICE-2 + DES370K + Splinter; train a small
  EquiformerV2/MACE trunk with energy+force+validity heads; validate force accuracy and PoseBusters/
  MolProbity correlation. Implement **Tier 1 + Tier 4** best-of-N rerank on a frozen generator. Prove
  the head improves physical validity of selected designs.
- **Phase 1 (~2–4 weeks):** generate the Tier-G interface set (MD sampling + foundation-MLIP + xTB +
  MM-GBSA on your own generator outputs/decoys, 30–60k patches); retrain on diffusion-noised geometries;
  add the GPU-DFT/delta-learning calibration; add **Tier 3** FK steering.
- **Phase 2:** add **Tier 2** gradient guidance; benchmark ATOMICA trunk vs. EquiformerV2; integrate
  the confidence head into the combined rank-based reward; wet-lab-style held-out evaluation.

---

### One-line summary
Build a small **conservative E(3)-equivariant all-atom trunk** (EquiformerV2/MACE or ATOMICA) that
emits a **differentiable physical-validity + interface-interaction energy with autograd forces**,
**distilled from a foundation MLIP (MACE-OFF/AIMNet2) onto diffusion-noised x̂₀ geometries** and
calibrated with reused QM data (OMol25/SPICE-2/DES370K/Splinter/PANIP) plus a small self-generated
interface set; plug it in beside the confidence head as **x̂₀ gradient guidance (late, time-gated) +
Feynman–Kac rank-based reranking + best-of-N final relax** on a frozen generator. All feasible on
2× A100 + CPU.
