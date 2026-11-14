# Steering-Model Compendium — All Ideas, Organized

**Purpose:** a single, complete, organized record of every idea we have developed for a learned **steering model** for generative protein-complex design — from reusing existing data, through confidence/interaction heads, to MLIPs and MLFFs — with well-thought descriptions of each module, the data to train them, how they steer a generator, and which contribution to pursue.

**Scope:** this is the index/compendium. It consolidates and supersedes the scattered detail in the three companion docs, which remain the deep dives:
- `steering-head-atom-interaction-literature.md` — raw literature survey (confidence heads, interaction modules, steering, SSL, datasets, Baker-lab sweep).
- `mlip-physics-steering-head-plan.md` — the MLIP physics-head design & data plan.
- `research-plan-icml2027.md` — the locked single-contribution research plan + timeline.

**Date:** 2026-11-14 · **Compute frame:** ≤ 2× A100 + CPU node; 1 week (~336 GPU-h) dedicated data generation.

---

## Executive summary (one screen)

**What we are building.** A learned scoring module that guides a *frozen* generator (RFdiffusion3, Boltz-design, Chroma, Proteína-Complexa) toward better interface quality, binding energetics, and physical validity — by scoring the generator's **denoised estimate x̂₀** and injecting the signal as (B) time-gated gradient guidance on x̂₀, (C) Feynman–Kac/best-of-N reranking, and (D) a final relax + hard validity gate. No generator retraining.

**The steering objective** (the through-line):
```
E_head(x̂₀) = E_feasibility            (always-on, robust on clashes)
            + λ · E_interface          (headline: binding energetics, uncertainty-gated)
```
- `E_feasibility` = a **learned-MM-parameter force field** (Grappa/Espaloma via DMFF) + a smooth physical-validity energy + a learned GB implicit-solvent term. Robust on near-clash geometry *by construction*; MM-cheap; evaluated every step.
- `E_interface` = a small **SE(3)-equivariant surrogate** regressing expensive interface energetics (Rosetta ΔG + xTB/MLFF interaction energy + reused experimental ΔG), **distilled onto diffusion-noised geometries**, emitting uncertainty. The binding-driving, novelty-bearing term.
- Confidence (ipSAE/PAE) and affinity (Kd) enter as **rank-combined auxiliary rerank signals**, not the headline.

**The locked contribution (for ICML 2027):** *a learned, generator-agnostic, re-folding-free, differentiable physics-steering head — distilled onto noised geometries, uncertainty-gated — that, trained once, steers two structurally distinct frozen generators toward better held-out interface binding energetics without per-generator retraining.* Headline is **binding energetics, not validity**; the ML novelty is the **distillation-onto-noised-manifold + uncertainty-gating + cross-generator transfer** mechanism, not "force-field guidance on x̂₀" (which already exists).

**Two decisions that shape everything:**
1. **MLFF (learned-MM-parameter, Grappa/Espaloma) is the always-on term, not a QM-MLIP** — because QM-MLIPs suffer the *softening problem* (near-equilibrium training → wrong/too-soft forces on the near-clash x̂₀ a generator emits). QM-MLIP / SO3LR demoted to sparse refinement.
2. **Spend the data week on the one regime no database has** — near-clash, generator-distribution interface energetics (a perturbation ladder + a small GPU-DFT hero anchor) — not on co-folding (reusable) or bulk MLFF forces (nearly free).

**How the three parts fit:**
- **Part I — Modules & Architectures:** every head/module and the three build strategies; the trunk.
- **Part II — Data, Training & Steering Mechanics:** reuse catalog, generation options, the 1-week plan, injection mechanics, the training curriculum.
- **Part III — Ideas Ledger, Novelty & Decisions:** the C1–C5 candidate ledger, novelty positioning, decisions made, open decisions, timeline, experiment matrix.

---

# Part I — Modules & Architectures

### 1. The steering problem at a glance

A **steering head** is a learned scoring module that guides a *frozen* generative protein-complex model (RFdiffusion3, Boltz-design, Chroma, Proteína-Complexa/La-Proteina) toward designs with better interface quality, binding energetics, and physical validity — without retraining the generator. The central design commitment is that the head scores the generator's **denoised estimate x̂₀** (the model's best "clean" structure prediction at diffusion step t), *not* the noisy intermediate x_t, because x̂₀ is near-physical even when x_t is pure noise. A steering signal enters a frozen generator at exactly three injection points, which are composable:

| Injection point | Mechanism | Needs gradients? | Signal can be |
|---|---|---|---|
| **(B) Gradient on x̂₀** | add `∇_{x_t} E(x̂₀)` to the denoiser drift (Tweedie/DPS reconstruction guidance), time-gated to late steps | yes (cheap, differentiable heads) | learned FF, validity energy, confidence surrogate |
| **(C) Particle reranking** | N particles resampled by reward (Feynman–Kac/SMC, best-of-N, beam, MCTS) at checkpoints | no | expensive black boxes: true MLIP, ipSAE, Rosetta ΔG |
| **(D) Final relax + hard gate** | best-of-N on clean x₀, then PoseBusters/MolProbity acceptance gate | no | full-fidelity potentials + categorical validity |

(Point **(A)**, gradient on the noisy x_t directly, is rejected throughout: a physics potential is meaningless off-manifold.)

### 2. The three architecture strategies

Every steering/scoring module in the field — and every one we considered — falls into one of three build strategies:

- **Strategy A — Reuse a frozen predictor, structure module stripped.** Take an existing co-folder (AF2/AF3/Boltz), remove the structure-generation head, and read a score off the retained trunk. **Exemplars:** *AF3Score* (score-only AF3 adaptation — scores given coordinates without re-diffusing, improves binder enrichment); *BindEnergyCraft* (casts a structure predictor's confidence as an energy-based model, replacing ipTM in a BindCraft-style loop). Cheapest to stand up; tied to one heavy model; its "energy" is confidence/geometry, not physical interaction energy.
- **Strategy B — Frozen trunk + small new head on latents.** Keep a frozen folding/representation trunk, train a small head on its intermediate representations. **Exemplars:** *PreFold-dG* (predicts PPI ΔG from a folding model's intermediate representations, 2026); *AlphaRank*-style rerankers on AF latents. Light to train; inherits the trunk's inductive bias; not generator-agnostic and not a true physics signal.
- **Strategy C — New standalone architecture.** Train a fresh model (usually SE(3)/E(3)-equivariant) on structures. **Exemplars:** *AlloGen* (differentiable, generator-agnostic interface scorer for conformational-state selectivity); *DSMBind* (SE(3) denoising score-matching energy, unsupervised binding energy); *PeSTo* (atom-level interface-contact transformer); *ATOMICA* (interface-native SE(3) model pretrained on ~2M complexes). The only strategy that can be simultaneously generator-agnostic, differentiable, and a true learned potential — and the one our stack is built on.

Our thesis occupies the **empty quadrant** none of the above fully reaches: a *learned + differentiable + in-loop-cheap-via-distillation + OOD-robust-on-noised-x̂₀ + generator-agnostic + protein-complex-interface* head.

### 3. The module / head catalog

#### 3.1 Confidence head
- **Predicts:** co-folder confidence — **ipSAE** (Dunbrack 2025, interface-restricted, length-robust; beats ipTM), **PAE** (N×N), ipTM, **pDockQ** — directly from a backbone, re-folding-free.
- **Architecture / I→O:** pooled interface embeddings from the shared trunk (§3.9) → scalar heads (ipSAE/ipTM/pDockQ) + optional N×N PAE head. Input: all-atom/backbone coords of a target+binder complex (or x̂₀). Output: confidence scalars (+ PAE matrix).
- **Differentiable:** yes (small head).
- **Steers via:** (B) gradient on x̂₀ or (C) reranking; ipSAE/PAE labels are **free** from any co-folder's PAE, so this is the data-abundant head.
- **Closest prior art:** AF3Score, AlloGen, BindEnergyCraft; ipSAE successors — Badonyi probabilistic interaction score (2026-07), reactifpTM (2026-08).
- **Role in our stack:** the "interface-quality" channel; *not* the headline (optimizing predicted ipSAE is reward-hacking bait — high confidence ≠ specificity/affinity, per Smorodina 2026). A **constraint/auxiliary** signal, reused-label-trained.

#### 3.2 Interaction / binding-energy head
- **Predicts:** interface **ΔG** and mutational **ΔΔG**, as ordinal bins + a regression head; optionally a DSMBind-style *unsupervised* energy.
- **Architecture / I→O:** shared SE(3) trunk → scalar energy head; DSMBind variant = SE(3) denoising score-matching EBM (perturb complex by rigid-body + torsion noise, learn to denoise; read energy off). Input: complex coords → output: ΔG/ΔΔG scalar (gradient = force).
- **Differentiable:** yes.
- **Steers via:** (B)/(C); the binding-driving term.
- **Closest prior art:** DSMBind + NERE (unsupervised binding energy); Boltz-2 affinity module (approaches FEP); PPAP; Kienlein transferable interface energetics (2026-09); PreFold-dG (2026-08).
- **Role in our stack:** **the headline signal** — binding energetics that validity and confidence terms cannot supply. Trained on Rosetta ΔG + xTB/MLFF interaction energy + reused experimental ΔG; **on diffusion-noised geometries**; uncertainty-gated; ranking loss (not pure regression) because labels are ranking-grade.

#### 3.3 Affinity head
- **Predicts:** experimental **Kd / ΔG_exp** (ordinal).
- **Architecture / I→O:** a small head on the shared trunk, fine-tuned *last* on top of an already-good interface representation. Input: complex → output: affinity bin/value.
- **Differentiable:** yes; used mostly as a reranking/calibration signal.
- **Steers via:** (C) reranking; too scarce to drive per-step gradients reliably.
- **Data:** **reuse only, never generate** — SKEMPI 2.0 (7,085 ΔΔG), PPB-Affinity (~8.2k Kd), PDBbind-PP (2,852), Affinity Benchmark v5.5, SAbDab, AB-Bind; **leakage-aware splits** (cluster by sequence/structure identity), ranking objective.
- **Role in our stack:** final calibration channel; the rare, precious ground-truth that only tunes a small head — which is why ~8k points suffice.

#### 3.4 MLIP physics head (QM-trained interatomic potential)
- **Predicts:** total/interface **energy + per-atom forces** at ~QM fidelity.
- **Architecture / I→O:** E(3)-equivariant potential (MACE-OFF/AIMNet2/SO3LR-class); **conservative** forces via autograd of a scalar energy. Input: all-atom coords → output: energy (+ forces = −∇E).
- **Differentiable:** yes (autograd).
- **Key liability — the "softening" / noisy-x̂₀ problem (Zhong et al. 2024):** QM-MLIPs are trained near equilibrium, so they underpredict repulsive PES curvature and give wrong/too-soft forces *exactly on the near-clash x̂₀ a generator produces*. Mitigations: evaluate on x̂₀ only; time-gate to late steps; **distill onto diffusion-noised geometries**; uncertainty-gate; or demote to refinement (§3.6).
- **Closest prior art:** Force-Guided SE(3) diffusion (uses a *smoothed force net* for this exact reason); PANIP/PDB-FRAGID (NequIP on PDB fragment dimers).
- **Role in our stack:** **not the always-on term** (softening). Either the distillation *teacher* for the interface-energy head, or a sparse low-noise-step refinement signal.

#### 3.5 MLFF head — learned-MM-parameter force fields (the always-on term)
- **Predicts:** a classical **MM energy + forces**, but with *learned* parameters.
- **Architecture / I→O:** a GNN predicts the parameters of a standard MM functional (bonds/angles/torsions/charges/LJ) **once per molecular graph** — **Grappa** (graph-attention + symmetry-preserving transformer, no hand-crafted features, outperforms Espaloma at MM speed), **Espaloma-0.3** (end-to-end differentiable, self-consistent protein+ligand+NA), **ByteFF/ByteFF-Pol** (adds polarization) — then evaluates the analytic MM energy via **DMFF** (JAX, PME long-range, optional polarizable AMOEBA). Add a **learned implicit-solvent GB** term (Airas–Zhang ESM3-GNN+GB, 2026) for interface desolvation. Input: graph + coords → output: energy (closed-form, analytically differentiable in coords).
- **Differentiable:** yes, analytically — after the one-time parameter pass, ∂E/∂x is cheap and exact.
- **Why it's the primary term:** robust on clashes/off-equilibrium *by construction* (analytic LJ wall + harmonic bonds), 2–4 orders of magnitude cheaper per eval than a QM-MLIP (so affordable every step), and auto-parameterizes arbitrary interface residues/ligands/non-canonical chemistry (no atom-typing). This is what makes stable per-step steering feasible on ≤2 A100.
- **Closest prior art / precedent:** DiffForce (2406.05832 — FF-gradient guidance on the denoised sample); Lai et al. 2026 (MMFF94 guidance); Elign (2601.21985 — MLFF as alignment signal). These make "FF-guidance-on-x̂₀" prior art — so our novelty is the *learned surrogate + noised-manifold + transfer*, not FF guidance per se.
- **Role in our stack:** **the always-on physical-feasibility regularizer term** `E_feasibility` of the steering objective.

#### 3.6 High-fidelity refinement potential (sparse)
- **Predicts:** energy+forces at the best available fidelity.
- **Exemplars:** **SO3LR** (SO3krates equivariant transformer + *explicit* long-range electrostatics + dispersion; validated in explicit solvent to ~200k atoms/GPU — the one biomolecular neural FF with native long-range + solvent + near-complex scale); or the QM-MLIP (§3.4).
- **Differentiable:** yes; expensive.
- **Steers via:** (C)/(D) only — low-noise steps / final rerank where x̂₀ is already near-physical, so softening is irrelevant and accuracy pays off. Also the head-to-head benchmark against the fast MM-param head.
- **Role in our stack:** sparse refinement + a fidelity ceiling for ablations.

#### 3.7 Physical-validity energy
- **Predicts:** a differentiable **physical-validity energy** = weighted sum of smooth, local, reference-free terms: bond-length & bond-angle harmonic deviations, **van der Waals clash/overlap** (intra-chain, inter-chain, protein–ligand), ligand/side-chain strain, **chirality** (signed-volume/improper dihedral at Cα + stereocentres), **planarity** (aromatic rings, sp², peptide planes), **omega** peptide-bond torsion (trans; proline exception), **Cβ deviation**. (Ramachandran/rotamer only as a low-weight smooth prior — never as an outlier *count*.)
- **Differentiable:** yes (all terms smooth and local) — usable as both a training target and a gradient-guidance signal.
- **Hard counterpart (NOT in the head):** **PoseBusters PB-valid**, **MolProbity** clashscore/Rama/rotamer thresholds, non-proline cis-peptide detection, chirality pass/fail, PoseCheck strain percentile — kept as a **non-differentiable outer acceptance gate** (point D).
- **Closest prior art:** closed-form validity projection (2610.07037, 2510.08946) reaches 97–100% PB-valid **with no learning** — which is *why validity must be a constraint term, never the headline*.
- **Role in our stack:** folded into `E_feasibility` as smooth terms; the categorical checks confirm the smooth energy actually yields valid structures.

#### 3.8 Atom-level masked-interaction / SSL module
- **What it is:** self-supervised pretraining for the trunk — **masked-atom modeling + coordinate denoising** (Uni-Mol/DSMBind-style; denoising ≈ learning a force field, Zaidi). Input: corrupted complex → reconstruct atoms/coords.
- **The critiqued idea:** "mask Kd → predict Kd bins from *random* interface atoms." **Verdict: fails as a standalone supervised task** — data scarcity, interface≠ΔG, hot-spot/O-ring effects, random-atom sampling discards binding determinants, OOD collapse. **It survives only as an SSL *pretext*** (masking to learn representations), with a small ordinal ΔG/Kd head fine-tuned on full interfaces afterward with leakage-aware splits + ranking loss.
- **Role in our stack:** encoder pretraining step (Step 1 of the curriculum); *not* a prediction head on its own.

#### 3.9 Shared all-atom equivariant trunk
- **The binding constraint:** the MLIP/MLFF head *requires* all-atom resolution + E(3)/SE(3)-equivariance + a scalar-energy output whose autograd gradient gives forces. This **rules out residue-level / invariant-only encoders** (e.g. GearNet — invariant & residue-level ⇒ cannot produce forces). Choose the trunk to satisfy the force head; hang the confidence/interface heads off pooled embeddings.
- **Conservative, not direct forces:** energy-based guidance needs a curl-free gradient of a true potential — use autograd/conservative models; avoid direct-force variants.
- **Options:**

| Trunk | Equivariance / resolution | Why / caveat |
|---|---|---|
| **EquiformerV2 / MACE (low L_max, ~10–31M)** | E(3)-equiv graph transformer / ACE / atom | **Rec #1** — only family with proven all-atom energy *and* force; scalar head → energy+ΔG/ipSAE, L=1 head → forces. Cap L_max 1–2 for loop speed; add interface featurization. |
| **ATOMICA** (Zitnik) | SE(3)-equiv hierarchical MP / atom+block+graph | **Rec #2** — interface-native, pretrained ~2M complexes with denoising+masking (physics-adjacent); strongest interface prior; energy/force head not yet demonstrated (you'd validate it). |
| **GVP-GNN (all-atom, vector-gated)** | SO(3)-equiv scalar+vector / atom | Budget fallback — tiny, lowest latency; scalar(energy)+vector(force) from one trunk; lower force-accuracy ceiling. |
| **GeoMFormer** | invariant-stream(energy)+equivariant-stream(force) | Not protein-tuned — adopt its **dual-head wiring template** on whichever trunk. |
| **PeSTo / AF3-Pairformer** | — | Confidence-head *reference only*: PeSTo has no forces; Pairformer too heavy for an inner loop. |

- **Key fact:** no existing protein model emits *both* a geometric/confidence property *and* a learned energy/force from one trunk — this is new work.

### 4. Head comparison table

| Head | Signal emitted | Trained against | Differentiable | Per-eval cost | Role in stack |
|---|---|---|---|---|---|
| Confidence (§3.1) | ipSAE/PAE/ipTM/pDockQ | co-folder PAE (free) | yes | low | interface-quality / constraint channel |
| Interaction–energy (§3.2) | ΔG/ΔΔG (+ unsup. energy) | Rosetta ΔG + xTB/MLFF + exp ΔG, on noised geoms | yes | low–mid | **headline binding signal** |
| Affinity (§3.3) | Kd/ΔG_exp | SKEMPI/PPB-Affinity/PDBbind (reuse) | yes | low | final calibration (rerank) |
| MLIP physics (§3.4) | energy + forces (QM) | QM E/F (SPICE/OMol25…) | yes (autograd) | high | distill teacher / sparse refine (softening) |
| **MLFF MM-param (§3.5)** | MM energy + forces | QM E/F → MM params (Grappa/Espaloma) | yes (analytic) | **cheap** | **always-on feasibility term** |
| Refinement potential (§3.6) | energy + forces (hi-fi) | QM + explicit solvent (SO3LR) | yes | high | sparse rerank / fidelity ceiling |
| Physical-validity (§3.7) | smooth validity energy | analytic ideals (ref-free) | yes | cheap | feasibility sub-terms |
| Hard gate (§3.7) | PB-valid / MolProbity pass-fail | — | **no** | cheap | outer acceptance gate |
| SSL / masked (§3.8) | reconstructed atoms/coords | masked-atom + denoising | yes | n/a | trunk pretraining only |

**Through-line:** the steering objective is `E_head(x̂₀) = E_feasibility (MLFF MM-param + smooth validity, always-on, robust on clashes) + λ·E_interface (distilled binding-energy head, headline, uncertainty-gated)`, computed on x̂₀, injected as late/time-gated gradient guidance + Feynman–Kac reranking + best-of-N final relax, with confidence/affinity as rank-combined auxiliary rerank signals and the hard validity gate at the end.

---

# Part II — Data, Training & Steering Mechanics

This part covers everything about *feeding and training* the steering model: what to reuse, what to generate, how to spend the one-week compute budget, how the steering signal enters a frozen generator, and the training curriculum. Companion to Part I (heads & architectures).

### 1. Reusable data catalog (generate nothing)

The governing principle: **spend GPUs only where no database exists.** Everything below is downloadable and covers most training needs; data generation (§2–3) is reserved for the one regime no public set has (near-clash, generator-distribution interface energetics).

#### 1a. Structures + confidence labels (ipSAE/PAE for the confidence head)

| Dataset | What it is | Size | Label type | Reuse role |
|---|---|---|---|---|
| **AFDB** (AlphaFold DB) | Predicted structures, **PAE downloadable** | 200M+ structures | PAE → ipSAE/ipTM/pDockQ computed free | Bulk confidence-label source; compute ipSAE from stored PAE at zero GPU |
| **PINDER** | Protein–protein interaction pairs incl. AFDB-paired partners; **leakage-aware splits provided** | ~2.5M PPI pairs | geometry + derivable confidence | Primary PPI structure pool + ready splits |
| **PLINDER** | Protein–ligand systems; leakage-aware splits | ~1.3M PL systems | geometry | PL structure pool for interface head |
| **Boltz-2 / Protenix distillation sets** | Co-folder-generated structure distributions | large | model confidence | Match the co-folding distribution without re-running it |

Key point: ipSAE is a deterministic function of PAE, so ipSAE/ipTM/pDockQ labels are **free** wherever PAE is stored. Do not spend GPUs co-folding for confidence labels AFDB/PINDER already cover.

#### 1b. Experimental affinity — **reuse only, never generate**

Used solely to fine-tune the affinity/ΔG head (small, precious signal; leakage-aware splits + ranking loss).

| Dataset | Content | Size |
|---|---|---|
| **SKEMPI 2.0** | Mutational ΔΔG for PPIs | 7,085 entries / 345 complexes |
| **PPB-Affinity** | PPI Kd / affinity | ~8.2k Kd (~12k aggregated) |
| **PDBbind v2020** | PL + PP affinity | PL 19,443; PP 2,852 |
| **Affinity Benchmark v5.5** | Curated PPI affinities | 179 |
| **SAbDab** | Antibody structures + affinity | 739 affinity points |
| **AB-Bind** | Antibody mutational ΔΔG | 1,101 mut / 32 complexes |
| **ATLAS (TCR-pMHC)** | TCR–pMHC affinity | ~137 |
| **BindingDB** | PL affinity (large) | ~2.9M |

#### 1c. QM energy+force sets (for the MLIP/MLFF side)

| Dataset | Level of theory | Size | Coverage / limits | Role |
|---|---|---|---|---|
| **OMol25** (Meta FAIR) — *center on this* | ωB97M-V/def2-TZVPD | >100M DFT single-points, ~83M systems, ≤350 atoms | Biomolecules, electrolytes (charged multi-molecule), metals, charges/spin; energies+forces | **Primary base** to pretrain/distill from; subsumes much of SPICE at higher theory. Data CC-BY; **models under separate Meta/FAIR license — verify before product use** |
| **SPICE-2** | ωB97M-D3(BJ)/def2-TZVPPD | ~2.0M conformers / 114k molecules | 17 elements incl. B,Si; dipeptides + ions + water; energies+forces; **CC0** | Best CC0 peptide/ion covalent+short-range set |
| **SPICE (v1)** | same | ~1.1M conformers | 15 elements; dipeptides, DES monomers/dimers, PubChem, ion pairs; **CC0** | Superset components; MACE-OFF training corpus |
| **ANI-1x / ANI-2x** | ωB97X/6-31G* | ~5M / ~4.6M | ANI-1x HCNO only; ANI-2x adds S,F,Cl (7 elts); **no P, no charged**; energies+forces | Clash/bonded diversity; narrow chemistry |
| **ANI-1ccx** | CCSD(T)*/CBS | ~500k | HCNO; **energies only (no forces)** | High-fidelity anchoring only |
| **Transition1x** | ωB97X/6-31G* | ~9.6M with forces | CHNO reaction paths | Off-equilibrium/strained geometry diversity |
| **nablaDFT** | ωB97X-D/def2-SVP | ~5M conformations | drug-like | Conformer diversity |
| **QMugs** | GFN2-xTB + ωB97X-D/def2-SVP | ~665k mol / ~2M conf | **no forces**, small molecules, no interfaces | Weak for force training |
| **OrbNet Denali** | ωB97X-D3/def2-TZVP | ~2.3M conf | **energies only**, small molecules | Energy anchoring only |
| **PubChemQC / QM9 / QM7** | PM6/B3LYP | large / 134k / 7k | mostly equilibrium, **no forces**, tiny organics | Legacy benchmarks only — not for interfaces/clash physics |

#### 1d. Non-covalent / interaction-energy sets (the **interface core**)

| Dataset | Level | Size | Why it matters | Limits |
|---|---|---|---|---|
| **DES370K** (DE Shaw) | **CCSD(T)/CBS** (gold) + SAPT/MP2 components | 370,959 dimer geoms / ~3,700 dimers | Highest-fidelity reusable non-covalent reference; incl. amino-acid sidechain analogs + ions; **CC0** | Small-molecule fragment dimers; **2-body only** |
| **Splinter** | **SAPT0** (elst/exch/ind/disp components) | ~1.7M configs / >9,000 dimers | SAPT **component decomposition** = exactly the interface physics a head wants; protein + ligand fragment sets | Synthetic fragment dimers, 2-body |
| **PANIP / PDB-FRAGID** (2026) | r2SCAN-3c (pool) + ωB97X-D3BJ/def2-TZVPP (distilled) | ~36.3M dimers pool; PDB-FRAGID ~3.15M | **PDB-derived protein fragment dimers** — closest existing thing; beats ANI-2x on charged/strongly-interacting | Pairwise, 2-body; weak on clashing geometries |
| **DES5M** | SNS-MP2 (ML-MP2 quality) | ~4.96M dimer geoms | Bulk pretraining scale | Lower fidelity than DES370K |

**The gap none of these fill:** real protein–protein/ligand **interface interaction energies** on realistic patches — many-body, backbone-involving, charged (salt bridges), **near-clash**, solvated. All public sets are small-molecule dimers (2-body), single peptides, or full complexes with MM-only energies. This gap is exactly what §2–3 generate.

#### 1e. MD trajectories (MM-level labels / geometry sources)

| Dataset | Content | Size | Labels | Caveat |
|---|---|---|---|---|
| **mdCATH** | 5,398 CATH domains, 5 temps × 5 replicas, CHARMM22* | ~62.6 ms aggregate | **coords + MM forces** | Forces are MM not QM; **single domains = monomers (no interfaces)**; **license CC-BY-ND** (NoDerivatives — check redistribution of a derived set) |
| **ATLAS** | ~1,390 chains, 3×100 ns, CHARMM36m explicit solvent | — | coords (forces not shipped) | Monomeric; **CC-BY**; use for sampling not labels |
| **MISATO** | ~19,443 protein–ligand complexes, MD + ligand QM | ~170 µs aggregate | MM energies/forces; QM on **ligand only** | **Best public source of realistic interface *geometries*** to sample and relabel; open (MIT/GPL) |

#### 1f. Loader
- **OpenQDC** — single loader bundling SPICE/ANI/DES/Splinter/QMugs/etc.; avoids per-dataset plumbing.

### 2. Data-generation options & feasibility

**The signal the physics head actually needs:** *per-atom forces on a consistent PES*, evaluated on the **off-equilibrium x̂₀ geometries generators actually produce** (near-clash, pre-relaxation) — plus scalar interface ΔG and confidence labels. This reframes what is worth generating: the free thing (MLFF forces) is *not* the novelty; the scarce thing (QM/xTB-anchored interface energetics on near-clash, generator-distribution geometries) is.

#### 2a. Throughput per option (336 GPU-h budget; CPU ≈ 10.7k CPU-h/week on a 64-core node)

| Option | Rate (A100, or CPU) | Datapoints if given the whole budget | Label | Fidelity |
|---|---|---|---|---|
| **Co-folding → ipSAE/PAE** (MSA-free) | ~55–90 GPU-s/complex | ~13–20k folds (**GPU hog**) | ipSAE/PAE/pDockQ | confidence proxy; **largely reusable free from AFDB/PINDER** |
| **Foundation-MLFF labeling** (MACE-OFF/AIMNet2; Espaloma=MM) | ~10–50 ms / ≤500-atom cluster | ~5–40M single points (**~40 GPU-h for whole corpus**) | **per-atom E+forces on a consistent PES** | teacher-limited (≈ωB97M-V short-range; AIMNet2 charged); **not the bottleneck, not the novelty** |
| **OpenMM MD / perturbation** | 100s ns/day/GPU | full geometry pool from ~60 GPU-h | geometries (then labeled) | MM ensemble |
| **GPU-DFT anchors** (GPU4PySCF, ≤50-atom, ωB97X-D/def2-SVP) | ~1–5 min each | ~2–4k realistic in ~70–90 GPU-h | QM E(+forces) | gold-ish; **de-biasing + OOD eval (the hero anchor)** |
| **GFN2-xTB** | sec–min/cluster (**CPU**) | millions (CPU-bound, cheap) | SQM E+forces | ~few kcal/mol; charged-interface second opinion |
| **Rosetta InterfaceAnalyzer score-only** | ~10–60 CPU-s | ~0.6–1.3M complexes (uses only ~1–2k CPU-h) | scalar interface ΔG | empirical, ranking-grade |
| **Rosetta cartesian_ddg** | ~0.5 CPU-h/mut | ~20k mutations (10.7k CPU-h) | ΔΔG | mutational ranking (**use this, not Flex-ddG ≈15 CPU-h/mut**) |
| **Rosetta relax+score** | ~3–20 CPU-min | ~30–60k complexes | relaxed ΔG | empirical |
| **MM-GBSA** (single-structure, OpenMM/AmberTools) | ~seconds/complex | all 100k feasible | scalar interface ΔG | **ranking-grade** (r≈−0.65 on PPIs), not accurate absolute |
| **FEP** | **~120 GPU-h / PPI edge** | **~3 edges in the whole week** | ΔΔG binding | high accuracy, **irrelevant at this scale → KILLED** |

**FEP is killed** for generation (3 edges ≠ a dataset). Reuse published FEP only, as a tiny *independent* eval anchor.

#### 2b. The perturbation ladder (the core generation idea)

Each complex → a **5–10-step ladder**: equilibrium → side-chain repack → backbone noise → **near-clash**. ~30–50k complexes × ladder = **~200k–400k geometries**.

**Why:** the head is trained on labels but *deployed* on the generator's noisy x̂₀ (near-clash, off-manifold). Labeling only equilibrium crystals makes the head OOD at deployment — the central failure mode. The ladder manufactures exactly the near-clash regime where MLFFs break and steering operates, so the head is accurate where it is actually queried. **A tooling fact that makes this necessary:** QM-MLIPs suffer the "softening" problem (near-equilibrium training → underpredicted repulsive curvature off-equilibrium); labeling near-clash geometries and/or using a learned-MM-parameter FF (analytic LJ wall) is the fix.

#### 2c. Reuse-vs-generate-vs-label framing

- **REUSE** (0 GPU): §1 — OMol25/SPICE-2/DES370K/Splinter/PANIP (MLFF physics), AFDB/PINDER/PLINDER (structures + confidence), all experimental affinity.
- **GENERATE** (GPU/CPU this week): the perturbation-ladder geometries on **generator-distribution + hard-decoy** complexes, bulk-labeled with foundation MLFF (cheap) + analytic validity terms.
- **LABEL** (small, targeted): GPU-DFT hero anchor (2–4k) + delta-learning (GFN2-xTB baseline → DFT correction) to de-bias the MLFF teacher; Rosetta/MM-GBSA scalar ΔG; ~10k targeted co-folds for ipSAE AFDB can't cover.
- **SKIP:** SPICE/OMol-scale QM regeneration (QM is CPU-bound; OMol25 ≈ 6.6B core-hours), full-complex QM opt, MD-sampled MM-PBSA at scale, GFN-FF on charged interfaces (it fails there), bulk FEP.

### 3. The 1-week (~336 GPU-h + CPU) data plan

**Guiding reallocation:** cut co-folding from ~30–50k to ~10k (rest reused from AFDB/PINDER), cut GPU-DFT from the earlier "GPU-weeks, 10⁴–10⁵" to a focused **2–4k hero anchor**, push bulk onto free MLFF + CPU xTB/Rosetta, and spend the freed GPU time on the **perturbation ladder** — the novelty-bearing regime.

#### 3a. Day-by-day schedule (GPU budget; CPU runs in parallel all week)

| Days | GPU work (~GPU-h) | CPU work (parallel) |
|---|---|---|
| **1–2** | Assemble reuse sets; collect RFD3 + latent-generator outputs; build hard decoys; **OpenMM MD + perturbation ladder (~60)**; stand up **MLFF labeling (~36)** over the growing pool | Launch Rosetta InterfaceAnalyzer score-only on everything |
| **3–4** | **Co-fold ~10k** generator-distribution + hard-decoy complexes for ipSAE/PAE (~120) | **cartesian_ddg ~20k mutations** |
| **5–6** | **GPU-DFT on ~2–4k ≤50-atom near-clash/charged interface clusters — the hero anchor (~90)**; finish MLFF labeling of late geometries | GFN2-xTB on a large interface-cluster subset |
| **7** | Buffer (~30): re-run failures, QC, carve held-out evals, finalize splits, package | — |

Total GPU ≈ 60+36+120+90+30 = **336 GPU-h.**

#### 3b. Resulting dataset spec

- **Systems:** ~30–50k complexes = reused PINDER/PLINDER/MISATO/PDBbind **+ your own RFD3 + latent-generator outputs + hard decoys** (near-native docking poses, chain-swapped cognate partners, backbone perturbations — **not** random mispairs, which teach a shortcut).
- **Geometries:** ~200k–400k spanning **equilibrium → near-clash** (perturbation ladder).
- **Labels per geometry:** MLFF (MACE-OFF **and** AIMNet2) E+forces on all; analytic validity terms (bond/angle/clash/chirality/planarity/omega/Cβ) on all; GFN2-xTB E+forces on a large CPU subset; **~2–4k GPU-DFT anchors** (held-out slice = key eval); Rosetta InterfaceAnalyzer ΔG on all + ~20k cartesian_ddg ΔΔG; ~10k co-folded ipSAE/PAE.
- **Splits (leakage-aware):** MMseqs2 ≤30% sequence clustering **and** interface-cluster holdout; adopt PINDER/PLINDER splits; hold out **whole clusters**; a **held-out DFT-anchor set** (force-accuracy/OOD eval) and a **held-out generator-target set** (steering eval), neither seen in training; ensure co-folded structures aren't also in the structure-training pool.

### 4. Steering mechanics

#### 4a. The three injection points (the central axis)

| Point | Where | Gradients? | Verdict |
|---|---|---|---|
| **(A) Gradient on noisy x_t** | add λ∇_{x_t}E(x_t) to the score/drift (classic classifier guidance) | yes | **Rejected** for physics heads — an MLIP/MLFF is meaningless on noisy, off-manifold x_t |
| **(B) Gradient through x̂₀** (Tweedie / DPS / reconstruction guidance) | evaluate E on the denoised estimate x̂₀(x_t) and backprop: λ∇_{x_t}E(x̂₀) | yes (one denoiser pass; double-backward if training the coupling) | **Correct entry point** — x̂₀ is approximately clean even when x_t is noise |
| **(C) Particle / population level** | evaluate E on x̂₀ (or final x₀); resample/select trajectories | no | For **expensive non-differentiable** signals (true MLIP relax, true Rosetta ΔG, ipSAE) |

These are **composable**: gradient guidance (B) on cheap differentiable signals + particle resampling (C) on expensive black-box signals in the same run.

#### 4b. Methods on each point
- **Time-gating:** zero the physics term at high noise, ramp it on over the last **~20–30%** of denoising steps as x̂₀ approaches equilibrium (DeltaDiff-style schedule).
- **Feynman–Kac / SMC steering:** N particles, resampled by a reward "potential" at chosen steps (Hartman "difference" potential on RFdiffusion is the template); lets an arbitrary reward steer; best when high-reward samples are rare.
- **Best-of-N / beam / MCTS:** simplest version — generate N, score each with the head, keep the best; strong trivially-parallel baseline.

#### 4c. The noisy-x̂₀ problem and its five mitigations
MLIPs/MLFFs are trained near equilibrium; mid-diffusion x_t (and even x̂₀ early) is off-manifold, where energies/forces are OOD and unreliable ("softening"). Stacking fixes:
1. **Evaluate only on x̂₀, never x_t** — the single most important mitigation.
2. **Time-gated guidance** — apply only in the last ~20–30% of steps.
3. **Use forces, not energies** (or a smoothed surrogate) — energies fluctuate too violently across the trajectory to guide stably; forces are smoother.
4. **Train the surrogate on diffusion-noised geometries** — sample x_t, compute x̂₀, label with the true MLIP/MLFF on x̂₀, train the cheap surrogate to be accurate *exactly where queried* (directly fixes OOD; this is what the perturbation ladder feeds).
5. **Uncertainty-gated guidance** — surrogate emits u(x); down-weight the term where it flags OOD (evidential/Bayesian UQ).
   - *Plus a structural fix:* a **learned-MM-parameter FF (Grappa/Espaloma)** is robust on clashes *by construction* (analytic LJ), so the always-on feasibility term sidesteps softening entirely; reserve QM-MLIP/SO3LR for low-noise refinement.

#### 4d. Multi-objective combination
Physics energy (kcal/mol, large dynamic range), ipSAE/PAE (bounded), and ΔG are **non-commensurable**. Combine by **rank or z-score inside the particle-resampling scheme**, not raw weighted gradient sums (which let the kcal/mol term swamp bounded confidence). For gradient guidance, **z-score/clip per-term gradient norms**.

#### 4e. Distillation & surrogate
Calling a heavy MLIP + backward per step per particle over hundreds of steps is prohibitive. **Distill** the expensive MLFF/MLIP (energy and/or forces) into a small equivariant surrogate that (a) is differentiable in one pass, (b) is trained on **noised** geometries, (c) emits uncertainty. Reserve the true MLIP/SO3LR for final rerank only (zero OOD risk on clean x₀).

#### 4f. The 4-tier integration recipe (assembled)
1. **Tier 1 — distill** (surrogate on noised geometries, uncertainty-aware); confidence head kept as-is.
2. **Tier 2 — x̂₀ gradient guidance, time-gated**, uncertainty- and z-score-weighted.
3. **Tier 3 — Feynman–Kac rank-combined reranking** with the true signals.
4. **Tier 4 — best-of-N final relax + hard gate** (PoseBusters/MolProbity/cis-peptide/chirality).
   - Build order: Tier 1 + Tier 4 first (low-risk best-of-N), then Tier 3, then Tier 2.

### 5. Training curriculum

| Stage | Data | Objective | Notes |
|---|---|---|---|
| **1. SSL pretrain trunk** | reused structures (or ATOMICA pretrained weights) | masked-atom modeling + coordinate denoising (≈ learning a force field) | gives a rich interface representation before labels |
| **2. Supervised energy+force + confidence** | OMol25 / SPICE-2 / DES370K / Splinter / PANIP **+ perturbation-ladder MLFF labels** | energy MSE + **force MSE** (+ SAPT-component loss where available) + analytic validity-energy terms; confidence head on ipSAE/PAE | **train on diffusion-noised geometries** so the head is accurate on x̂₀ |
| **3. Delta-learning calibration** | ~2–4k GPU-DFT hero anchors | fit GFN2-xTB→DFT correction; de-bias the MLFF teacher | **report teacher-vs-DFT divergence as a result, not a footnote** |
| **4. Affinity / ΔG head fine-tune** | SKEMPI / PPB-Affinity / PDBbind + Rosetta cartesian_ddg | **ranking loss** (not MSE — labels are ranking-grade), leakage-aware splits | small head on an already-good representation → ~8k affinity points suffice |
| **5. Validate against hard gate** | held-out targets + held-out DFT-anchor + held-out generator-target sets | PoseBusters / MolProbity / cis-peptide / chirality pass-rates; correlation with FEP/Rosetta; decoy discrimination; enrichment; diversity/novelty | confirm the smooth validity energy yields categorical validity; **anti-reward-hacking: evaluate with independent oracles never used in training/guidance** |

**Loss-design rule:** never mix label types in one regression target (QM forces, Rosetta ΔG, MM-GBSA, experimental Kd are different scales/fidelities) — separate heads, ranking losses for ranking-grade labels.

---

# Part III — Ideas Ledger, Novelty & Decisions

### 1. Candidate contributions ledger (C1–C5)

| ID | One-sentence claim | Novelty | Key risk | Feasible on 2×A100 + 1-wk data-gen? | Verdict |
|----|--------------------|---------|----------|--------------------------------------|---------|
| **C1** | Predict co-folder confidence + interface quality (ipSAE/PAE/ipTM/pDockQ/ΔG) from a backbone in one cheap pass, to steer without re-folding. | Weak — shadowed by AF3Score, AlloGen, BindEnergyCraft; ipSAE is a deterministic function of PAE. | Optimizes the metric reviewers trust least (confidence ≠ specificity); reward-hack bait. | Yes | **Fold into C2** as the "quality" channel; not a standalone paper. |
| **C2** | Distill expensive interface energetics into a small SE(3) surrogate trained on diffusion-*noised* geometries, uncertainty-gated, that steers multiple frozen generators on x̂₀ with no per-generator retraining. | **Strong** — owns the empty quadrant (learned + differentiable + in-loop-cheap + OOD-robust-on-noised-x̂₀ + generator-agnostic transfer + protein-complex interface). | Train/deploy distribution mismatch; teacher-accuracy ceiling. | Yes — exactly what the week produces. | **WINNER. Lock.** |
| **C3** | Use an MLFF as the guidance potential / train a new biomolecular MLFF. | Using one: incremental (DiffForce/Lai/Elign shadow). Building one: large but out of scope. | Training a new MLFF is CPU-QM-bound (OMol25 ≈ 6.6B core-h) — infeasible in 2.5 mo. | Standalone: **no.** As C2's teacher: yes. | **Not standalone; becomes C2's teacher** (Grappa/Espaloma/SO3LR). |
| **C4** | Mask Kd, predict Kd bins from random interface atoms only. | Low — fails as stated (data scarcity, interface≠ΔG, hot-spot/O-ring, random-atom info loss, OOD collapse). | Signal doesn't exist in the inputs. | N/A | **Rejected as stated; survives only as an SSL pretext.** |
| **C5** | One trunk, one head emitting confidence + physical-validity + interface energy, differentiable, generator-agnostic. | Highest ("no model does both from one trunk") but ≈2 papers of scope. | Each component under-evaluated; reviewers punish "two things done shallowly." | Not by late-Jan. | **Right 2-year program; wrong Jan-2027 submission. Defer.** |

**C1** — a faster confidence/ipSAE regressor; crowded and incremental. Its value is as C2's auxiliary "interface-quality" reward channel, not a headline.
**C2** — the learned, portable, in-loop physics steerer. Its ML contribution is a *mechanism* (distill-onto-noised-manifold + uncertainty-gated x̂₀ guidance + cross-generator transfer) general beyond proteins.
**C3** — a learned biomolecular force field is the *teacher/potential* C2 distills and the sparse high-fidelity refiner, not a contribution on its own here.
**C4** — dies on information-theoretic grounds; keep masked-atom modeling only as encoder pretraining.
**C5** — the union of C1+C2; the destination after C2 lands.

### 2. The locked recommendation — C2

> **A learned, generator-agnostic, re-folding-free, differentiable physics-steering head that distills expensive interface energetics (biomolecular force field + Rosetta) into a small SE(3)-equivariant surrogate trained on diffusion-*noised* geometries and gated by its own uncertainty, so it is accurate on the denoised estimate x̂₀ where guidance acts — and, trained once, steers two structurally distinct frozen generators toward better held-out interface binding energetics and near-perfect physical validity, without per-generator retraining.**

Why it beats the rest: C1 is incremental, C3 is infeasible/teacher-only, C4 fails, C5 is too big for the deadline. C2 is the only option that is simultaneously novel, feasible on the hardware/time budget, and exactly matched to what the 1-week data-gen can manufacture.

**Two framing rules (non-negotiable):**
- **R1 — Headline is interface *binding energetics*, not validity.** Closed-form validity projection (arXiv 2610.07037) already reaches 97–100% PB-valid with *no learning*; selling validity is dead on arrival. Validity is a constraint term + hard gate only.
- **R2 — The novelty is the *mechanism*, not "force-field guidance on x̂₀."** FF/MLFF-gradient guidance already exists (below). We claim the *distillation-onto-noised-manifold + uncertainty gating + generator-agnostic transfer* combination.

### 3. Novelty positioning — the papers to beat

**(a) Mechanism precedents (FF/MLFF-gradient guidance of a generator):**

| Paper | What it does | How C2 differs |
|-------|--------------|----------------|
| DiffForce (arXiv 2406.05832, 2024) | Differentiable FF-energy gradients on the denoised sample. | Hand-crafted FF, single generator; no distillation/OOD/transfer. |
| Lai et al. (J. Cheminf. 2026) | Training-free MMFF94 guidance in SemlaFlow/EDM. | Protein–**ligand**, classical FF, single generator. |
| Elign (arXiv 2601.21985, Jan 2026) | Post-training *alignment* of equivariant diffusion with a pretrained MLFF. | Alignment/fine-tuning, not in-loop guidance; not complex-interface; not generator-agnostic. |
| Inference-time ensemble (arXiv 2602.24007, Mar 2026) | AF3 + FF energy *reweighting* for ensembles. | Reweighting, not per-step guidance. |

**(b) Signal precedents (learned interface-energy / ΔG-from-backbone):**

| Paper | What it does | How C2 differs |
|-------|--------------|----------------|
| Kienlein et al. (bioRxiv 2026-09-24) | Transferable interface-energetics maps. | Prediction only, not a denoiser-steering gradient. |
| PreFold-dG (Bioinformatics 2026-08) | PPI ΔG from folding-model intermediates. | Prediction only. |
| AlloGen (arXiv 2606, 2026-06) | Differentiable (allosteric) interface scorer to steer generation. | Single-property/generator-specific; not a physics energy. |
| RFOptimization (Baker, bioRxiv 2026-09-09) | Guides design optimization with all-atom structure prediction. | Optimization against a predictor, not a learned physics gradient. |
| Didi latent search (bioRxiv 2026-09-18) | Test-time latent search / generative-pretraining compute. | Search/reranking, not in-loop differentiable physics. |

**(c) Confidence successors to watch (do not headline on these):** Badonyi & Toth-Petroczy ipSAE-successor (Protein Science 2026-07); reactifpTM (bioRxiv 2026-08); Vilya-2 (arXiv 2026-07).

**The empty quadrant we claim:** *learned + differentiable + in-loop-cheap-via-distillation + OOD-robust-on-noised-x̂₀ + generator-agnostic transfer + protein-complex interface binding.* No single prior work occupies all six axes; each precedent misses ≥1.

### 4. Key technical decisions already made

| Decision | Rationale |
|----------|-----------|
| **Always-on term = learned-MM-parameter FF (Grappa, else Espaloma-0.3)** via DMFF + learned GB implicit solvent. | MM form is robust on near-clash geometry *by construction* (analytic LJ wall), MM-speed, differentiable, auto-parameterizes arbitrary interface chemistry. |
| **QM-MLIP / SO3LR = sparse refinement/rerank only** (low-noise steps, final pose). | **Softening problem** (Zhong et al. 2024): QM-MLIPs underpredict PES curvature off-equilibrium → wrong/too-soft forces on exactly the near-clash x̂₀ a generator emits. |
| **Score x̂₀, never x_t.** | Tweedie/DPS reconstruction guidance; x̂₀ is near-physical even when x_t is noise. |
| **Distill onto diffusion-*noised* geometries.** | Makes the surrogate accurate where it's queried; mitigates train/deploy mismatch. |
| **Headline = binding; validity = constraint + hard gate** (PoseBusters/MolProbity/cis/chirality). | R1 above. |
| **Time-gated guidance (last ~20–30% of steps) + uncertainty gating + FK/SMC rank-combined reranking.** | Avoids high-noise bias; composes differentiable + non-differentiable signals; handles non-commensurable scales. |
| **FEP killed for generation** (≤3 edges/week); reuse published FEP as an independent eval anchor only. | ~120 GPU-h/PPI edge. |
| **Co-folding minimized** to ~10k targeted folds. | PAE/ipSAE largely reusable free from AFDB/PINDER; GPUs redirected to the near-clash interface-energy anchor. |
| **Ranking losses, not regression,** for ranking-grade Rosetta/MM-GBSA; don't mix label types in one target. | Label-noise realism. |

### 5. Open decisions

| Open decision | Options | Lean |
|---------------|---------|------|
| **Second generator (for transfer)** | Proteína-Complexa / La-Proteina (latent) vs Boltz-design. | A latent generator maximizes the "structurally distinct" claim; pick one latent + RFD3. |
| **Trunk** | Fresh EquiformerV2/MACE (low L_max) vs ATOMICA pretrained interface weights. | Prototype both; ATOMICA de-risks the interface prior, Equiformer/MACE de-risks force fidelity. |
| **Interface-energy target** | SAPT-component decomposition vs single scalar. | SAPT components more informative but need Splinter-style labels; decide by eval gain. |
| **PAE-matrix head** | Emit full N×N PAE vs interface scalars only. | Defer unless the confidence channel needs it; adds cost. |

### 6. Timeline reality

- **ICLR 2027: closed.** Not an option.
- **ICML 2027: ~late-Jan submission, ~2.5 months out.** Budget: one crisp mechanism + two-generator transfer + ablation ladder. No unified multi-head system.
- **ICLR 2028: clean fallback** if transfer slips.
- **Descope rule:** cut breadth (DFT hero anchor → xTB-only, drop SO3LR refinement, one latent generator) **before** cutting two-generator transfer — transfer *is* the novelty.

### 7. The experiment matrix

**Headline experiments (3):**
1. **Generator-agnostic transfer (money figure):** train once; plug into RFD3 (all-atom) + a latent generator; show improved held-out interface ΔG/ipSAE/designability on both, **zero per-generator retraining.**
2. **Mechanism ablation ladder (the ML contribution):** raw MLIP guidance vs distilled surrogate; clean- vs noised-manifold training; x̂₀ vs x_t; uncertainty-gate on/off; time-gate on/off; MM-param feasibility term on/off — each shown necessary.
3. **Beats the right baselines on binding, not validity:** vs unguided; vs closed-form validity projection (2610.07037); vs Hartman FK-with-Rosetta; vs BindEnergyCraft / DiffForce / Lai MMFF94.

**Anti-reward-hacking protocol (centerpiece):** evaluate with **independent held-out oracles** (an un-distilled co-folder, reused-FEP anchor, Rosetta protocol variant); report guided-vs-independent-metric gap; report **designability AND diversity/novelty** (no mode collapse); decoy discrimination + enrichment curves; leakage-aware cluster splits. This is the accepted ICML in-silico bar without a wet lab.

---

## Appendix — document map

| Doc | Role |
|---|---|
| `steering-model-compendium.md` (this) | Organized index of all ideas: modules, data, decisions. Start here. |
| `research-plan-icml2027.md` | The locked single-contribution plan + build timeline to the deadline. |
| `mlip-physics-steering-head-plan.md` | Deep dive on the MLIP/physics head design and its data tiers. |
| `steering-head-atom-interaction-literature.md` | Raw literature survey underpinning all of the above. |
