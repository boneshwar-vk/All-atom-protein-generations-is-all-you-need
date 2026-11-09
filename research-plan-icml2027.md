# Research Plan — Learned Physics-Steering Head for Protein-Complex Generators

**Target venue:** ICML 2027 (≈ late-Jan 2027 abstract/submission) · fallback **ICLR 2028** (ICLR 2027 has already closed)
**Compute:** ≤ 2× A100 + CPU node · **1 week (~336 GPU-h) dedicated data generation** + Rosetta on CPU
**Status:** planning · synthesizes a 12-agent literature + critique sweep (see `steering-head-atom-interaction-literature.md`, `mlip-physics-steering-head-plan.md`)
**Date:** 2026-11-09

---

## 0. Timeline reality (drives all scoping)

Today is 2026-11-09. An ICML-2027 January deadline is **~2.5 months out**. On 2 A100s in that window you can land **one crisp mechanism with strong in-silico evaluation on ≥2 generators** — *not* a unified multi-head system. ICLR 2027 is closed; the real ICLR fallback is 2028. **Plan for the tight case.** If anything is cut, cut breadth before cutting the two-generator transfer experiment — transfer is the novelty.

---

## 1. The locked contribution (one sentence)

> **A learned, generator-agnostic, re-folding-free, differentiable physics-steering head that distills expensive interface energetics (biomolecular force field + Rosetta) into a small SE(3)-equivariant surrogate trained on diffusion-*noised* geometries and gated by its own uncertainty, so it is accurate on the denoised estimate x̂₀ where guidance acts — and, trained once, steers multiple structurally distinct frozen generators (RFdiffusion3 + a latent generator) toward better held-out interface binding energetics and near-perfect physical validity, without per-generator retraining.**

Headline = **interface binding energetics**. Physical validity = a **constraint term**, never the headline (closed-form validity projection, arXiv 2610.07037, already reaches 97–100% PB-valid with no learning — selling validity is dead on arrival).

This is Critic-selected **C2** over: C1 (re-folding-free confidence head — incremental; shadowed by AF3Score/AlloGen/BindEnergyCraft), C3 (train a new MLFF — infeasible in 2.5 months; QM is CPU-bound), C4 (masked-Kd-from-random-atoms — fails as stated), C5 (unified confidence+physics oracle — the right 2-year program, too big for January).

---

## 2. The energy the head emits (one objective, two terms)

`E_head(x) = E_feasibility(x)  +  λ · E_interface(x)`

- **E_feasibility — physical-feasibility / clash / geometry regularizer (always-on, every step).**
  Implemented with a **learned-MM-parameter force field** — **Grappa** (best reported accuracy; graph-attention transformer predicting MM params, outperforms Espaloma at MM speed) or **Espaloma-0.3** — evaluated differentiably via **DMFF** (JAX, PME long-range, optional polarizable AMOEBA) + a **learned implicit-solvent GB term** (Airas–Zhang ESM3-GNN+GB, 2026) because interface ΔG is desolvation-dominated. *Why MM-param, not a QM-MLIP, for the always-on term:* the **softening problem** (Zhong et al. 2024) — QM-MLIPs underpredict PES curvature off-equilibrium, so they misbehave on exactly the near-clash x̂₀ a generator produces; a learned-MM-parameter FF is robust there by construction (analytic LJ wall + harmonic bonds), differentiable, MM-cheap, and **auto-parameterizes arbitrary interface residues / ligands / non-canonical chemistry from the graph** (no atom-typing). This is the term that makes per-step steering stable.

- **E_interface — interface binding energetics (the headline signal).**
  A small **SE(3)-equivariant surrogate head** (shared all-atom trunk; EquiformerV2/MACE-class low-L_max, or ATOMICA's pretrained interface weights) regressing an **expensive interface energy label** (Rosetta InterfaceAnalyzer ΔG + xTB/MLFF interaction energy + reused experimental ΔG), **trained on diffusion-noised geometries** and emitting an **uncertainty** u(x). This is the term MM cannot capture well (true binding, desolvation, polarization) and the term that drives the money metric.

- **High-fidelity refinement (sparse, optional):** **SO3LR** (the one biomolecular neural FF with explicit long-range electrostatics + dispersion + explicit-solvent validation at ~200k atoms/GPU) or the QM-MLIP, used only at low-noise steps / final rerank where x̂₀ is already near-physical and its off-equilibrium fragility is irrelevant. Also the natural head-to-head benchmark.

---

## 3. Mechanism (where it enters the sampler)

Frozen generator; inference-time steering only.
1. **x̂₀-only evaluation** — never on noisy x_t (Tweedie/DPS reconstruction guidance).
2. **Time-gated gradient guidance** — `g = ∇_{x_t}[E_feasibility(x̂₀) + λ·E_interface(x̂₀)]` ramped on over the last ~20–30% of denoising steps; per-term gradient norms z-scored/clipped; E_interface down-weighted by its uncertainty u.
3. **Feynman–Kac / SMC reranking** — N particles resampled by a **rank-combined** reward {E_interface, confidence, Rosetta ΔG} at a few checkpoints (handles non-commensurable scales; lets non-differentiable true signals steer).
4. **Best-of-N final relax + hard gate** — PoseBusters / MolProbity / cis-peptide / chirality as a non-differentiable acceptance gate on clean x₀.

The **ML contribution is the surrogate mechanism**: distillation-onto-noised-manifold + uncertainty-gated x̂₀ guidance + generator-agnostic transfer — *general beyond proteins* (steering diffusion with expensive, OOD-fragile objectives).

---

## 4. Novelty positioning — the papers to beat

"Force-field-gradient guidance of a generator" and "learned interface-energy prediction" each already exist **separately**. The defensible novelty is their **coupling under off-equilibrium robustness + generator-agnostic transfer, on protein COMPLEX interfaces**. Position explicitly against:

**(a) FF/MLFF-gradient guidance of generators (the mechanism precedents):**
- **DiffForce** (arXiv 2406.05832, 2024) — differentiable FF-energy gradients on the denoised sample. *Differs:* general/antibody, hand-crafted FF, single generator, no distillation/OOD/transfer.
- **Lai et al.** (J. Cheminformatics 2026) — training-free MMFF94 guidance in SemlaFlow/EDM. *Differs:* protein–**ligand**, classical FF, single generator.
- **Elign** (arXiv 2601.21985, Jan 2026) — post-training **alignment** of equivariant diffusion with a pretrained MLFF. *Differs:* alignment/fine-tuning (not in-loop guidance), not complex-interface steering, not generator-agnostic.
- **Inference-time ensemble** (arXiv 2602.24007, Mar 2026) — AF3 + FF energy **reweighting** for ensembles. *Differs:* reweighting, not per-step guidance.

**(b) Learned interface-energy / ΔG-from-backbone heads (the signal precedents):**
- **Kienlein et al.** (bioRxiv 2026-09-24) — transferable interface energetics maps. *Differs:* prediction, not a denoiser-steering gradient.
- **PreFold-dG** (Bioinformatics 2026-08) — ΔG of PPIs from folding-model intermediates. *Differs:* prediction only.
- **AlloGen** (2026-06), **RFOptimization** (Baker, bioRxiv 2026-09-09), **Didi latent search** (bioRxiv 2026-09-18). *Differ:* generator-specific or reranking/search, not a learned differentiable in-loop physics gradient.

**(c) Confidence successors to watch (do not build on these as the headline):** Badonyi ipSAE-successor (Protein Science 2026-07), reactifpTM (2026-08), Vilya-2 (2026-07).

**The empty quadrant we claim:** learned + differentiable + in-loop-cheap-via-distillation + **OOD-robust on noised x̂₀** + **generator-agnostic transfer** + **protein-complex interface binding**. No single prior work occupies it.

---

## 5. The 1-week data-generation plan (~336 GPU-h + CPU)

**Principle:** spend GPUs only where no database exists. The free thing (foundation-MLFF forces, ~40 GPU-h for the whole corpus) is *not* the novelty; co-folding is the GPU hog that buys mostly-reusable confidence labels; the **paper-defining signal is interface energetics on generator-distribution, near-clash geometries.**

**Reuse (0 GPU):** OMol25 / SPICE-2 / DES370K / Splinter / PANIP (MLFF side); PINDER / PLINDER / MISATO / PDBbind (systems); AFDB/PINDER PAE (confidence); SKEMPI / PPB-Affinity / PDBbind (experimental ΔG/ΔΔG, reuse only).

**Generate this week:**
- **Systems:** ~30–50k complexes = reused PPI/PL + **your own RFD3 + latent-generator outputs + hard decoys** (near-native docking poses, chain-swapped partners, backbone perturbations — *not* random mispairs).
- **Perturbation ladder (the core idea):** each complex → 5–10 steps from equilibrium → side-chain repack → backbone noise → **near-clash** = **~200k–400k geometries** spanning the regime where MLFFs break and steering operates. OpenMM MD + perturbation, ~60 GPU-h.
- **Bulk labels (cheap):** MACE-OFF **and** AIMNet2 energy+forces on all (~40 GPU-h); analytic validity terms (bond/angle/clash/chirality/planarity/omega/Cβ) on all.
- **Hero anchor:** **GPU-DFT (GPU4PySCF, ωB97X-D/def2-SVP) on ~2–4k ≤50-atom near-clash/charged interface clusters** (~90 GPU-h) → de-biases the MLFF teacher; **held-out slice = the key force-accuracy / OOD eval**. GFN2-xTB (CPU) on a large subset.
- **Targeted co-fold:** **~10k** generator-distribution + hard-decoy complexes (MSA-free, ~120 GPU-h) → ipSAE/PAE AFDB can't cover. (Down from the earlier 30–50k — the rest is reused.)
- **Rosetta (CPU, parallel, free of the GPU budget):** InterfaceAnalyzer score-only ΔG on all; **cartesian_ddg ~20k mutations** for a ΔΔG channel + held-out ΔΔG eval. (Not Flex-ddG; no FEP — ≤3 edges/week, reuse published FEP as an independent eval anchor only.)

**Splits:** MMseqs2 ≤30% sequence clusters + interface-cluster holdout; adopt PINDER/PLINDER splits; **held-out DFT-anchor set** (force accuracy) and **held-out generator-target set** (steering eval), neither seen in training.

**Day-by-day:** D1–2 assemble reuse + generator outputs + decoys, run perturbation MD (~60) + MLFF labeling (~36); D3–4 co-fold ~10k (~120), CPU cartesian_ddg; D5–6 GPU-DFT hero anchor (~90) + CPU xTB; D7 (~30 buffer) QC, carve held-out evals, finalize splits.

---

## 6. Experiment matrix (what makes it a strong paper)

**Headline experiments (3):**
1. **Generator-agnostic transfer (the money figure):** train the head once; plug into **RFdiffusion3 (all-atom)** + **a latent generator (Proteína-Complexa / La-Proteina or Boltz-design)**; show improved held-out interface ΔG / ipSAE / designability on **both, with zero per-generator retraining.**
2. **Mechanism ablation ladder (the ML contribution):** raw foundation-MLIP guidance vs distilled surrogate; clean-trained vs **diffusion-noised-trained**; x̂₀ vs x_t; uncertainty-gating on/off; time-gating on/off; MM-param feasibility term on/off. Each shown necessary.
3. **Beats the right baselines on binding, not just validity:** vs unguided; vs **closed-form validity projection (2610.07037)**; vs **Hartman FK with raw Rosetta**; vs **BindEnergyCraft / DiffForce / Lai MMFF94**. Show the learned interface energy adds binding quality that validity-only and confidence-only methods don't.

**Anti-reward-hacking protocol (centerpiece, not a footnote):** evaluate with **independent held-out oracles** — a co-folder never distilled from, a small reused-FEP anchor, a Rosetta protocol variant. Report guided-metric vs independent-metric gap; report **designability AND diversity/novelty** (no mode collapse); decoy discrimination + enrichment curves. This is the accepted ICML in-silico bar without a wet lab.

---

## 7. Risks & descoping

- **Train/deploy distribution mismatch** (the central risk): mitigated by the **perturbation ladder** (label the near-clash geometries) + the MM-param feasibility term (robust there by construction) + uncertainty gating.
- **Teacher-accuracy ceiling:** MLFF caps at the teacher; MACE-OFF short-range fails on charged interfaces → use AIMNet2 (charged) + the DFT anchor; **report teacher-vs-DFT divergence as a result.**
- **Reward hacking / mode collapse:** independent oracles + diversity metrics (above).
- **Decoys / label noise / leakage:** hard decoys; **ranking losses** for ranking-grade Rosetta/MM-GBSA labels (don't mix label types in one regression target); cluster-level splits.
- **Descope order if time-constrained:** keep transfer (exp 1) and the noised-manifold ablation (exp 2) at all costs; drop the DFT hero anchor to xTB-only, drop SO3LR refinement, reduce to one latent generator — never drop two-generator transfer.

---

## 8. Build timeline to late-January

- **Wk 1:** data-generation week (§5). In parallel: implement Grappa/Espaloma-via-DMFF feasibility term + the x̂₀ guidance harness on a frozen RFD3.
- **Wk 2–3:** train the interface-energy surrogate on noised geometries; wire uncertainty + time-gating; Tier-1+Tier-4 best-of-N working on RFD3.
- **Wk 4–5:** second generator (transfer); FK reranking; ablation ladder.
- **Wk 6–7:** baselines (projection, FK-Rosetta, DiffForce/BindEnergyCraft), independent-oracle eval, diversity metrics.
- **Wk 8–9:** writing, figures, buffer.

### One-line summary
Ship **one** learned differentiable steering head — **Grappa/Espaloma feasibility term (robust on clashes) + a distilled, uncertainty-gated, noised-manifold-trained interface-energy head (the binding headline)** — and prove it **transfers across two frozen generators** with an anti-reward-hacking eval; the 1-week data budget exists to manufacture the near-clash, generator-distribution, QM-anchored interface-energetics regime that makes it work.
