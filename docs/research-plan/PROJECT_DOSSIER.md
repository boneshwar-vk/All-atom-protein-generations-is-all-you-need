# Project dossier — Identity, Cause, and Truth-vs-Confidence in biological foundation models

A single self-contained record of the project: the background concepts, the state of the field, how the
idea was arrived at (and what was ruled out), the final plan, the first experimental result, the risks,
and what remains to be done. Written to stand on its own.

**Status:** planning complete; Phase 0 (synthetic) run and passed. **Owner:** generated with Claude Code on
the `claude/sharp-bell-3hmd24` branch. **Date of this version:** 2026-11-22.

> Convention: claims are marked **UNVERIFIED** when they were not confirmed against a primary source in this
> work (network access to some sites was blocked during preparation). Treat those as to-be-checked.

---

## Table of contents
1. [Executive summary](#1-executive-summary)
2. [Background concepts, in plain terms](#2-background-concepts-in-plain-terms)
3. [State of the field](#3-state-of-the-field)
4. [How the idea was reached (and what was ruled out)](#4-how-the-idea-was-reached-and-what-was-ruled-out)
5. [The project definition](#5-the-project-definition)
6. [The six claims](#6-the-six-claims)
7. [Step-by-step plan](#7-step-by-step-plan)
8. [Kill-gate ladder](#8-kill-gate-ladder)
9. [Controls and baselines](#9-controls-and-baselines)
10. [Datasets, models, tools](#10-datasets-models-tools)
11. [Phase 0 result](#11-phase-0-result)
12. [Prior art to differentiate from](#12-prior-art-to-differentiate-from)
13. [Risks and scoop exposure](#13-risks-and-scoop-exposure)
14. [Ethics and dual-use](#14-ethics-and-dual-use)
15. [Minimum publishable result](#15-minimum-publishable-result)
16. [Open items and next actions](#16-open-items-and-next-actions)
17. [References](#17-references)
18. [Repository file index](#18-repository-file-index)

---

## 1. Executive summary

**The question.** Biological foundation models (protein and genomic models) hold, inside their activations,
quantities whose true values are known exactly from physics and chemistry — a backbone dihedral angle is a
point on a circle, two together are a point on a torus, the twist of a helix advances at a fixed rate, and the
binding strength of two molecules has a laboratory-measured value (Kd). This exact external ground truth is the
asset that interpretability of ordinary language models does not have. We use it to ask three things that the
existing literature has not answered together, for such a quantity:

1. **Identity** — is the internal feature we recover actually *that* physical quantity (up to an unimportant
   rotation of axes, i.e. "up to isometry"), not merely some directions that happen to reconstruct the activation?
2. **Cause** — if we change that feature inside the model, does the output move in the matching way, including
   returning to its starting value after a full circle for an angular quantity?
3. **Truth vs confidence** — does the model keep the real physical quantity in a *different* internal direction
   from its own self-confidence about that quantity?

**The contribution** is causal + identifiable + truth-vs-confidence analysis on exact physical ground truth.
Explicitly **not** a contribution: proposing a fancier dictionary geometry (sphere/hyperbolic), because that
space is already occupied (see §12).

**Scope.** One paper. The spine is the protein track (ESM3-open, structural angles). A binding track (a
co-folding model, Kd vs the model's interface-confidence score ipSAE) is an extension that generalises the hook.

**First result.** A pure-numpy calibration test (Phase 0) confirms the predicted behaviour: a curved code ties a
flat block of equal dimension at matched information budget (so geometry is a control, not the headline), the
angle is recovered up to isometry, and single-direction features pay a covering-number penalty. It also surfaced
an honest caveat: recovering a *torus* (two coupled circles) needs a real factorization method, not naive PCA.

---

## 2. Background concepts, in plain terms

No analogies; each term defined on first use.

**Activation (representation).** When a network reads an input, it produces at every layer and position a list
of numbers. That list is the activation. Interpretability studies what information those numbers carry and
whether the model uses it.

**SAE (sparse autoencoder).** A helper network trained to rewrite each activation as a sum of a few features,
where each feature is a single fixed direction multiplied by a strength, and only a few features are active at
once. Good for "this concept is present and this strong"; a single feature is one direction, so it cannot by
itself smoothly represent a quantity that varies continuously and wraps around.

**BSF (block-sparse featurizer).** An SAE variant in which each feature is a small block of several directions
(default three) rather than one. An active feature is then a point in a small multi-dimensional space, which can
hold a continuous or cyclic quantity. Only a few blocks are active at once (sparse across blocks), but within an
active block several coordinates may be non-zero (dense within a block). Trains on cached activations only.
Published on vision models.

**Jacobian lens.** Takes a middle-layer activation and estimates how it pushes the final output, by averaging
the derivative of the output with respect to that layer across many inputs, then reading the result through the
model's output layer. Shows what a middle layer is pointing toward in the output. Built for language models.

**J++ (improved Jacobian lens).** The same object with three fixes: cluster inputs and average within each
cluster instead of all together; change the backward computation to track the real network more faithfully; and
drop meaningless output tokens. Still one linear map per layer.

**ipSAE.** A score a structure model computes about an interface (the region where two molecules touch) to state
how confident *it* is. It is the model's opinion of itself, not a laboratory measurement, and despite the
letters it is **not** a sparse autoencoder (it stands for "Score from Aligned Errors", Dunbrack lab).

**Kd (dissociation constant).** A laboratory-measured number for how strongly two molecules bind; smaller means
stronger. Because it is measured, it is real physical ground truth, unlike the model's own confidence.

**Two ways to study a model.** (A) *Probing*: decide on a known quantity first and test whether it can be read
out of the activations with a simple classifier. (B) *Dictionary learning* (SAE/BSF): break the activations into
features without deciding anything in advance, then label the features afterwards. A property being readable
(decodable) does **not** prove the model uses it — this "decodable is not steerable" gap is shown on a real
structure model and is central to this project.

**Identifiability up to isometry.** "Isometry" is a distance-preserving change such as a rotation or reflection.
A feature identified "up to isometry" means the real shape and the real coordinate on it have been recovered,
except for an unimportant rotation of axes. Current methods recover *a* subspace that reconstructs the
activation, but not necessarily *the* physical coordinate.

**Wrap-around causal test.** For a quantity that lives on a circle (an angle), steer it all the way around and
check that the output returns to its starting value after a full turn. A single-direction feature cannot pass
this by construction.

**Persistence / commitment.** Generative biological models build their answer over several internal steps. An
edit made in the middle can be undone by later steps. Any "steer in the middle" plan must first check that
mid-way edits survive to the final output.

**Matched bits (fair comparison).** To compare two methods fairly you hold the information budget equal, so a
win reflects the method and not extra capacity.

---

## 3. State of the field

Mechanistic interpretability of biological foundation models is a fast-moving, mostly 2025–2026 literature. The
verified picture (full reading list in `FINAL_FINDINGS.md` and `appendix/`):

- **Protein language models** are the most studied: sparse autoencoders (InterPLM in *Nature Methods* 2025;
  Adams/AlQuraishi; Gujral/Berger in *PNAS* 2025; Matryoshka SAEs from Reticular), cross-layer transcoder
  circuits (ProtoMech), activation patching, and steering are all established on ESM-2 and increasingly ESM3.
- **Structure and co-folding models** (ESMFold, Boltz, RoseTTAFold) are a 2026 frontier: SAEs on trunk and pair
  representations (PairSAE on Boltz-2), and the key negative result that a property can be **decodable but not
  steerable** (Jedryszek on Boltz-1).
- **Genomic models**: SAEs on Evo 2 (layer 26 of the 7B model, with Arc and Goodfire), Nucleotide Transformer
  (Decode-gLM), and strong cautions that naive validation is confounded by base composition (Ali).
- **Single-cell models**: SAEs decompose activations, but a recurring finding (much of it from one author) is
  that these models capture co-expression rather than causal regulation.
- **Cross-cutting cautions**: features are often validated against annotation databases the model trained on
  (circular); random-initialised baselines and seed stability are rarely checked; feature "death" is severe on
  some structure models.

Two **open gaps** this project targets: (i) validation against **physically computed** quantities rather than
database labels, and (ii) **causal + identifiability** claims (does the model *use* the quantity, and did we
recover the *real* coordinate) rather than reconstruction or single-layer steering alone.

---

## 4. How the idea was reached (and what was ruled out)

The project narrowed through several rounds; recording the dead ends so they are not revisited.

**Starting point.** Apply two recent methods — the Jacobian lens / J++ (Anthropic) and BSF (Goodfire) — to
biological models (ESM3, AlphaGenome, Evo 2). Finding: neither had been applied to a biological model, but the
Jacobian lens does not transfer cleanly (it assumes a causal token vocabulary; genomic models have four tokens,
ESM3 is bidirectional, AlphaGenome has no token output). BSF transfers easily (cached activations only).
"AlphaGenome 2" is API-only with no weights, so it is unusable for this; "Bonbon" is a real but closed-weight,
sequence-only model and was dropped.

**Three initial tasks.** (1) SAE vs BSF plus alternative geometries; (2) compare which model is "better"; (3)
localise a property and steer early. Vetting: (2) is already published (Steenwyk; Whitfield) and folds into (1);
(3) needed reshaping and switches to a diffusion model, and "ipSAE" was clarified to be an interface-confidence
score, not an autoencoder.

**Four sharper pitches** were then proposed and independently vetted:
- *Learnable-curvature dictionary* (let the model reveal each concept's geometry): mechanism already exists
  (Skopek 2019) and is reported to be badly conditioned; open only narrowly.
- *Phase-velocity validator* (confirm a circular feature by its physical advance rate): the DNA codon leg is
  already published (two NeurIPS 2026 papers), and on proteins the model's positional encoding (RoPE) can fake a
  constant advance rate; only the structure-derived "phase slip" version survives.
- *Measure the capacity wall*: **already published in full** (SASA / Dal26, NeurIPS 2026 — the definition, the
  theorem, the curve, and the block remedy), and a flat block of equal dimension already ties a curved one.
- *ESM3 causal-efficacy steering*: two sub-claims were already contradicted (single-direction features do steer
  ESM3 — Steenwyk; and ESM3 has its own version of the "edits wash out" problem).

**What survived**, flagged independently by more than one vetting, is the **causal + identifiability** core, now
broadened to include binding: identity up to isometry, the wrap-around causal test, the truth-vs-confidence
separation, and the honest persistence/commitment finding. This is the project in §5.

---

## 5. The project definition

**Thesis (one sentence).** On quantities fixed by physics — backbone (φ,ψ) and helix phase in ESM3 (from DSSP),
and laboratory Kd in a co-folding model — a biological foundation model represents the quantity as an
identifiable, gauge-free coordinate that (i) matches the physical truth up to isometry on held-out data, (ii)
causally drives the output with a wrap-around test a single-direction feature cannot fake, and (iii) lives in a
direction distinct from the model's own calibrated confidence (pLDDT; ipSAE) — shown convergently for a periodic
structural and a scalar thermodynamic variable.

**Two tracks, one method.**

| | Track A (spine) | Track B (extension) |
|---|---|---|
| Model | ESM3-open (1.4B, masked generative, uses RoPE) | A co-folding model (Boltz; AF3/Chai alternates) |
| Quantity | backbone (φ,ψ) torus; helix phase; SASA as a non-periodic control | experimental binding strength Kd |
| Confidence contrast | the model's pLDDT / structure-token confidence | ipSAE / ipTM (interface self-confidence) |
| Ground truth from | experimental PDB structures via DSSP/mdtraj | laboratory Kd (PDBBind/PLINDER/BindingDB/SKEMPI) |
| Dual-use | low | capability-adjacent; auditing/defensive framing only |

ipSAE is always a **contrast**, never a steering target (steering toward the model's own confidence is circular).
A model-independent geometric interface descriptor (buried surface area, contacts) is kept as a third, distinct
target for Track B so the causal test does not depend on the model's own score.

**The intellectual hook.** *Does a biological model separate what is true from how sure it is?* This is
interesting beyond biology because biology is the one setting where both the truth (physics) and a deployed
confidence head are externally pinned, which lets us add the causal and identifiability dimensions that the
current language-model work on truth-vs-confidence cannot.

---

## 6. The six claims

Each claim lists the experiment, the figure it produces, the control that makes it bulletproof, and the prior
work it must out-position.

- **C1 — Identity up to isometry.** The recovered coordinate equals the physical truth on a held-out split.
  *Metric:* circular/toroidal correlation and geodesic R² after a train-fitted isometric alignment. *Control:*
  the nuisance-only baseline (§9). *Out-positions:* SASA (reconstruction only, no identity claim).
- **C2 — Wrap-around causal dose-response.** Steering the coordinate moves the output and it returns to baseline
  after a full 2π turn; a single-direction latent cannot. *Control:* matched-norm random direction; validity
  filter. *Out-positions:* Steenwyk (single-layer scalar steering, no circular closure).
- **C3 — Truth ⟂ confidence (the hook).** The truth coordinate and the model's confidence occupy distinct
  directions (principal angle > 0; a causal double dissociation). *Control:* confidence-matched negative pairs
  (vary true physics, hold confidence fixed) to kill the "one entangled signal" rebuttal. *Out-positions:* the
  language-model truth/confidence literature (no external truth, no causality, no identifiability).
- **C4 — RoPE-proof phase slip.** The feature tracks structure-derived irregularities (kinks, 3₁₀-helices) that
  positional encoding alone cannot manufacture. *Control:* trained-minus-random-init at matched position.
- **C5 — Persistence / commitment.** Whether mid-computation edits survive iterative decoding; reported honestly
  (likely a partial negative). *Extends:* the off-manifold-collapse and edit-persistence antecedents.
- **C6 — Cross-track invariance.** The same identity + cause + truth-vs-confidence pattern holds for a periodic
  structural variable (A) and a scalar thermodynamic one (B) — the answer to "cherry-picking".

---

## 7. Step-by-step plan

Phases are ordered cheapest-decisive-first. Each has a go/no-go gate; on a no-go the project pivots to a smaller
but still publishable result (§15) rather than dying.

### Phase 0 — Synthetic calibration / geometry kill-test (CPU, done)
Plant known shapes (circle, torus) in a high-dimensional space with noise; compare a flat block, a curved
(unit-circle) block, and single-direction codes at matched information budget; measure reconstruction, the
covering-number penalty, and whether the planted angle is recovered up to isometry. **Gate G0:** proceed only if
curved ties flat (so geometry is a control), the angle recovers, and single-direction codes pay a penalty.
Result in §11. **No weights or network needed.**

### Phase 1 — Persistence / commitment gate (the most likely death)
Add a known displacement to a middle-layer activation at target positions, run the rest of the model's
generation, and measure whether the final output reflects the edit versus a matched-random displacement, with a
validity filter (pLDDT/pTM/clash/refold) so a change is not off-manifold garbage. Test this **before** any
expensive steering. **Gate:** edits must persist above matched-random and stay on-manifold; otherwise the
persistence *negative* becomes a headline finding.

### Phase 2 — Identity + truth-vs-confidence (the core)
Cache ESM3 residual-stream activations on sequence-only input at several layers; compute (φ,ψ)/helix
phase/SASA from experimental structures; fit a circular/torus coordinate plus single-direction and probe
baselines; align the recovered coordinate to truth on a held-out split (C1); measure the principal angle between
the truth direction and the pLDDT-confidence direction and run the confidence-matched-pair control (C3). Track B:
leakage-aware splits, cache the co-folding model's representations, fit a Kd coordinate, and measure the
three-way angle among experimental Kd, ipSAE, and the model's own predicted affinity. **Note from Phase 0:** the
torus needs a genuine factorization (not PCA) to separate the two angles — build this here.

### Phase 3 — Cause (wrap-around) + RoPE-proof validator
Steer the recovered coordinate across at least eight targets including a full 2π wrap; measure the decoded
structural change and its closure; compare curved vs single-direction vs matched-random at equal norm (C2); run
the phase-slip panel on kinks and 3₁₀-helices (C4). Track B: monotone dose-response on buried surface area with
SKEMPI mutation agreement as a causal oracle, against a guidance-steering baseline. **Heaviest phase.**

### Phase 4 — Cross-track invariance, full sweep, write-up
Full multi-seed, matched-bits, random-init sweep on both tracks; stitch the truth-vs-confidence result across the
structural and thermodynamic variables (C6); freeze pre-registered predictions; write the paper.

**Compute (order-of-magnitude, UNVERIFIED):** Phase 0 CPU-only; Phase 1 a single 24 GB GPU for the protein track;
Phases 2–3 one to two 80 GB GPUs; Phase 4 a small cluster for the full ensemble (the core result needs far less).

---

## 8. Kill-gate ladder

Run probe-level gates to completion before launching any costly steering/decoding job (steering is 10–100× more
expensive and wasted if a probe gate fails).

```
G0 synthetic matched-bits + identifiability (CPU)  ||  G-B1 Kd-leakage probe
      -> G1 RoPE / positional control (probes only)
      -> G2 persistence / commitment (one decode cell)   <- most likely death
      -> G-B2 ipSAE-collinearity (is Kd just ipSAE?)
      -> G3 decodable != steerable + geometry lift
      -> G4 wrap-around dose-response
      -> G-B3 ipSAE-Goodhart + SKEMPI agreement
```

The three gates most likely to kill the project, earliest testable: **G2** (persistence — kills the causal half),
**G-B1** (Kd prediction is leakage, not learning), **G1** (RoPE manufactures the periodicity).

---

## 9. Controls and baselines

- **Nuisance-only baseline** (the single highest-value control): a decoder given only {positional / RoPE phase}
  ∪ {train-set membership + cluster id} ∪ {model self-confidence}. Every go-threshold is "beat this by a
  pre-registered margin." It pre-empts the RoPE, leakage, and confidence confounds at once.
- **Confidence-matched negative pairs** (vary true physics, hold confidence fixed) — run first inside Phase 2 to
  kill the "truth and confidence are one entangled signal" rebuttal.
- **Random-initialised same-architecture model** through the whole pipeline.
- **≥5 seeds** with a stability statistic; the stable block-selection variant.
- **Positional / local-window control run first** (RoPE can fake periodicity); **sequence-only** input for Track A.
- **Matched bits / MDL** comparisons, never matched feature count.
- **Matched-norm random-direction** control for every steering claim.
- **Leakage-aware time + scaffold splits** plus an explicit leakage probe for Track B.
- **Validity filter** (pLDDT/pTM/clash/ProteinMPNN refold) on every steered output.

---

## 10. Datasets, models, tools

- **ESM3-open** weights: gated, non-commercial; needs account acceptance. Exact terms and hidden width (~1536)
  **UNVERIFIED**.
- **Boltz**: open, MIT-licensed. AlphaFold3 / Chai are alternates.
- **ipSAE**: Dunbrack lab code (and a PyPI package); used only as a contrast.
- **PLINDER** (leakage-aware protein–ligand splits); **PDBBind / BindingDB / SKEMPI / CASF** for Kd and
  mutational ΔΔG. Exact counts and licences **UNVERIFIED** — confirm before use.
- **geoopt** (sphere / torus / hyperbolic manifolds); **DSSP / mdtraj** for angles and SASA; **ProteinMPNN** for
  refold-based validity.
- Phase 0 needs none of these (pure numpy).

---

## 11. Phase 0 result

Run: `python3 experiments/phase0_synthetic/phase0_synthetic.py` (pure numpy, ~1 second). Raw numbers in
`experiments/phase0_synthetic/phase0_results.json`.

- **Curved vs flat at matched dimension (planted circle):** FVU 0.136 (curved) vs 0.134 (flat); difference
  0.002 → **a tie**, as predicted. The "curved geometry beats flat" idea is correctly demoted to a control.
- **Identifiability up to isometry (circle):** recovered-angle correlation **0.999** → pass.
- **Single-direction covering-number penalty:** ~**16** single-direction atoms to match the reconstruction of
  **1** two-dimensional block → penalty present, consistent with the published capacity result.
- **Honest caveat (planted torus):** naive PCA + angle extraction recovers the 4-D subspace well (FVU 0.07) but
  does **not** separate the two circular factors (per-angle correlation ~0.60). So recovering the real (φ,ψ)
  torus will need a genuine factorization method, not PCA — recorded as a Phase 2 task.

**Gate G0: GO.** Interpretation: commit to the identity + cause + truth-vs-confidence paper and keep
curved/block dictionaries only as controls.

---

## 12. Prior art to differentiate from

Read the full texts before writing.

- **SASA (Dal26, NeurIPS 2026)** — owns the capacity theorem and the block remedy. *Concede, do not compete.*
- **Skopek (2019), mixed-curvature VAE** — owns learnable signed curvature and documents its ill-conditioning.
- **Steenwyk (2026)** — already steers ESM3 at layer 33 with a single-direction SAE feature. *Must beat on
  matched-norm geometry lift and add identity + truth-vs-confidence.*
- **Off-manifold collapse in guided PLMs; masked-pLM edit persistence** — the wash-out antecedents; we extend to
  a layer map and an honest negative.
- **PairSAE (Boltz-2)** — predicted the model's *own* affinity (circular); we use *experimental* Kd.
- **Jedryszek (Boltz-1)** — "decodable ≠ steerable"; we add geometry and the wrap-around test.
- **Two NeurIPS 2026 codon-frame papers** — own the DNA codon leg; cite, do not compete.
- **Setlur (2026)** — geometry-annotated ESM-2 SAE features; closest protein prior art.
- **Language-model truth-vs-confidence papers (2026)** — the nearest conceptual scoop; **UNVERIFIED**, verify
  before committing the hook.

---

## 13. Risks and scoop exposure

- **Strongest unanswered rebuttal:** truth and confidence might be two readouts of one "well-structured" signal
  (the model is confident where geometry is regular). The confidence-matched-pair control and a "confidently
  wrong" slice are designed to close this; it is not closed on paper yet.
- **Scoop:** truth-vs-confidence is an active 2026 language-model topic. Differentiate on external ground truth +
  a deployed confidence head + causality + identifiability (which the language-model work structurally lacks).
- A human should check goodfire.ai/research and the BioEmu/HyperSAE items before submission.
- **Heaviest-effort phases:** Track B's mid-generation causal test, the Kd identity on leakage-aware splits, and
  per-step persistence caching. Keep Track B minimal until Track A is end-to-end.

---

## 14. Ethics and dual-use

Track A (structural angles) is low risk. Track B (binding) is capability-adjacent (close to binder design) and
is framed strictly as **auditing / calibration**: finding and *trusting* the binding representation, and flagging
confidently-wrong outputs via the truth-vs-confidence "trust axis." Rules: benign complexes only (exclude toxins,
viral-fusion, immune-evasion, and select agents); ipSAE is a contrast, never a steering target; no
affinity-maximising steering pipeline is released. **This section owes a human / compliance pass** — an automated
safety filter repeatedly truncated earlier attempts to draft the full dual-use statement, so it must be written
and reviewed by a person.

---

## 15. Minimum publishable result

If the ambitious causal claims fail, the project still yields a paper: a leakage-aware, exact-geometry
**benchmark + calibration/negative** paper consisting of (1) the physical-ground-truth suite and the RoPE-proof
protocol, (2) identity-up-to-isometry recovery, (3) the ESM3 persistence/commitment negative, and (4) the
truth-vs-confidence angle (Kd vs ipSAE; (φ,ψ) vs pLDDT). This concedes the published capacity and single-layer
steering results and contributes what they lack.

---

## 16. Open items and next actions

**Needs a human (cannot be done from this session):**
1. **ESM3-open weight access** — gated to your Hugging Face account under non-commercial terms.
2. **Verify** the dataset counts and licences (PDBBind, PLINDER, BindingDB, SKEMPI, CASF) and the three 2026
   language-model truth-vs-confidence papers (the nearest scoop) from primary sources.
3. **Dual-use statement** — write and review the §14 statement with a person / compliance.

**Can proceed without you:**
4. Scaffold **Phase 1** (persistence/commitment gate) and the **torus factorization** method that Phase 0 flagged.
5. Build the synthetic test harness out into the shared metric code reused by later phases.

---

## 17. References

Cite keys follow the project's Undermind workspace. DOIs/links are in `FINAL_FINDINGS.md` and the `appendix/`.
This list is the core set; it is not exhaustive and several items are **UNVERIFIED** as noted above.

- BSF — "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds" (Goodfire, arXiv 2606.25234).
- BSF critique — Jerpelea & Ananthram, "A Deeper Analysis of Block-Sparse Featurizers" (arXiv 2608.27515).
- SASA — "Subspace-Aware Sparse Autoencoders" (Dal26, arXiv 2606.06333, NeurIPS 2026).
- Jacobian lens — Gurnee et al., "Verbalizable Representations Form a Global Workspace in LMs" (arXiv 2607.15495).
- Jacobian-lens theory — Ostrow et al., "A Mathematical Theory of the Jacobian Lens" (arXiv 2610.31955).
- J++ — Ayonrinde & Lindsey (github.com/safety-research/jpp_lens; blog + code; **UNVERIFIED** via databases).
- Mixed-curvature VAE — Skopek et al. (2019).
- InterPLM — Simon & Zou, *Nature Methods* 2025 (10.1038/s41592-025-02836-7).
- Gujral et al., *PNAS* 2025 (10.1073/pnas.2506316122).
- Decodable ≠ steerable — Jedryszek et al., Boltz-1 (arXiv 2608.11475).
- Off-manifold collapse in guided PLMs (arXiv 2608.18597).
- ESM3 steering / structure features — Steenwyk (bioRxiv 2026).
- PairSAE — Migliorini et al., Boltz-2 (arXiv 2606.27440).
- Evo 2 — Brixi et al., *Nature* 2026 (10.1038/s41586-026-10176-5); SAE on layer 26 of the 7B model.
- Categorical Jacobian (contacts) — Zhang et al., *PNAS* 2024.
- Nucleotide-dependency maps — Tomaz da Silva et al., *Nature Genetics* 2025 (10.1038/s41588-025-02347-3).
- DeepSCENIC — Enformer/Borzoi internals into a CRISPRi-validated regulatory network (Par26e).
- ipSAE — Dunbrack lab interface score (bioRxiv 10.1101/2025.02.10.637595; code github.com/dunbracklab/IPSAE).
- Field review — Orlov et al., "What Do Biological Foundation Models Compute?" (bioRxiv 2026).

---

## 18. Repository file index

- `docs/research-plan/PROJECT_DOSSIER.md` — **this file** (the single comprehensive record).
- `docs/research-plan/MASTER_PLAN.md` — the step-by-step plan with phase and gate detail.
- `docs/research-plan/primer_bsf_jacobian_landscape.md` — plain-language primer.
- `docs/research-plan/PROJECT_BRIEF.md` — the one-page project definition.
- `docs/research-plan/FINAL_FINDINGS.md` — verified landscape, reading list, model availability, gaps.
- `docs/research-plan/PITCH_VETTING_consolidated.md` — the four pitches and their already-done / disproven checks.
- `docs/research-plan/PLAN_ICML_synthesis.md`, `evidence_base.md`, `technical_plan.md`, `my_pitches.md` — provenance.
- `docs/research-plan/appendix/` — the independent agent reports (deep searches, per-track plans, vettings, reviews).
- `experiments/phase0_synthetic/` — the Phase 0 script and its result.

_Provenance: generated with Claude Code. External claims are UNVERIFIED unless a primary source is cited above._
