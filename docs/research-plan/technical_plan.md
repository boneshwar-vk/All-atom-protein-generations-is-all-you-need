# BSF x Jacobian-transport on bio foundation models — scoping report

Date: 2026-11-19. Planning/research only; repo not modified. "UNVERIFIED" marks
anything not confirmed from a primary source (arxiv.org, biorxiv.org, hf.co and
the Bonbon secondary hosts were all DNS-blocked in this session; see note at end).

================================================================================
## 1. BONBON — identification
================================================================================

**Finding: "Bonbon" is a real, named 2026 model, but it is NOT an all-atom
protein–ligand co-folding model. It is a sequence-only protein–ligand language
model, and it is almost certainly a poor fit for the project as currently
framed.** Details below; the mismatch is important and is revisited in the
"three decisions" at the end.

- **Exact name:** "Bonbon" (that spelling; a secondary host also wrote "BonBon"
  once, but the paper/press use "Bonbon"). It is the model; the paper is titled
  **"Emergent Biological Capabilities in a Foundation Model for Molecular
  Interactions."**
- **Author / lab:** single author **Adi Mashiach**. Affiliation surfaced only as
  **"Output Labs" (OutputLabs)** via the HuggingFace papers submitter field —
  UNVERIFIED as the formal institutional affiliation; no lab page confirmed.
- **Date:** arXiv preprint **2611.14504**, listed **10–12 Nov 2026**.
- **Venue:** reported **accepted at NeurIPS 2026, Main Track, Oral**
  (neurips.cc/virtual/2026 oral/poster 149425). UNVERIFIED beyond listing pages.
- **Links:** arXiv abs/pdf `2611.14504`; HuggingFace papers page `papers/2611.14504`
  (community page only, no checkpoint); NeurIPS virtual 149425. (All DNS-blocked
  here; identifiers confirmed across ≥4 independent secondary indexes.)
- **Architecture (from abstracts + secondary reviews; UNVERIFIED in detail):**
  decoder-style **transformer language model** trained by **self-supervised
  learning on paired protein–ligand "sequences."** Reported ingredients:
  pre-layer-norm, rotary position embeddings. **No 3D structure, no coordinates,
  no pair representation, no diffusion/flow head** — this is explicitly pitched
  as an *alternative* to co-folding that skips structure. How the ligand is
  tokenised (SMILES vs SELFIES vs atom tokens) is **NOT stated in any primary
  source I could reach — UNVERIFIED and material** (see decisions). Objective
  (masked vs autoregressive/next-token) likewise **UNVERIFIED**; the decoder +
  RoPE hints point to autoregressive, unconfirmed.
- **Parameter count:** production model **1.6B**; scaling ladder 42M / 315M /
  800M / 1B / 1.3B / 1.6B. Training corpus **~3.2T tokens** (1:2000 param:token).
- **Reported capabilities (zero-shot, emergent at 1.6B):** functional
  (mechanism of action), residue (binding-site localisation incl.
  orthosteric/allosteric), bond (covalent vs non-covalent); a 4th atom-level
  "active moiety" capability appears in the NeurIPS listing but not arXiv v1.
  Wet-lab claim: **61.5%** dose-response hit rate (arXiv) / **70%** (NeurIPS
  listing) across nine targets — figures conflict, both UNVERIFIED, targets and
  scoring undisclosed.
- **Weights / code / licence:** **No public weights, no code repo, no model card,
  no API found.** Secondary reviews state training data and weights are
  proprietary. **Treat as closed-weights unless Output Labs releases them.**
- **Interpretability work on it:** **none found.** The paper's own analysis
  (attention-weight correlations, MLP probes on frozen interaction reps) is the
  only mechanistic content to date.

**Bottom line for the project:** Bonbon matches the name the user gave, but it is
sequence-only and closed-weight. It has **no pair representation and no diffusion
head**, so the "pocket geometry / interatomic distance / pose" manifolds the
thesis wants from a ligand model are *not* architecturally present, and without
weights we cannot hook it at all. **Recommendation: do not build the plan around
Bonbon.** If an open protein–ligand co-folding model is wanted, the closest real,
*weights-available* candidates found are **Boltz-2** (already has PairSAE interp,
arXiv 2606.27440), **Chai / RoseTTAFold-All-Atom**, and the sequence-only
zero-shot binding-site model **Pou25b** (Pourmirzaei et al., protein seq + SMILES).
I did not pick one; flagging for the user's decision. If the user specifically
wants *Bonbon itself*, the only viable interp would be behavioural/black-box
(attention export if ever released), not BSF/Jacobian on internals.

================================================================================
## 2. Per-model activation-caching + BSF plan
================================================================================

Common BSF setup (from the local repo, verified in code):
- Three variants: **Vanilla** (free `(W,b)` encoder, block-TopK, free decoder),
  **Grassmannian** (tied `z_g = γ·x Dg^T`, orthonormal frames, block-TopK),
  **Group-Lasso** (free encoder, block-JumpReLU gate via STE to a target L0).
- Key knobs: `d` (activation width), `n_groups`, `group_size` (default 3),
  `l0` (blocks kept per token). API e.g.
  `bsf.GrassmannianBSF(d=768, n_groups=256, group_size=3, l0=16)`.
- **Sweep grid (all models):** group_size ∈ {1 (=ordinary SAE control), 2, 3, 4,
  6, 8}; n_groups set so total width = {8d, 16d, 32d}; l0 ∈ {8, 16, 32, 64}.
  Run Grassmannian first (cleanest manifold recovery, fewest knobs), then
  Group-Lasso (adaptive L0), Vanilla as ablation. Address the Jerpelea &
  Ananthram seed-dependence critique by running **≥5 seeds** per config and the
  **Tournament Top-K** fix at the headline config.
- **SAE baselines at equal width/L0:** TopK, BatchTopK, Matryoshka. group_size=1
  BSF is itself a matched TopK control. Report: reconstruction (FVU), L0, and
  manifold-recovery metrics (below), not just loss recovered.

### 2a. ESM3-open (esm3-sm-open-v1, 1.4B, 48 layers) — PRIMARY model
- **Hook sites:** residual stream after each transformer block; cache layers
  {8,16,24,32,40,46} for the sweep, full sweep on the best 2. ESM3 is a
  *multi-track* model — hook the **sequence-track residual stream** (d≈ model
  width, UNVERIFIED exact d; esm3-sm has ~1536 hidden UNVERIFIED). Optionally
  also cache the **structure-track** tokens to test structural manifolds.
- **Corpus:** 50–100k single-chain domains from a PDB/CATH non-redundant set
  (<30% identity), length-capped 512. Residue-level tokens → ~20–40M activation
  vectors; cache fp16.
- **Storage:** 40M × 1536 × 2B ≈ **120 GB** per layer; cache 3–4 layers to disk,
  stream the rest. Pilot: 5k proteins, one layer ≈ 6–12 GB.
- **Ground-truth continuous variables (physically computed, not DB annotations):**
  - backbone **φ/ψ** dihedrals from coordinates (mdtraj/DSSP) → periodic; test
    for 2-torus manifold recovery per block.
  - **per-residue SASA** (DSSP / Shrake–Rupley) → scalar.
  - **secondary-structure helix phase** (residue index mod 3.6) → periodic.
  - **χ1 side-chain** dihedral; **contact number / local density**.
  Validation = can a block's group_size-D code *linearly/periodically* decode the
  variable where a matched 1-D SAE latent cannot (circular-correlation, kNN
  decoding R²).

### 2b. Bonbon — DROP (see §1). No weights, no internals, wrong architecture.
If weights appear: hook the decoder residual stream at a ligand/protein boundary,
same BSF grid; ground truth from the ligand (RDKit descriptors, Morgan-bit
presence, covalent-warhead SMARTS) and protein (same as ESM3). Hold as contingency.

### 2c. AlphaGenome v1 (JAX, weights public, non-commercial)
- **Hook sites:** the **fixed-resolution transformer tower** (the 1 bp→…→
  multi-kb U-Net). Cache the transformer trunk at its coarsest resolution
  (richest semantics) and one mid U-Net resolution. UNVERIFIED exact widths.
- **Corpus:** 2–5k genomic windows (≈1 Mb context) spanning coding/intronic/
  UTR/intergenic; sample positions to balance classes.
- **Storage:** dominated by sequence length; cache strided positions (every 128 bp)
  to keep one layer ≈ tens of GB. Pilot one chromosome arm.
- **Ground-truth continuous/periodic variables:**
  - **codon phase** (position mod 3 within annotated CDS) → periodic, the
    cleanest periodic target in the whole study.
  - **exon relative position** (0–1 within exon), distance to splice site.
  - **GC content** in a sliding window; nucleosome-phasing periodicity (~10.5 bp)
    as a stretch periodic target.
  All computed from GENCODE annotation + the reference sequence directly.

### 2d. Evo 2 (open; public Goodfire SAE baseline exists)
- **Hook sites:** residual stream of the StripedHyena/attention blocks; pick the
  layer Goodfire's public SAE used so BSF vs their SAE is a clean head-to-head.
- **Corpus:** reuse the Goodfire SAE's activation corpus if published; else 50k
  genomic/CDS windows.
- **Ground truth:** codon phase, reading-frame, GC — as AlphaGenome. **Primary
  value of Evo 2: the existing SAE baseline makes it the cheapest "does BSF beat
  a real deployed SAE on a periodic variable" test.** Run this first.

================================================================================
## 3. Jacobian-transport component (per architecture)
================================================================================

From the local jacobian-lens repo (verified): `J_l` = **average input–output
Jacobian over a corpus**, expectation over prompts × source positions × all
current-and-future target positions; estimator = cotangents summed over target
positions then averaged over source positions; **~100 prompts saturates**; cost
is one backward pass per target-position group per prompt. J++ adds k-means
"expert" Jacobians + supervised weights + LRP backward rules + readout filtering.

**Core redesign for bio models:** keep `J_l` as the **linear transport map**
d h_final / d h_l; **replace the token-unembedding readout** (bio models lack a
rich causal token vocab). Readout options: (a) the model's own output head
(structure logits / track head / coordinate update), (b) a **BSF basis** at the
final layer — project J_l-transported block subspaces onto final-layer BSF
concepts, (c) a supervised linear probe for the target continuous variable.
**Combine with BSF:** for each source-layer block subspace `Dg` (group_size dims),
push it through `J_l` and measure its norm/alignment in the readout space →
**output-relevance score per manifold**. Rank manifolds by this; confirm causally
by ablating/steering that block subspace and measuring the readout change.

- **ESM3 (bidirectional, non-causal):** the causal "current-and-future target"
  summation is undefined. Redefine J_l = averaged Jacobian of a **final-layer
  readout of interest** (structure-token logits, or a φ/ψ probe) w.r.t. layer-l
  residuals, averaged over **all** positions (no causal mask). Readout = structure
  track or probe. Cost: one VJP per readout direction × ~100 proteins. **Keep it.**
- **AlphaGenome (U-Net):** resolution changes make a single square `J_l`
  ill-defined across the U-Net neck. Restrict J_l to **within the fixed-resolution
  transformer tower** (square, well-defined); treat the U-Net up/down-sampling as
  fixed linear maps folded into the readout. Readout = an expression/accessibility
  track head. **Keep it, tower-only.**
- **Evo 2 (causal decoder):** closest to the original J++ setting; token readout
  replaced by a nucleotide/codon probe or BSF basis. **Keep it — this is where
  J++ transfers most directly and should be the Jacobian pilot.**
- **Co-folding model w/ pair tensor + diffusion head (only if a real one is
  swapped in for Bonbon):** pair representation is rank-4 (i,j,c) — J_l becomes a
  tensor; restrict to **single-representation residual** transport and treat the
  pair trunk + diffusion head as the readout (differentiate a pose/coordinate
  scalar w.r.t. layer-l single rep). If the diffusion head is stochastic, average
  Jacobian over diffusion timesteps. **High risk — drop the Jacobian component
  for co-folding unless time allows.**

**Drop Jacobian for:** Bonbon (no weights), the AlphaGenome U-Net neck, any
diffusion head in a first pass.

================================================================================
## 4. Staging + GPU budgets
================================================================================

**Pilot → full sequence (go/no-go gates):**
1. **Evo 2 BSF-vs-SAE on codon phase** (cheapest; existing SAE baseline).
   GO if BSF group_size=3 recovers codon-phase periodicity with circular-R²
   materially above matched TopK/BatchTopK. NO-GO kills the periodic-manifold
   thesis early.
2. **ESM3 φ/ψ torus recovery**, one layer, 5k proteins. GO if a block decodes
   the 2-torus where 1-D latents fragment it (manifold-splitting shown).
3. **Jacobian transport on Evo 2** (codon-phase readout) → does manifold ranking
   predict causal ablation effect? GO if ranking correlates with intervention.
4. **Full ESM3 sweep** (all variants, seeds, Tournament Top-K) + AlphaGenome
   codon/exon. 5. **Write-up.** Bonbon stays a contingency box.

**GPU-hour estimates (order-of-magnitude, UNVERIFIED):**
- **Single 24 GB GPU:** Evo-2 pilot + ESM3 one-layer pilot only. Activation
  caching streamed, BSF training is cheap (BSF fits on cached activations, not
  the big model). ~1–2 weeks wall-clock; ~150–300 GPU-h. Enough for gates 1–2.
- **1–2 × 80 GB:** full ESM3 sweep + Evo2 + AlphaGenome tower, Jacobian on Evo2
  and ESM3. ~800–1500 GPU-h over 4–6 weeks. The intended working budget.
- **Small cluster (8–16 × 80 GB):** add seed ensembles (≥5), Tournament Top-K,
  full AlphaGenome, co-folding contingency, all Jacobian variants. ~4–8k GPU-h;
  the full paper.

================================================================================
## THREE DECISIONS MOST LIKELY TO SINK THE PROJECT
================================================================================

1. **Bonbon is the wrong/absent fourth model.** It is sequence-only and
   closed-weight with no pair rep and no diffusion head, so the ligand-geometry
   manifolds the thesis promises cannot be probed and we cannot hook it at all.
   Building the paper's "protein–ligand" leg on Bonbon fails; must substitute an
   open co-folding model (Boltz-2, already interp-ed, is the safe choice) — which
   changes scope and overlaps prior work.
2. **Jacobian transport may be ill-defined exactly where it matters.** It is
   clean on Evo 2 (causal), awkward on bidirectional ESM3, and genuinely
   ill-posed across the AlphaGenome U-Net neck and any diffusion head. If the
   "which manifold is output-relevant" ranking only works on the easy causal
   model, the central methodological novelty (BSF × transport) shrinks to one
   model.
3. **Manifold recovery might be a metric artefact, not a real win.** BSF with
   group_size≥2 has strictly more capacity than a 1-D SAE latent, so it will
   decode continuous variables better *by construction*. Without a capacity-matched
   control (group_size=1 at equal total width, Tournament Top-K for seed
   stability, and a causal intervention confirming the manifold is used by the
   model) the headline result is unfalsifiable and reviewers will reject it.

Network note: arxiv.org, biorxiv.org, huggingface.co, neurips.cc, alphaxiv.org,
aiweekly.co, astrobiology.com, themoonlight.io all failed DNS in this session, so
the Bonbon architecture/objective/tokenisation and exact model widths are
UNVERIFIED and taken from search snippets + secondary indexes only. Undermind PDF
read of 2611.14504 was not available (paper not in corpus). Confirm Bonbon
details from the arXiv PDF before relying on them.
