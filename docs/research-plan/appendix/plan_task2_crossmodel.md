# Task 2 — "Use SAEs and BSFs to visualise all the models and compare which model's
# representations are relatively good": critical research plan + blunt review

Author: senior interpretability researcher / skeptical reviewer pass. Date: 2026-11-19.
Research/planning only — no repository changes. Describes published work only.
Companion to the reshaped Task-1 plan in FINAL_FINDINGS.md (§5). Prior-art verified via Undermind
workspace 33899782-4f3e-48f1-9e4a-a3b6ac8a7af7 and WebSearch on 2026-11-19; "UNVERIFIED" where noted.

--------------------------------------------------------------------------------
## 0. ONE-PARAGRAPH VERDICT (read this first)
--------------------------------------------------------------------------------
As stated, Task 2 is **not a standalone paper**. "Visualise all the models and compare which
representations are good" is under-specified in exactly the way reviewers punish, and its two
obvious sharp versions are **both already occupied on biological models**: (i) cross-model
SAE **feature-sharing** between protein models was done by Steenwyk (ESM-2 vs ESM3, "78% feature
convergence", 2026), and (ii) a cross-model **concept-recovery leaderboard** via probing was done
by Whitfield et al. [Whi26] across 8 protein LMs. The enabling **methods are off-the-shelf**
(Universal SAEs [Tha25]; feature-space universality via SAEs [Lan24]; crosscoders [Min25, Jir26]),
so "port dictionary learning to multi-model bio comparison" is not itself a contribution.
**Recommendation: MERGE Task 2 into Task 1** as a 2–3-model *generality* section whose only fair
axis of comparison is **recovery of the same external ground-truth variables** (Task-1's metrics),
never internal reconstruction loss. A standalone is defensible only in one narrow, scoop-exposed
form: a **crosscoder/USAE on the continuous/manifold variables across two bio models** (shared-vs-
unique *manifold* features), explicitly differentiated from Steenwyk (scalar matching) and Whitfield
(probing). Verdict: **MERGE** (primary) / thin standalone only if forced (see §5).

================================================================================
## 1. SHARPEN THE QUESTION — what "good representations" can operationally mean
================================================================================
"Which model's representations are relatively good" is undefined three times over: *good for what*,
*good by which measure*, and *good compared on what common scale*. Enumerate the candidate operational
definitions and mark each defensible / circular / apples-to-oranges.

| # | Operational definition of "good" | What you'd measure | Verdict |
|---|---|---|---|
| D1 | **Annotation/concept coverage** — fraction of Swiss-Prot/GO/InterPro (protein) or NCBI/Ensembl (DNA) annotations that some dictionary feature detects at FDR control | #concepts hit, F1 per concept | **CIRCULAR + apples-to-oranges.** The annotations are largely in training data (Orlov circularity); and a wider model / wider dictionary trivially "covers" more. Not a model-quality signal. |
| D2 | **Reconstruction–sparsity frontier** — reconstruction loss (or FVU) at matched L0 of each model's own SAE/BSF | Pareto curve per model | **APPLES-TO-ORANGES — the central pitfall (see §1a).** Different d, tokenizer, output space, activation norm. A lower FVU says the *dictionary* fit that model's activations better, not that the *model* is better. **Never use for cross-model ranking.** |
| D3 | **Recovery of shared external ground-truth variables** — from each model's activations, how well a capacity-matched dictionary (and a linear probe baseline) recovers the SAME computed variables (φ/ψ torus, ω, χ1, helix phase, rel-SASA, contact number; codon phase, exon position, splice distance, GC, nucleosome phasing) | circular-R²/decoding accuracy per variable, per model, at matched capacity + vs random-init | **DEFENSIBLE — the only fair axis.** Same yardstick, same units, external to all models. This is literally Task-1's metric set evaluated across models. |
| D4 | **Feature stability across seeds** — reproducibility of the dictionary under re-training | Jaccard / consensus-feature fraction over ≥5 seeds | **DEFENSIBLE as a property, NOT as "goodness".** It measures the *dictionary's* stability, not the *model's* quality (Sha26: reproducibility ≠ meaning; Pau26: same-data SAEs learn different features; Ger26: unstable features, reproducible subspaces). Use as a *control*, not a leaderboard axis. |
| D5 | **Feature universality / sharing across models** — fraction of features one model shares with another | matched-feature fraction vs permutation null | **DEFENSIBLE as a *descriptive* claim, but it ranks *pairs*, not models, and it is the Steenwyk result already (see §2).** "Shared" ≠ "good". |
| D6 | **Downstream-task usefulness** — frozen-embedding probe performance on held-out biological tasks | task accuracy per model | **DEFENSIBLE but NOT a dictionary-learning result** — it's representation benchmarking (Bos24, Dat26, Tan25c, May26 review) and does not need SAEs/BSFs at all. Using SAEs here adds nothing a linear probe doesn't. |

**Keep: D3 (primary), with D4 and random-init as controls, and D5 as a descriptive sub-result.
Drop D1 and D2 as ranking axes. D6 belongs to a different (crowded) paper.**

### 1a. The central pitfall, stated for a reviewer
Comparing dictionaries **by their internal reconstruction loss across models is meaningless** because
the quantity being reconstructed is different for each model:
- **Different width d** (ESM-2 650M: 1280; ESM3-open: 1536; Evo 2 7B: 4096; NT/AlphaGenome differ) →
  FVU and L0 live on different scales; "bits reconstructed" is not comparable.
- **Different tokenizer / sequence granularity** (single-aa vs BPE vs k-mer vs single-nt vs byte) →
  the activation at "a position" means different things; feature counts are not comparable.
- **Different output space / objective** (masked-LM logits over 20 aa; autoregressive over 4 nt;
  multi-track regression heads; multimodal structure tokens) → no shared readout to normalise against.
- **Different activation geometry** (anisotropy/nativeness axis differs across families — Big26, Hak25)
  → raw loss conflates "how compressible the activations are" with "how good the model is".

Therefore **the only fair comparison is an EXTERNAL yardstick that is identical for every model**:
the recovery of the *same* computed ground-truth biological variables (D3). Those variables are
defined on the *input* (a residue's dihedral, a codon's phase), not on any model's internals, so they
put every model on one ruler. **This is exactly Task-1's validation metric set — so Task 2, done
correctly, is Task-1's pipeline run across models, not a new method.** That is the strongest argument
for merging, and the strongest argument that a naive "reconstruction-loss leaderboard" paper is wrong.

================================================================================
## 2. PRIOR ART (verified) — what already exists, so this is not reinvented
================================================================================
**Confidence:** CONFIRMED-REC = resolved to the real record/abstract via Undermind; WEB = web snippet
only; all DOIs from Undermind get_paper_info unless marked. None invented.

### 2.1 Cross-model SAE / dictionary comparison already done ON BIO MODELS (the direct scoops)
- **Steenwyk, "Structure tokens sharpen the feature vocabulary of protein language models"** (bioRxiv
  2026; DOI 10.64898/2026.05.12.724593, WEB — not indexed in the Undermind workspace, abstract/
  summary pages only, **UNVERIFIED against full text**). Trains SAEs on **ESM-2 650M (seq-only)** and
  **ESM3 1.4B (multimodal)** and reports **78% of features converge across the two models**
  (permutation null 14.2%, p<0.001); structure-activated features are *more* convergent with ESM-2
  than structure-invariant ones (r=0.54 vs 0.45); identifies **one attention head L0H7** as the
  structural-information bottleneck; reports steering + attribution patching + sparse feature circuits
  as causal confirmation. **This is a cross-model SAE feature-sharing study on protein models — i.e.
  D5 already executed.** Any "cross-model SAE comparison on bio" framing must cite and clear it.
  (Note: the same L0H7 / ESM3 steering collision is already flagged in FINAL_FINDINGS §3 as the
  under-weighted Steenwyk threat to the Task-1 ESM3 plan.)
- **Whitfield et al. [Whi26], "High-resolution dissection of concept acquisition in different families
  of protein language models"** (bioRxiv 2026-07, DOI 10.64898/2026.07.20.739599, CONFIRMED-REC).
  Layer-by-layer interpretability of **8 models (ESM2 + AMPLIFY families) on 22 human-proteome
  concepts**; concepts emerge by depth (physicochemical/motifs early → secondary structure → domain
  semantics mid); PCA separations + phosphomimic intervention; **pretraining data/compute matter more
  than parameter count** for linear concept emergence. **This is a cross-model concept-recovery
  comparison (D1/D3 via probing) — i.e. the "leaderboard" design, minus BSF.** Direct partial scoop
  of method option (a).

### 2.2 Cross-model representational comparison on bio FMs WITHOUT dictionary learning (crowded)
- [Bos24] "Are genomic language models all you need?" (Bioinformatics 2024) — cross-model gLM-vs-pLM
  on protein downstream tasks (D6).
- [Dat26] "Frozen but Not Always Accessible" (2026) — frozen-probe representation analysis across
  DNABERT-2/NT/HyenaDNA/GENERATOR-v2/Omni-DNA; task-dependent accessibility (D6).
- [Hak25] "Isotropy and Geometry of Pretrained Protein LMs"; [Ris25] "Decoding protein language
  models: insights from embedding space analysis"; [Big26] "Viral Proteins Reveal Geometry of
  Protein Language Models" (nativeness axis across ESM families); [Sij26] "From Words to Amino Acids:
  Does the Curse of Depth Persist?" — all cross-model representational-geometry on protein LMs.
- [Wan26k] "Asymmetric Structural Transfer Between Natural Language and Biological Foundation Models"
  (bioRxiv 2026) — directional transfer across ESM-2/ProtBERT vs language models.
- [Fan26d] "When Do Biological Reasoning Models Use Their Biological Inputs?" (2026) — probes Evo2 &
  ESM3 representations inside reasoning models; **decodable ≠ used** at the system level.
- [May26] "Interpreting embeddings from genome and protein language models" (Biotechnology Advances
  2026) — **a review of exactly this space**; [Tan25c] "Benchmarking genomic language models"
  (Nature Methods 2025). → the "compare bio-model embeddings" lane has a review and a benchmark
  already; a reviewer will know it is populated.

### 2.3 The cross-model dictionary-learning METHODS (all non-bio — the method is off-the-shelf)
- **[Tha25] Universal Sparse Autoencoders (USAEs)** (2025, 73 cit; DOI 10.48550/arXiv.2502.03714) —
  one overcomplete SAE that ingests any model's activations and decodes to any other's; learns a
  **shared universal concept space across models**. This IS the "shared dictionary across models"
  design (method option b), already built (on vision). Porting to bio is the only novelty available.
- **[Lan24] Quantifying Feature Space Universality Across LLMs via SAEs** (2024, 27 cit;
  arxiv 2410.06981) — "Analogous Feature Universality": pair SAE features by activation correlation,
  then compare spaces under rotation-invariant RSA. The exact recipe for a cross-model SAE comparison.
- **Crosscoders / model-diffing (all NLP/vision):** [Min25] "Overcoming Sparsity Artifacts in
  Crosscoders to Interpret Chat-Tuning" (NeurIPS 2025, Nanda — Latent Scaling + BatchTopK crosscoder;
  the methodological state of the art and its known artifact trap); [Jir26] "Cross-Architecture Model
  Diffing with Crosscoders" (Bricken 2026 — Dedicated Feature Crosscoders, first *cross-architecture*
  diffing); [Kas26] Delta-Crosscoder; [Cha26d] Sparse Crosscoders for MoE-vs-dense; [Bat26] Multimodal
  Model Diffing; [Nag26c] **SSM-vs-Transformer latent comparison via SAEs** (Mamba vs Pythia, "99.98%
  cluster toward alignment" — a direct template for an ESM-vs-Evo style claim, in NLP).
- **"Do independently trained models learn the same features / same data → different features":**
  [Li15] Convergent Learning (the origin question); [Pau26] "SAEs Trained on the Same Data Learn
  Different Features" (ICLR 2026 — the key caution: feature-matching is unstable); [Ger26] "Unstable
  Features, Reproducible Subspaces" (seed dependence → compare *subspaces*, not features); [Son25]
  Semantic Convergence across scaled LLMs; [Lee26b] "Same Concept, Different Directions" (cross-modal
  feature heterogeneity in SAEs); [Liu26o] Similarity of NN Representations in Superposition.
- **RSA/CKA lineage:** [Ban21] model stitching; [Her23]; [Csi21]; [Bo24] / [Wu25h] (discriminative
  capacity of similarity metrics — CKA/RSA caveats); [Dhi26] unifying concept-based RSA.

### 2.4 What is genuinely still open (the narrow gap)
- **No FORMAL crosscoder / USAE / model-diffing has been applied to ANY biological model.** Confirmed:
  every crosscoder/model-diffing paper above is NLP or vision; FINAL_FINDINGS gap #2 ("no crosscoder/
  model-diffing on any bio model — HIGH") holds *for the formal method*. **But** Steenwyk already did
  the *result* a crosscoder would produce (shared-vs-unique SAE features across two protein models),
  so the gap is "the formal method / the continuous-manifold version", not "the question".
- **No cross-model comparison on the CONTINUOUS/PERIODIC manifold variables** (the Task-1 least-
  scoopable atom). Steenwyk and Whitfield both compare *scalar* features / *discrete* concepts; nobody
  has compared models on how they encode φ/ψ-torus-type manifolds, nor built a crosscoder that finds a
  *shared manifold* feature. This is the only standalone-worthy slice, and it is small.

================================================================================
## 3. METHOD OPTIONS — three realisable designs, with the alignment problem spelled out
================================================================================
**The alignment problem (common to all three):** models have different d, different tokenization, and
(for protein-vs-DNA) different sequence spaces, so "feature i of model A" and "feature j of model B"
are not comparable by index. Three escapes: (i) **don't align — compare external recovery** (option a);
(ii) **learn a shared space** that forces alignment (option b); (iii) **align geometry up to rotation**
(option c). Only token-aligned models (same sequence, e.g. two protein models on the same residues)
permit activation-correlation feature pairing [Lan24]; protein-vs-DNA models are NOT position-aligned,
so cross-domain feature pairing is undefined — cross-domain comparison can ONLY be option (a).

### Option (a) — Matched dictionaries per model + shared-ground-truth LEADERBOARD  [RECOMMENDED]
- **Design:** train the *same* dictionary family (BSF Grassmannian/Group-Lasso/Vanilla + a group_size=1
  TopK-SAE control, Tournament Top-K per Jer26) at *matched capacity* on each model's activations;
  score each by **D3** = circular-R²/decoding of the identical ground-truth variables, plus a linear-
  probe ceiling and a random-init floor per model.
- **Metric:** per-(model, variable) recovery at matched bits/MDL; a "manifold-recovery score".
- **Alignment problem:** **sidestepped** — never pair features across models; compare only the external
  scalar/manifold recovery, which is model-agnostic. This is the *only* design that extends to
  protein-vs-DNA.
- **"Model A represents concept X better than B" means:** at matched dictionary capacity and above each
  model's own random-init floor, A's best block recovers X's ground-truth manifold with higher
  circular-R² / lower geodesic error than B's — reported with a seed-ensemble CI and the probe ceiling.
- **Cost/novelty:** cheapest; reuses Task-1 infra wholesale. **Novelty is low on its own** (it is
  Whitfield's design with BSF instead of probes) → belongs *inside* Task 1, not as a paper.

### Option (b) — Crosscoder / USAE: one shared dictionary across two models  [standalone-if-forced]
- **Design:** train a BatchTopK crosscoder (Min25) or a USAE (Tha25) jointly on two *position-aligned*
  protein models (e.g. ESM-2 650M vs ESM3-open on identical residues; or ESM3-open vs ESM C) to split
  latents into **shared** (direction in both) vs **model-unique**. Use the **continuous/manifold**
  ground-truth variables as the external label of what the shared vs unique features encode.
- **Metric:** shared/unique latent counts after **Latent Scaling** (Min25 — mandatory; naive L1
  crosscoders *misattribute* shared concepts as unique, the single most common reviewer kill); causal
  effect of shared vs unique latents on each model's own output head.
- **Alignment problem:** *solved by construction* (shared space), but only for position-aligned models;
  requires a shared activation site and comparable sequence tokenization.
- **"A represents X better" means:** X's manifold is carried by a **shared** latent that is *stronger/
  more causal* in A — or is **unique** to A. This is the only design that gives a crisp shared-vs-unique
  statement.
- **Cost/novelty:** higher compute (joint training, activation caching for 2 models); **novelty = first
  crosscoder on bio + first on manifold features.** Scoop-exposed to Steenwyk (did the SAE-matching
  version) and Goodfire (owns BSF + crosscoder-adjacent methods). This is the only defensible standalone.

### Option (c) — Representational-geometry (RSA/CKA) baseline  [cheap baseline, not a paper]
- **Design:** CKA / RSA / mutual-kNN between models' activations (and between their dictionaries'
  feature-activation matrices) on the same inputs; [Lan24]'s rotation-invariant space comparison.
- **Metric:** CKA/RSA similarity; but **Wu25h/Bo24 show these metrics have weak discriminative capacity
  and disagree** → report ≥2 metrics and a sanity check, never a single CKA number.
- **Alignment problem:** handled up to rotation; still undefined across non-aligned token spaces.
- **Role:** a **cheap baseline/control** that says whether two models are even comparable before you
  invest in (a)/(b). On bio this is already partly done (Hak25, Big26, Whitfield PCA) → **not novel
  alone.** Include as one figure, not the contribution.

================================================================================
## 4. VISUALISATION — what "visualise all the models" must concretely produce
================================================================================
"Visualise all the models" is the weakest word in the task; purely qualitative eyeballing (UMAP
galleries, hand-picked feature activations) is **exactly what reviewers reject** (Zho26pos: "move
beyond anecdotal interpretability evaluation"; Orlov: circular annotation-matching). Every visual must
carry a quantitative, pre-registered claim with a null.
- **Per-concept recovery curves (the core figure):** x = dictionary capacity (bits/L0), y = circular-R²
  / decoding of each ground-truth variable, one line per model, with random-init floor (shaded) and
  linear-probe ceiling (dashed). Replaces "which model is good" with a dominance plot per variable.
- **Shared-vs-unique feature map (option b only):** a 2-model scatter of latents in (strength-in-A,
  strength-in-B) space *after Latent Scaling*, coloured by which ground-truth variable (if any) each
  decodes; the diagonal = shared, axes = unique. This is the honest version of a "model-diffing" figure.
- **Feature/subspace atlas per model:** for each ground-truth variable, the recovered block's subspace
  and its seed-consensus (Ger26-style reproducible-subspace overlay), NOT a single-seed cherry-pick.
- **A leaderboard table, not a vibe:** (model × variable) recovery at matched capacity, with CIs over
  ≥5 seeds, probe ceiling, random floor, and an explicit "not significant" where it is not.
- **Guardrails to pre-commit:** matched capacity/bits across models; same inputs; random-init floor per
  model; ≥5 seeds with consensus features (Sha26); report ≥2 similarity metrics for any geometry claim
  (Wu25h); never rank models by reconstruction loss (§1a).

================================================================================
## 5. VERDICT — standalone, section, or benchmark? scoop, risks, recommendation
================================================================================
### Is it a standalone paper?
**No, not as stated.** The "leaderboard of which bio model is best" framing (option a) is Whitfield's
design with BSF swapped in for probes; the "cross-model SAE feature-sharing" framing (option b's SAE
version) is Steenwyk's executed result; the methods (USAE/Lan24/crosscoders) are off-the-shelf from
NLP/vision; and the broad "compare bio-model embeddings" lane already has a review (May26) and a
benchmark (Tan25c). None of those leave room for a *general* cross-model comparison paper.

### Does it hit the crosscoder/model-diffing gap enough to be novel?
**Partially, and only in one narrow form.** The *formal* crosscoder-on-bio gap (FINAL_FINDINGS #2)
is real — no crosscoder/USAE has touched a bio model. But Steenwyk already produced the *finding*
(shared vs model-specific SAE features across ESM-2/ESM3), so a crosscoder paper cannot claim to be
the first to ask the question — only the first to use the formal method and (if scoped to the
continuous/periodic variables) the first on **manifold** features. "RSA-on-bio-models" (option c) is
**already too crowded** (Hak25, Big26, Whitfield, Bos24, Dat26) to carry a paper.

### Strongest framing (if a standalone is forced)
A **crosscoder/USAE across two position-aligned protein models (ESM-2 650M vs ESM3-open) restricted to
the continuous/periodic ground-truth manifolds**, asking: *is a backbone-dihedral / helix-phase
manifold a **shared** latent or a **model-unique** one, is it more **causal** in the multimodal model,
and does BSF's block structure recover a shared manifold that scalar crosscoder latents fragment?* —
with Latent Scaling (Min25) mandatory, Tournament Top-K (Jer26), seed consensus (Sha26/Ger26), random-
init floor, and the model's own output head as the causal readout. This inherits Task-1's crown-jewel
(toroidal wrap-around causal test) and adds exactly one new axis (shared-vs-unique). It is the only
version that is both novel and not a leaderboard.

### Scoop exposure (ranked)
1. **Steenwyk** — maximal. Already owns cross-model protein-SAE feature convergence + ESM3 L0H7 +
   causal confirmation. A human must read the Steenwyk full text before committing (WEB-only here).
2. **Goodfire + Arc** — own BSF *and* the Evo 2 SAE; best placed to put BSF-crosscoder on bio first.
3. **Whitfield/Mila (Fournier, Sridhar)** — own the cross-model protein concept-emergence leaderboard;
   a BSF version is an incremental extension they could pre-empt.
4. **Bricken/Anthropic (Jir26) + Nanda (Min25)** — own crosscoder methodology; a "crosscoder goes to
   biology" note is a natural extension for them.
5. **EvolutionaryScale (ESM3)** — any ESM3-centric cross-model claim.

### The two biggest risks
- **R1 — "this is just a benchmark/leaderboard with no mechanism."** Option (a) alone re-finds
  Whitfield with a fancier dictionary. Mitigation: do NOT ship option (a) alone; carry the causal
  shared-vs-unique manifold result (option b) or fold into Task 1.
- **R2 — the comparison is not actually fair / is scooped on fairness too.** If you rank by anything
  touching reconstruction loss (§1a) a reviewer kills it; if you rank by ground-truth recovery you are
  doing Task 1 across models (merge), and Whitfield already did the probe version. Narrow escape: the
  *manifold* variables + the *shared-vs-unique causal* axis, which neither Whitfield nor Steenwyk did.
- (R3, lurking) **Latent-Scaling omission**: a naive crosscoder will report spurious "model-unique"
  features (Min25); shipping that is an instant desk-reject from anyone who knows the method.

### HONEST RECOMMENDATION
**MERGE into Task 1.** Add a **"generality across models"** section that runs Task-1's ground-truth-
recovery metrics (D3) on **2–3 models** (ESM3-open primary; ESM-2 650M as the position-aligned partner;
Evo 2 as the genomic point) as a *shared external yardstick*, plus one **RSA/CKA figure** as a
comparability check — explicitly framed as generality of the Task-1 result, not a model ranking. This
costs little, reuses all infrastructure, strengthens Task 1's external validity, and dodges every scoop
and fairness trap. **Only if a reviewer/advisor demands a separate artifact**, spin out the **two-model
protein crosscoder/USAE on the continuous-manifold variables** (option b, §5 "strongest framing") as a
*workshop* paper, after a human has read the Steenwyk full text to confirm it does not already cover
the manifold/shared-vs-unique angle. **Do NOT pursue:** a general multi-model "which bio model is best"
leaderboard (option a standalone), or an RSA/CKA-on-bio-models paper (option c) — both are crowded and
largely pre-empted.

--------------------------------------------------------------------------------
## 6. OPEN ITEMS FOR A HUMAN
--------------------------------------------------------------------------------
- **Read the Steenwyk full text** ("Structure tokens sharpen the feature vocabulary of protein language
  models", DOI 10.64898/2026.05.12.724593) — the 78%-convergence numbers and the claim scope are WEB-
  only here; it is the #1 scoop and determines whether even the narrow standalone survives.
- Confirm ESM-2 650M vs ESM3-open share a residue-level tokenization suitable for position-aligned
  crosscoder training (needed for option b); if not, option b is limited to within-family pairs.
- Check goodfire.ai/research and recent Mila/Fournier output for an in-flight BSF- or crosscoder-on-bio
  preprint before committing (consistent with the FINAL_FINDINGS pre-submission scoop check).
- Decide capacity-matching unit up front: match **bits/MDL**, not raw feature count, or the cross-model
  comparison inherits the §1a width confound.
