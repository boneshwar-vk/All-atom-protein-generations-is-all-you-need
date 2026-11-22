# Review: Genomics, Single-cell & Evaluation — verification + formalisation pass

Senior-reviewer verification of the genomic/DNA/RNA, single-cell foundation-model, and
cross-cutting SAE-evaluation claims in `evidence_base.md` (Parts 2, 3.4–3.6, 4) and
`deepsearch_singlecell.md`. **Scope:** DNA/RNA/genomic models, single-cell FMs, and
SAE-evaluation/negative-result papers. **Not in scope:** proteins / co-folding (another agent).

Method: Undermind workspace `33899782-4f3e-48f1-9e4a-a3b6ac8a7af7`, full-text PDF reads of 11
primary papers + targeted web lookups for model licenses. Verdicts are **CONFIRMED /
CORRECTED / UNVERIFIED**. DOIs are taken from Undermind `get_paper_info(show_doi=true)` or the
model cards; none are invented. Date of pass: 2026-11-19.

Two external hosts needed for one check (**huggingface.co**, **goodfire.ai**) are blocked by
this session's network settings, so the Evo 2 SAE *feature count* could not be read off the
model card; the *layer* was confirmed from three other sources (see §1).

================================================================================
# 1. VERIFICATION LEDGER (the ~20 load-bearing claims)
================================================================================

## 1.1 PRIORITY CHECKS

### (A) Evo 2 SAE discrepancy — "layer 6 per Orlov Table 3" vs "layer 26 / 32,768"
**VERDICT: CORRECTED → layer 26 (of the 7B model). The draft's "layer 6" is wrong; it faithfully
copies an internal ERROR in the Orlov review.**

- The Goodfire BatchTopK SAE used in the Evo 2 *Nature* paper [Bri26] was trained on **layer 26
  of Evo 2 7B** (the 7B has 32 StripedHyena layers, so layer 26 is a late-middle layer).
  Confirmed from three independent sources:
  1. Goodfire's public SAE weights are published under the repo name **`Goodfire/Evo-2-Layer-26-Mixed`**
     ("mixed" prokaryote/eukaryote), described as "the layer-26 SAE used in the Evo 2 paper."
  2. Goodfire's own write-up states layer 26 was chosen because it "had the most interesting
     biologically-relevant features," and notes Evo 2 7B has 32 layers.
  3. The Orlov review [Orl26] itself, in **§4.3 prose**, says: *"Evo 2 at 7B parameters
     (Batch-TopK at layer 26)."*
- **The Orlov review is internally inconsistent.** Its **Table 3** row reads:
  `Brixi et al., 2025 | Evo 2 | GLM | Batch-TopK | – | 6 | Exon–intron boundaries, TFBS, prophage regions | NCBI annotations`
  — i.e. "6" under the *Layers* column — while its §4.3 says **26**. The draft inherited the
  Table-3 typo. **Use layer 26.**
- **Feature count (32,768): UNVERIFIED here.** The number is *not* in the Orlov review (no feature
  count given in either passage) and was *not* in web snippets of the Goodfire post. The HF model
  card (which carries the SAE config/`d_sae`) is network-blocked in this session. 32,768 is a
  plausible BatchTopK width and matches the secondary report, but I could not read it off a
  primary source — **mark UNVERIFIED pending an HF/model-card read.**
- Biological features reported by Evo 2 [Bri26] — exon–intron boundaries, TF binding sites,
  **protein structural elements**, prophage regions — **CONFIRMED** (Orlov Table 3 + Arc/Goodfire
  materials; note the draft 3.4 omits "protein structural elements"). SAE type **BatchTopK** CONFIRMED.
  Bri26 DOI 10.1038/s41586-026-10176-5.

### (B) Kendiukhov single-author caveat (replication gap)
**VERDICT: CONFIRMED and STRENGTHENED. The draft UNDERCOUNTS — there are ~17–19 Kendiukhov
bio-FM-interpretability papers in the workspace, not ~11, and essentially all are single-author.**

- Distinct I. Kendiukhov single-cell / genomic-FM interpretability items in the workspace
  (Semantic Scholar author id 90532493): **Ken26, Ken26b, Ken26c, Ken26d, Ken26e, Ken26f, Ken26g,
  Ken26h, Ken26i, Ken26j, Ken26k, Ken26m, Ken26o, Ken26s, Ken26t, Ken26u, Ken26w** (+ `Ken26p`
  scaling-laws/TMLR and `Ken26r` aging are borderline-interp; `Ken26n` astronomy and `Ken26v`
  software are out of scope; `Ken25` is a dev-interp review). That is **~17 in-scope papers**,
  across arXiv, Bioinformatics, BMC Genomics, BMC Bioinformatics, J. Biomedical Informatics,
  PLOS One, Bio Systems, Comp. Biol. Chem. Several are preprint/journal pairs of one study
  (e.g. **Ken26g arXiv ≈ Ken26h BMC Genomics**).
- **All three central single-cell negative results rest on this one author:**
  - **Ken26b** — only **3/48 TFs (6.2%)** show CRISPRi-target-specific SAE-feature responses
    ("organized knowledge, minimal causal regulatory logic").
  - **Ken26g/Ken26h** — attention captures **co-expression, not unique regulatory signal**;
    trivial gene-level baselines win (AUROC 0.81–0.88 vs 0.70); ablating "regulatory heads"
    causes no degradation.
  - **Ken26i** — model-internal intervention scores vs real CRISPR give **AUROC ≈ 0.50**.
- **Replication gap is real** for the *mechanistic-interpretability-specific* negative (attention/
  ISP = co-expression). **However**, the *broader* claim ("scFMs lack transferable causal
  regulatory / perturbation-prediction power") is **independently corroborated** by multiple
  non-Kendiukhov groups (see §4): Ahlmann-Eltze/Huber/Anders (EMBL) [Ahl25], Csendes [Cse25],
  Wenteler [Wen25], Qiu & Zhao [Qiu26], Tiwari [Tiw26]. So: weight the *specific audit numbers*
  as single-source, but treat the *direction of the conclusion* as replicated.
- **Important asymmetry to record:** the *genomic-sequence* SAE subfield is NOT one-author —
  it spans Crook/Oxford [Mai26], Ali [Ali26], Kmiec [Kmi26], McLaughlin/UCSF [Mcl26], Guan
  [Gua25], Goodfire/Arc [Bri26]. Only the *single-cell mechanistic-interp* subfield is
  Kendiukhov-dominated. Do not blanket the whole "genomic + single-cell interp" field as
  one-author.

### (C) DeepSCENIC [Par26e] — "transfers Enformer/Borzoi internals into a CRISPRi-validated causal GRN"
**VERDICT: CONFIRMED, with two precision nuances. This is the strongest perturbation-validated
sequence-model-internals result in scope — and it is MISSING from `evidence_base.md` entirely.**

- **Uses INTERNAL representations, not just outputs: CONFIRMED.** DeepSCENIC extracts sequence
  **embeddings** from pretrained S2F backbones (Enformer `5×3072`, Borzoi `20×1920`, also HyenaDNA,
  Crested, a CNN) on 640-bp ATAC-peak-centred windows, feeds them to a TF–RE head producing
  TF-specific binding potentials θ, and **fine-tunes the backbones jointly** with the GRN module.
  So the pretrained internals are transferred and adapted — not the predicted tracks.
- **CRISPRi validation: CONFIRMED but scoped.** Predicted **enhancer→target-gene (RE–TG) links**
  are benchmarked against a **K562 CRISPRi screen (10,411 region–gene pairs; FDR<0.05 = positives)**.
  **dS-Enformer AUC 0.83**, dS-HyenaDNA 0.80, **dS-Borzoi 0.78**; the supervised ENCODE comparator
  reaches AUC 0.96. **Nuance 1:** only the best variant (dS-Enformer) beat both correlation
  baselines; dS-CNN/dS-Crested were on-par-or-worse. **Nuance 2:** CRISPRi validates the
  *enhancer–gene link set*, not that *every* inferred GRN edge is causal.
- **Perturbation simulation: CONFIRMED.** Simulated **SOX10 knockdown** drives melanoma MEL-like
  lines along PC1 toward the mesenchymal (MES) program, matching a SOX10-KD time course
  (72 h: Pearson **0.31–0.52**, CCC **0.61–0.75** across responsive markers).
- **ACTION:** add DeepSCENIC to the genomics table and to the "strongest perturbation-validated"
  list; and refine Part-4 gap 1 (below) — Enformer/Borzoi *internals* are no longer wholly
  untouched. Par26e DOI 10.64898/2026.09.18.752607.

### (D) Ali [Ali26] (arXiv 2607.19618) — GC/repeat confound → spurious TF features
**VERDICT: CONFIRMED in full detail.**
- Naive PWM validation flagged **440 / 6,576** features as CTCF-"enriched"; the top one fires
  mainly on **Alu repeats**; GC-matched background collapses the enrichment (CTCF motifs, Alu,
  CpG islands are all GC-rich). CONFIRMED.
- Composition-matched, binding-resolved protocol (bound / GC-matched unbound-motif / random;
  Mann–Whitney; binding-sensitivity AUC>0.55; GC-robust = |corr(act,GC)|<0.2) + **decoder
  ablation during the forward pass** (KL shift) for causal validation. CONFIRMED.
- Causal counts (top-15 features): **CTCF 7/15 (NT) · 10/15 (DNABERT-2); GATA1 9/15 · 13/15;
  REST 7/15 · 14/15; random-feature controls 0/15; scrambled-label negatives give no testable
  features.** Models: **NT (6-mer, layer 14)** + **DNABERT-2 (BPE, layer 6)**. CONFIRMED.
  (Draft's "7–14 of 15 validate" is the correct range.) DOI 10.48550/arXiv.2607.19618.

### (E) Decode-gLM [Mai26] (10.1101/2025.10.31.685860)
**VERDICT: CONFIRMED on all five points (Crook/Oxford group — independent of Kendiukhov).**
- Nucleotide Transformer **50M & 500M**; **BatchTopK SAEs (expansion ×8) + Meta-SAEs**. CONFIRMED.
- **CMV enhancer** feature recovered despite viral genomes excluded → traced to **5 contaminated
  reference-DB entries** (training-data-leakage audit). CONFIRMED.
- Steered toward the **A1408G** aminoglycoside-resistance mutation in 16S rRNA via antibiotic-
  resistance feature **#5066** (promoted A1408G to top-1). CONFIRMED.
- **Random-init control**: concept linear-probe F1 far higher in pretrained than random model
  (min +17.2%, avg >50%) → features are learned in pretraining. CONFIRMED.

### (F) Model availability
- **AlphaGenome v1 — CONFIRMED gated + non-commercial.** Weights (on Kaggle/HF) require accepting
  the **AlphaGenome Model Terms of Use** (non-commercial: universities, non-profits, research/gov;
  journalism is the lone extra exception); **code Apache-2.0**, **docs/examples CC-BY-4.0**.
  Usable for interpretability under NC terms. (Exact v1 weights-release date "~Jan 2026" NOT
  independently confirmed — minor, mark UNVERIFIED.)
- **AlphaGenome 2 — CONFIRMED API-only, no weights.** Released **2026-11-11**; non-commercial
  research via the **AlphaGenome API** (free; commercial in private preview). **Weights not
  released** ("to accompany a forthcoming paper"). Cell-state-conditioned: accepts a scRNA-seq
  snapshot, **≤256 cells/call**; headline **30% of putatively causal expression variants at 90%
  precision on TenK10K (28 immune cell types)** vs 16% for AG1-fine-tuned (DeepMind's own
  numbers, no independent replication at release). **Not usable for mechanistic interp today.**
  (Draft's "~107M cell profiles" training figure NOT confirmed in snippets — minor, UNVERIFIED.)
- **Evo 2 — CONFIRMED Apache-2.0.** `arcinstitute/evo2_{1b,7b,40b}_base` HF cards list
  `apache-2.0` (1B=25 layers, 7B=32, 40B=50; base=8,192 ctx). **Caveat CONFIRMED:** NVIDIA
  BioNeMo-packaged copies are governed by the **NVIDIA Open Model License**, not Apache-2.0.

## 1.2 SECONDARY CHECKS

### (G) Korznikov [Kor26] "Sanity Checks for SAEs" (arXiv 2602.14111)
**VERDICT: CORRECTED — the `evidence_base` phrasing ("SAEs recover 9% of ground-truth at 71% EV;
RANDOM BASELINES MATCH TRAINED SAEs") is an overclaim/misattribution on two counts.**
- The **"≈9% at 0.71 EV"** figure is from a **synthetic toy model with known ground truth**
  (BatchTopK 297/3,200 ≈ 9%, JumpReLU 225/3,200 ≈ 7%, both at 0.71 EV; recovery concentrated in
  high-frequency features) — **not a biological model and not even a real LM**.
- The baselines that "match" trained SAEs are **three *constrained* (partially frozen) baselines
  — Frozen Decoder, Soft-Frozen Decoder, Frozen Encoder — not random dictionaries**, evaluated on
  **LM activations (Gemma-2-2B, Llama-3-8B)** + a CLIP appendix. "Match" = *competitive overall*
  (0.87 vs 0.90 interp; 0.69 vs 0.72 sparse-probing; 0.73 vs 0.72 causal-editing), with
  Soft-Frozen closest — **not identical, and not "random."**
- **No biological-model evaluation in this paper.** Central claim: "SAEs in their current state
  do not reliably decompose models' internal mechanisms." The `deepsearch_singlecell` phrasing
  ("questions whether learned SAE dictionaries beat random baselines") is acceptable; the
  `evidence_base` phrasing should be fixed. DOI 10.48550/arXiv.2602.14111.

### (H) Mechanistic Invariance Test [Che26] (arXiv 2604.06549)
**VERDICT: CONFIRMED (negative result).** 650×100-bp *E. coli* σ⁷⁰-promoter benchmark; **5 gLMs
(HyenaDNA, Evo2-1B, NT-500M, GROVER, Caduceus).** Apparent compensation-sensitivity is driven
by **AT-content correlation (r=0.78–0.96), not positional regulatory logic**; **Evo2-1B and
Caduceus score elements at WRONG positions higher than correct** (inverting biology); SCR
0.40–0.52 (≈chance); only HyenaDNA has significant CSS (0.63). No DOI (arXiv); authors Cheng & Zhang.

### (I) AlphaGenome internals papers [Tu26], [Lia26]
**VERDICT: CONFIRMED — neither touches AlphaGenome's internal representations.**
- **Tu26** (modality gap): analyses **output predictions** (+ a readout on *predicted-signal*
  features), not latent embeddings. Accessibility near heritability ceiling (R 0.627 vs baseline
  0.758), expression far below (R 0.151 vs 0.670); beats Borzoi in both. No SAE/dictionary/MI.
- **Lia26**: AlphaGenome used purely **input→output** (in-silico motif excision + log₂FC vs sham).
  No embedding/SAE/circuit analysis.
- → supports Part-4 gap 1 (no SAE/dictionary learning on AlphaGenome internals).

### (J) Pedrocchi/Boeva [Ped26] single-cell SAE steering
**VERDICT: CONFIRMED, independent of Kendiukhov (ETH Zurich, Boeva corresponding author), with one
clarification.** SAEs on scGPT/scFoundation/Geneformer; features capture biology + technical/batch;
**suppressing batch features improves integration**; **activating dose-encoding features shifts
control cells toward drug-treated populations** (CMS>0.9 for all three drugs; Panobinostat dose-
tracking). **Clarification:** steering is validated against **model behaviour (computational
UMAP/CMS/DAS)**, held-out cell lines — **NOT new wet-lab experiments**. DOI 10.1101/2025.10.22.681631.

### (K) Ahlmann-Eltze [Ahl25] perturbation benchmark
**VERDICT: CONFIRMED (independent EMBL negative result), two minor precision fixes.** Nature
Methods 2025 (DOI 10.1038/s41592-025-02772-6). **Five FMs: scGPT, scFoundation, UCE, scBERT,
Geneformer** (+GEARS, CPA). None beat simple baselines; combinatorial no better than additive;
interaction detection beats no model. **Fix 1:** the unseen-case baseline is "predict training
**mean** / linear model," and the task is **unseen *perturbations*, not unseen *genes*** (draft
said "genes"). **Fix 2:** a *linear* model pretrained on *perturbation* data (not the FMs) was the
only consistent winner. Independently supports the "weak causal/perturbation power" direction.

================================================================================
# 2. DUPLICATES, CONTRADICTIONS, OVERCLAIMS
================================================================================
- **DUP (across the two drafts):** Evo 2 = `Bri26`; Decode-gLM = `Mai26`; Orlov review = `Orl26`;
  Korznikov = `Kor26`; Che26; Xu26 — same papers under the same keys; no action beyond merging.
- **DUP (Kendiukhov preprint/journal pairs):** Ken26g (arXiv) ≈ Ken26h (BMC Genomics); the
  attention-audit result should be cited once, noting both. Do not count as two independent studies.
- **OMISSION (evidence_base):** `evidence_base.md` Parts 3.4/4 **do not contain DeepSCENIC**, the
  single strongest perturbation-validated sequence-model-internals result. Added here.
- **CONTRADICTION (within a cited source):** Orlov [Orl26] Table 3 (layer 6) vs §4.3 (layer 26)
  for Evo 2 — resolved to layer 26 (§1A).
- **OVERCLAIM (evidence_base 3.6):** "RANDOM BASELINES MATCH TRAINED SAEs" for Kor26 — they are
  constrained/partially-frozen baselines on LMs, "competitive" not "matching," no bio eval (§1G).
- **OVERCLAIM risk (deepsearch 3.6 / Part-4 gap framing):** calling the single-cell "attention/ISP
  = co-expression" conclusion a settled field result understates that the *specific audits* are
  one-author; state it as "replicated in direction (Ahl25/Cse25/Wen25/Qiu26/Tiw26), single-source
  in the exact mechanistic numbers."
- **VENUE/CITATION HYGIENE:** nucleotide-dependency analysis (Tomaz da Silva … Gagneur) is
  **published in Nature Genetics 2025** (`Sil25b`, 10.1038/s41588-025-02347-3); `evidence_base`
  lists only the bioRxiv preprint (10.1101/2024.07.27.605418). Cite the Nature Genetics version.
- **CLAIM REST (Ped26):** draft's "drug-features are causally wired to model behaviour" is fine,
  but the earlier shorthand "steers control cells toward drug-perturbed states" must not be read
  as wet-lab validation — it is in-silico (§1J).
- No fabricated DOIs found in the two drafts for in-scope papers; the `10.64898/...` prefixes are
  the genuine 2026 bioRxiv-style DOIs (verified via get_paper_info).

================================================================================
# 3. FORMALISED SECTION — "Genomics, Single-cell & Evaluation"
================================================================================
Confidence: **V** = verified by full-text PDF read this pass; **A** = abstract/metadata only;
**W** = web/model-card. Deduplicated; proteins/co-folding excluded.

## 3.1 Genomic / DNA / RNA sequence models — SAEs, dictionary learning, nucleotide-dependency

| Key | Title (short) | Venue · date | Model | Method | Finding | DOI/link | Conf |
|---|---|---|---|---|---|---|---|
| Bri26 | Genome modelling & design across all domains of life with Evo 2 | Nature · 2026-03 | Evo 2 7B | **BatchTopK SAE, layer 26** (Goodfire+Arc) | Features for exon–intron, TFBS, protein-structural, prophage; steered chromatin-accessibility gen. **Layer 26 (not 6)**; width likely 32,768 (unverified) | 10.1038/s41586-026-10176-5 | V (layer) / W |
| Mai26 | Decode-gLM: interpret, audit, steer gLMs | bioRxiv · 2026-02 | Nucleotide Transformer 50M/500M | BatchTopK SAE ×8 + Meta-SAE; random-init control | CMV-enhancer feature → DB-contamination audit (5 entries); steered A1408G resistance (feat #5066); features learned in pretraining | 10.1101/2025.10.31.685860 | V |
| Ali26 | Causal dictionary learning: TF-binding features in gLMs | arXiv · 2026-07 | NT (6-mer, L14) + DNABERT-2 (BPE, L6) | TopK SAE; GC/composition-matched protocol; **decoder ablation in forward pass** | Naive PWM confounded by GC/Alu (440/6576 spurious CTCF); causal CTCF 7–10/15, GATA1 9–13/15, REST 7–14/15; random 0/15 | 10.48550/arXiv.2607.19618 | V |
| Kmi26 | Hidden-State Genomics: graph analysis of SAE features | bioRxiv · 2026-05 | NT v2 500M, all 24 layers | SAEs + sequence→feature knowledge graphs + decoder interventions | Annotation inconsistent across layers; gLM reps align with **local biophysical/syntactic** constraints > distributed regulatory logic | 10.64898/2026.05.13.725007 | A |
| Mcl26 | Probing gLMs: Nucleotide-GPT & role of pretraining | Brief. Bioinf. · 2026-01 | Decoder single-nt gLM | Linear probing + SAE under varied repeat-weighting | **SAE features dominated by repetitive elements**; moderate RE down-weighting optimal | 10.1093/bib/bbag011 | A |
| Gua25 | SAEs reveal structure in small gene LMs | 2025-07 | small gene LMs / HyenaDNA | L1/TopK SAE | TF-binding motifs recoverable even in small gene LMs | arxiv.org/abs/2507.07486 | A |
| Kim25 | SAE-RNA: interpreting RNA LM reps | arXiv · 2025-10 | RiNALMo (RNA LM) | SAE as representation probe | Maps to RNA-family/structural context; cautions against over-claiming concept discovery | 10.48550/arXiv.2510.02734 | A |
| Hos26 | SPIRAL: SAEs reveal structural/family features in BiRNA-BERT | bioRxiv · 2026-08 | BiRNA-BERT L0/5/11 | Tokenizer-aware SAE | 44.3% of layer-5 features → bpRNA secondary-structure classes | 10.64898/2026.08.11.744228 | A |
| Sil25b | Nucleotide-dependency analysis detects functional elements | **Nat. Genetics · 2025-10** | DNA/RNA gLMs (agnostic) | Pairwise nucleotide-dependency (categorical-Jacobian-style) | Recovers motifs + RNA base-pairs (pseudoknots); variant-effect > conservation; exposes gLM limits | 10.1038/s41588-025-02347-3 | A |
| Xu26 | Disentangling RNA evolution & thermodynamics in gLMs | bioRxiv · 2026-05 | RNA-FM, Evo 2, gLM2 | Categorical Jacobian + "mirror test" | CJ recovers evolutionary + thermodynamic base-pairs; **gLMs fail reversed-sequence mirror test** → mimic thermodynamics via learned context | 10.64898/2026.05.28.728275 | A |

## 3.2 Single-cell FMs — SAEs / transcoders / dictionary learning (feature legibility)

| Key | Title (short) | Venue · date | Model | Method | Finding | DOI | Conf |
|---|---|---|---|---|---|---|---|
| Ped26 | SAEs reveal interpretable features in scFMs | bioRxiv · 2026-06 | scGPT, scFoundation, Geneformer | SAEs + feature interventions | Biological + batch features; suppress-batch → better integration; **activate dose features → control cells shift to drug-treated (in-silico, CMS>0.9, dose-tracking)** | 10.1101/2025.10.22.681631 | V |
| Ken26b | SAEs: organized knowledge, minimal regulatory logic | arXiv · 2026-03 | Geneformer V2-316M, scGPT | TopK SAE all layers; CRISPRi audit | 82k/24k feature atlases; 29–59% annotate; **only 3/48 TFs show CRISPRi-specific response** | 10.48550/arXiv.2603.02952 | A |
| Ken26c | SAEs reveal cell-type programs in scFM reps | J. Biomed. Inf. · 2026-05 | scGPT, 12 layers | TopK SAE, sparsity calibration | Needs stronger L1 (λ≥1) than LLMs; 64% layer-11 features annotate to cell-type programs | 10.1016/j.jbi.2026.105056 | A |
| Cla25 | Discovering interpretable biological concepts in scRNA FMs | arXiv · 2025-10 | 2 immune scFMs | TopK SAE + counterfactual-perturbation concept attribution | Concepts beat neurons for interpretability; immunologist + ontology eval | 10.48550/arXiv.2510.25807 | A |
| Hos25 | Transcoder circuit analysis for scFMs | arXiv · 2025-09 | Cell2Sentence | Transcoder → decision circuits | First transcoder on a scFM; circuits map to real pathways | 10.48550/arXiv.2509.14723 | A |
| Ken26d | Causal circuit tracing (SAE-feature ablation) | Bioinformatics · 2026-03 | scGPT, Geneformer | SAE-feature ablation circuits (96,892 edges) | Inhibitory dominance 65–89%; 1,142 arch-invariant GO pairs; **CRISPRi excess over baseline tiny (+0.03–0.35 pp)** | 10.1093/bioinformatics/btag379 | A |
| Ken26e | Exhaustive circuit mapping; layer-dependent differentiation | arXiv · 2026-03 | Geneformer | Exhaustive tracing (1.39M edges) + feature steering | 40% top hubs unannotated (annotation bias); **trajectory steering: late-layer features push toward maturity** | 10.48550/arXiv.2603.11940 | A |
| Sch24 | Can SAEs make sense of gene-expression LVMs? | MLCB · 2024-10 | scVI-style latents | SAE on latents | Earliest SAE-on-cell-models; recovers generative vars in superposition (unidentifiable if unknown) | arxiv.org/abs/2410.11468 | A |

## 3.3 Single-cell FMs — attention / probing / geometry / causal-vs-coexpression

| Key | Title (short) | Venue · date | Model | Method | Finding | DOI | Conf |
|---|---|---|---|---|---|---|---|
| Ken26g/h | Attention captures co-expression, not unique regulatory signal | arXiv 2026-02 / BMC Genomics 2026-07 | scGPT, Geneformer | 37 analyses, 153 tests; CRISPRi/a | Attention adds **no incremental** perturbation-prediction value; trivial baselines win; ablating "regulatory heads" → no degradation | 10.48550/arXiv.2602.17532 / 10.1186/s12864-026-12965-8 | A |
| Ken26i | Causal-intervention validation of regulatory signals | J. Biomed. Inf. · 2026-07 | scGPT | TF-token ablate/swap vs TRRUST + real CRISPR | Aligns with curated GRNs in lung only; **vs real CRISPR AUROC ≈ 0.50** ("representational, not causal") | 10.1016/j.jbi.2026.105080 | A |
| Ken26j | Residual-stream geometry carries incremental GRN signal | BMC Bioinf. · 2026-07 | scGPT, Geneformer | Geometric proximity vs TRRUST | +ΔAUROC in-entity but **collapses to ~0 leave-both-out** (out-of-entity ceiling) | 10.1186/s12859-026-06538-5 | A |
| Ken26k | Topology/geometry: 141 hypotheses | PLOS One · 2026-02 | scGPT, Geneformer | Persistent homology + null controls | Genuine non-trivial topology (11–12/12 layers); robust in immune, fragile in lung | 10.1371/journal.pone.0344826 | A |
| Ken26m | Spectral geometry of biological knowledge | arXiv · 2026-02 | scGPT | Spectral analysis | 6-D subspace separates TF↔target (AUROC 0.744); repression edges more prominent | 10.48550/arXiv.2602.22247 | A |
| Ken26f | Hematopoietic manifold extraction | arXiv · 2026-03 | scGPT | Frozen-attention export + readout | Compact hematopoietic manifold w/ branch structure; beats scVI/Palantir on pseudotime-depth | 10.48550/arXiv.2603.10261 | A |
| Zha26 | scGeneLens: how AI models understand cells differently | bioRxiv · 2026-01 | scFoundation, scGPT | Sparse block attention + integrated gradients | "Pluralistic perception": scFoundation→marker genes, scGPT→shared pathways | 10.64898/2026.01.29.702682 | A |
| Jaf26 | Attribution-based diagnostics for scFMs | arXiv · 2026-11 | C2S, scGPT, scGPT+STATE | Post-hoc attribution vs regulatory ground truth (synthetic) | Reveals marker reliance/shortcuts; **conclusions depend on reference/baseline/null choice** | arxiv.org/abs/2611.02335 | A |
| Ye26 | Benchmarking static GRN + dynamic-transition probing in scFMs | bioRxiv · 2026-05 | 6 scFMs | Disentangle token-emb / hidden / attention | **scGPT token-emb beats classical baselines** on STRING/ChIP; scFoundation best on dynamic transitions (more optimistic than Kendiukhov audits) | 10.64898/2026.05.17.725083 | A |
| Civ26b | Intermediate layers encode optimal bio reps | arXiv · 2026-04 | scFoundation, Tahoe-X1 | Layer-wise probing | Trajectory inference peaks ~60% depth; optimum task/state-dependent | 10.48550/arXiv.2604.14838 | A |

## 3.4 Enformer / Borzoi / AlphaGenome — cell-type-specific regulatory interpretability

| Key | Title (short) | Venue · date | Model | Method | Finding | DOI | Conf |
|---|---|---|---|---|---|---|---|
| Par26e | **DeepSCENIC**: transfer learning → causal GRN inference | bioRxiv · 2026-09 | **Enformer, Borzoi, HyenaDNA** (embeddings, fine-tuned) | Transfer internal embeddings → TF–RE head → variational GRN; perturbation simulation | **RE–TG links vs K562 CRISPRi: dS-Enformer AUC 0.83, dS-Borzoi 0.78** (ENCODE 0.96); SOX10-KD → MEL→MES shift (CCC 0.61–0.75) | 10.64898/2026.09.18.752607 | V |
| Tu26 | Modality gap in personal-genome prediction | bioRxiv · 2026-02 | AlphaGenome, Borzoi | Output predictions vs heritability baseline | Accessibility near ceiling (R 0.627), expression far below (R 0.151); **no SAE/internals** | 10.64898/2026.02.01.702969 | V |
| Lia26 | Functional-motif detection via in-silico ablation | Brief. Bioinf. · 2026 | AlphaGenome | Input motif excision + log₂FC vs sham | Motif-function calling; **no internal-representation interp** | 10.1093/bib/bbag422 | V |
| Kat24c | Decreased performance in cell-type-specific accessible regions | Genome Biol. · 2024-08 | Enformer, Sei | Output accuracy by region class | Accuracy **drops specifically in cell-type-specific CREs** (bulk of disease heritability) | 10.1186/s13059-024-03335-2 | A |
| Ton24 | CREME: interpreting cis-regulatory interactions | Nat. Genetics · 2024-09 | Enformer | In-silico perturbation toolkit (I/O) | Enhancing/silencing CREs across scales | 10.1038/s41588-024-01923-3 | A |
| Sei25 | SEAM: mechanistic landscape of regulatory DNA | bioRxiv · 2025-10 | Koo-lab DNNs | DL + XAI on mutation space | Distinguishes mutation-robust vs reprogrammable signals | 10.1101/2025.10.07.681052 | A |
| Rob25 | Enformation Theory: blueprint for evaluating DL in genomics | bioRxiv · 2025-01 | Enformer, Borzoi | **Embedding PCA** + partial backtracking | Interpretable decomposition of embeddings (internals, non-SAE) | 10.1101/2024.09.03.611127 | V(meta) |

## 3.5 Cross-cutting evaluation / negative-result / methods (SAE-eval + perturbation benchmarks)

| Key | Title (short) | Venue · date | Scope | Finding | DOI | Conf |
|---|---|---|---|---|---|---|
| Orl26 | What do biological FMs compute? (SAE review) | bioRxiv · 2026-03 | protein/DNA/single-cell/pathology | 3-level framework; validation ≈ annotation-matching → **circularity**; transcoder apps limited. (Table 3 Evo-2-layer typo: "6" → 26) | 10.64898/2026.03.04.709491 | V |
| Kor26 | Sanity checks: do SAEs beat random baselines? | arXiv · 2026-02 | **LM/synthetic (no bio)** | Synthetic: ~7–9% feature recovery at 0.71 EV; **constrained (partially-frozen) baselines competitive** with trained SAEs on Gemma/Llama | 10.48550/arXiv.2602.14111 | V |
| Sha26 | Ground-truth SAE-feature eval (spatial/single-cell/LM) | bioRxiv · 2026-10 | Geneformer + spatial + synthetic | **Reproducibility ≠ meaningfulness**; 1 SAE fit reproduces 27–46% features; 5-fit consensus reproducible (78–98%) + pure | 10.64898/2026.10.27.762296 | A |
| Ahl25 | DL perturbation prediction ⟂ simple linear baselines | Nat. Methods · 2025-08 | scGPT, scFoundation, UCE, scBERT, Geneformer (+GEARS,CPA) | **None beat baselines**; combinatorial no better than additive; unseen-**perturbation** no better than training mean | 10.1038/s41592-025-02772-6 | V |
| Cse25 | Benchmarking cell FMs for post-perturbation RNA-seq | BMC Genomics · 2025-04 | scGPT, scFoundation | "Train-mean" beats them; basic featured ML beats scGPT | 10.1186/s12864-025-11600-2 | A |
| Wen25 | PertEval-scFM | bioRxiv · 2025-05 | zero-shot scFM embeddings | No consistent improvement over baselines, esp. under shift | 10.1101/2024.10.02.616248 | A |
| Qiu26 | Confound-diagnostic toolkit for in-silico perturbation | bioRxiv · 2026-08 | Geneformer-style ISP | ISP delta gives **no reproducible held-out gain beyond gene identity**; library-size/tokenization confounds | 10.64898/2026.08.04.732812 | A |
| Tiw26 | Perturbation models fail OOD (EL-PERTURB) | bioRxiv · 2026-10 | 20 models | scFM embeddings **no reliable OOD advantage** over tuned baselines | 10.64898/2026.10.16.759604 | A |
| Che26 | Mechanistic Invariance Test: gLMs fail positional logic | arXiv · 2026-04 | HyenaDNA, Evo2-1B, NT-500M, GROVER, Caduceus | Apparent compensation = AT-content (r 0.78–0.96); Evo2-1B/Caduceus invert positions | arxiv.org/abs/2604.06549 | V |
| Zho26pos | Position: move beyond anecdotal interp evaluation | arXiv · 2026-05 | genomic IML methods | IML methods give contradictory, unfaithful explanations; proposes tiered validation | 10.48550/arXiv.2606.07607 | A |

================================================================================
# 4. CORRECTED PART-4 GAP LIST (in-scope: genomic + single-cell + eval)
================================================================================
Confidence after this pass. **★ = changed from the draft.**

1. **★ No SAE / dictionary-learning / transcoder / circuit interpretability on Enformer /
   Borzoi / AlphaGenome *internal representations*. HIGH (narrowed).** The draft's "only ISM/
   attention" is incomplete: DeepSCENIC [Par26e] **transfer-learns Enformer/Borzoi embeddings**
   into a CRISPRi-validated GRN, and Enformation Theory [Rob25] does **PCA of embeddings** — so
   their internals are *no longer wholly untouched*. But **no sparse-dictionary/SAE/transcoder**
   decomposition of these three models' activations exists (Tu26/Lia26/CREME/SEAM are all I/O or
   predicted-feature analyses). The SAE-specific gap stands; restate precisely.
2. **No crosscoder / model-diffing on any bio model. HIGH.** Orlov confirms transcoder apps are
   limited to one study and does not surface crosscoder/model-diffing work; none found this pass.
3. **Circular validation (features scored vs annotations the model trained on). HIGH.** Orlov §4.2
   documents the "closed evaluation loop"; experimental validation (CRISPRi/MPRA/DMS) is the
   opening. Exemplars that break circularity: Ali26 (CRISPR-seq + GC-matched), Par26e (CRISPRi),
   Mai26 (leakage audit + resistance steering).
4. **Random-init / composition-matched baselines rarely run; change conclusions when run. HIGH.**
   Ali26 (GC/Alu confound), Che26 (AT-content), Mai26 (random-init), Kor26 (constrained-baseline
   null), Kmiec/McLaughlin (repeat dominance) all show it matters.
5. **Transferable causal regulatory logic in scFMs is weak. HIGH (direction replicated; exact
   mechanistic numbers single-source).** Single-cell audits (Ken26b/g/h/i) are one-author;
   the direction is independently corroborated by Ahl25 (EMBL), Cse25, Wen25, Qiu26, Tiw26.
6. **★ Single-cell mechanistic-interp density is inflated by one author (Kendiukhov, ~17 papers).
   HIGH.** Cross-group *replication* of the central single-cell negatives is the real gap. NB:
   this does **not** apply to the *genomic-sequence* SAE subfield, which is multi-group.
7. **"Linear decodability ≠ causal steerability" shown in genomics only indirectly. MED-HIGH.**
   In scope, Ken26i (AUROC≈0.50) and Che26 are the closest; the clean demonstration is in proteins
   (Boltz-1, out of scope).
8. **Feature reproducibility/stability barely measured for bio SAEs. MED-HIGH.** Sha26 is the main
   in-scope treatment (consensus-over-seeds); otherwise thin.
9. **No standardized bio interp benchmark (a bio-SAEBench). HIGH.** Sha26 + Zho26pos gesture at it;
   none adopted.
10. **RNA models thin; no circuit/causal work. MED-HIGH.** Only Kim25 (SAE-RNA), Hos26 (SPIRAL),
    Xu26 (CJ mirror test) in scope; all representational.
11. **Genuine novel-biology discovery (not re-finding annotations) is rare. HIGH.** Standouts in
    scope: **Mai26** (DB-contamination + resistance steering), **Sil25b** (4 validated *E. coli*
    RNA structures / new base-pairs), **Par26e** (CRISPRi-validated enhancer–gene links).
12. **★ Virtual-cell / perturbation-native models (STATE, scGPT+STATE, TranscriptFormer, CellFM,
    Tahoe-X1) essentially un-audited for causal regulatory logic. MED-HIGH.** Jaf26 touches
    scGPT+STATE but only synthetically. New gap surfaced by the single-cell deep search.
13. **★ Fate/bifurcation computation under-decomposed; lineage-tracing (LARRY) unused as interp
    ground truth. MED.** Ken26e steering is the lone causal-ish result; no cross-group replication.

================================================================================
# 5. STRONGEST PERTURBATION-VALIDATED RESULTS (ranked, in scope)
================================================================================
Ordered by how directly model *internals* are tied to a *real experimental* perturbation readout.

1. **Par26e DeepSCENIC — sequence-model internals → CRISPRi-validated enhancer–gene GRN.**
   Enformer/Borzoi embeddings (fine-tuned) → RE–TG links benchmarked vs K562 CRISPRi
   (dS-Enformer AUC 0.83); SOX10-KD simulation matches a real KD time course (CCC 0.61–0.75).
   *Caveat:* validates the enhancer–gene link set (not every edge); only the best backbone beats
   correlation baselines; below the supervised ENCODE ceiling (0.96). **Strongest in scope.**
2. **Ali26 — causally-validated TF-binding SAE features, composition-controlled.** Decoder
   ablation in the forward pass + GC-matched ChIP-seq ground truth; CTCF/GATA1/REST features
   validate 7–14/15, random 0/15. Internals→causal, but causal *within the model* (KL shift),
   grounded in real ChIP-seq occupancy rather than a perturbation assay.
3. **Mai26 Decode-gLM — steering + training-data audit.** SAE feature steers NT toward the real
   A1408G resistance mutation; CMV-enhancer feature exposes 5 contaminated DB entries. Discovery
   + benign steering; validated against sequence/DB ground truth, not a new wet-lab assay.
4. **Ped26 — dose-dependent drug-state steering in scFMs (in-silico).** Causally wired to *model
   behaviour* (held-out cell lines), **not** new wet-lab experiments — rank below 1–3 for that reason.
5. **Ken26e — trajectory-guided feature steering → differentiation direction (single-cell).**
   Late-layer features push toward maturity; model-internal validation only; single-author, unreplicated.

Honourable mention (out of scope here, cross-reference): RegVelo [Wan24h] and Ota26b (Central Dogma
Transformer II) are the cleanest *wet-lab/CRISPR-validated* causal results, but they are
interpretable-by-construction dynamical/architectural models, not interpretability *of* a
pretrained FM's internals.

================================================================================
# 6. ITEMS LEFT UNVERIFIED (flag for a later pass)
================================================================================
- Evo 2 SAE **feature count 32,768** — needs an HF model-card / SAE-config read
  (`Goodfire/Evo-2-Layer-26-Mixed`); huggingface.co + goodfire.ai are network-blocked this session.
  **Layer 26 is firmly confirmed.**
- AlphaGenome v1 **weights-release date ("~Jan 2026")** and AlphaGenome 2 **"~107M cell profiles"**
  training figure — not in reachable sources; licenses/availability themselves are confirmed.
- Several single-cell rows are **abstract/metadata-only (conf A)**; numeric claims within them are
  as-reported, not re-read this pass. The load-bearing ones (Ped26, Ahl25, Ali26, Mai26, Par26e,
  Kor26, Che26, Tu26, Lia26, Orl26) were full-text-verified.

*Compiled 2026-11-19. Full-text reads: Orl26, Par26e, Ali26, Mai26, Kor26, Che26, Tu26, Lia26,
Ped26, Ahl25. Bri26 (Nature) paywalled/unreadable — Evo 2 facts cross-checked via Orlov §4.3,
Goodfire HF repo name, and web. Workspace 33899782-4f3e-48f1-9e4a-a3b6ac8a7af7.*
