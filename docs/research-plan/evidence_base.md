# Evidence base — bio-model interpretability project (consolidated 2026-11-19)

Consolidation of four completed research/planning reports produced in this session.
"UNVERIFIED" marks anything not confirmed against a primary source by the reporting
agent. Several agents could not open arxiv.org / biorxiv.org / huggingface.co directly
and relied on Undermind PDF reads, GitHub clones, or search snippets — flagged inline.

This file + `technical_plan.md` + the two forthcoming deep-search reports
(`deepsearch_drug_plm.md`, `deepsearch_singlecell.md`) are the inputs for the
review/formalise stage.

================================================================================
# PART 1 — METHODS: what "J++ Jacobians" and "BSF" are
================================================================================

## 1.1 J++ Lens (the "J++ Jacobians")
- Name: "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses",
  Kola Ayonrinde & Jack Lindsey (Anthropic). LessWrong post, Oct 2026; reportedly
  also in the Oct 2026 Circuits Update (UNVERIFIED — page not opened).
  Post: lesswrong.com/posts/nc9dHfcB22JdMzGbr/...
- Code: github.com/safety-research/jpp_lens (Apache-2.0). Lenses on HF: koayon/jpp-lenses.
- Parent method: Jacobian lens (J-lens), Gurnee et al., "Verbalizable Representations
  Form a Global Workspace in Language Models", arXiv 2607.15495, Jul 2026 (senior
  author Lindsey). Code: github.com/anthropics/jacobian-lens (Apache-2.0, "reference, not maintained").
- Intermediate: R-Lens (Blank & Bhatia), source of the LRP backward rules.
- NOT the same as: attribution-graph/circuit-tracing Jacobians, Jacobian SAEs, or the
  protein-LM "categorical Jacobian" (Zhang et al. PNAS 2024) — naming collision to manage.

How J-lens works (verified from PDF + code):
- J_l = E over prompts, source positions t, target positions t'>=t of
  d h_final,t' / d h_l,t  → one d_model x d_model matrix per source layer.
- Readout: softmax(W_U . norm(J_l h)); logit lens is the special case J_l = I.
- Estimator: inject one-hot cotangent at all valid target positions, backprop, sum over
  later targets, average over sources. ~100 prompts saturate. Default takes gradient
  from the penultimate-layer residual. Cost: 1 fwd + ceil(d_model/dim_batch) bwd per prompt.
- "J-space": activations ≈ sparse (k<=25) non-negative combo of J-lens vectors (rows of W_U J_l).
- Needs full weights + autograd, no training. Assumes: single residual stream, causal
  decoder (t'>=t), unembed = final norm + LM head over a token vocab.

J++ additions (from code):
1. Jacobian Filtering: per layer, normalise→centre→PCA→k-means++ into E=8 clusters;
   average an "expert" Jacobian per cluster; combine with learned signed weights
   (unit L2) fitted by Adam on a recall@k surrogate over labelled (prompt, known
   intermediate token) items → SUPERVISED step. Linearity trick assumes RMSNorm+linear unembed.
2. LRP backward pass (detach RMSNorm scale; detach SiLU/GELU gate; half-rule on gated MLP;
   attention keeps normal grads). Per-architecture patchers (Qwen/Gemma/OLMo/DeepSeek).
3. Readout filtering: drop non-alphanumeric tokens.
Result is still ONE linear map per layer (drop-in for J-lens). Fitted on Qwen3.6-27B.

Findings / limitations / critiques:
- J-lens reads unspoken intermediate concepts on Claude 4.5 family; mid-layer band;
  causally potent under swaps; used in alignment audits. Beats logit lens early, beats
  tuned lens at exposing intermediates.
- J++ recall@10 of intermediates on Qwen3.6-27B: 55.2% vs 35.7% (J-lens) / 37.7% (R-lens);
  intermediates surface earlier. Removing Jacobian Filtering costs 11.7 of 19.5pp (UNVERIFIED snippet).
- Stated limits: single-token vocab; averaged first-order linearisation blurs context;
  reads only "verbalizable" content; early layers noisy; k arbitrary.
- Critique — Ostrow, Qian, Liu, Fiete, "A Mathematical Theory of the Jacobian Lens"
  (Oct 2026): lens selects high amplification x cross-position coherence x cross-context
  coherence; MEAN Jacobian need not resemble any individual one; prominence unrelated to
  variance explained; toy-model validation only. NOTE: a later agent flagged that Ostrow
  et al. does NOT mention J++ or fitted expert weights — it critiques the base J-lens.
- Critique — Yan et al., "Short Horizons and Sparse Concepts" (arXiv Aug 2026): Jacobian
  energy is sparse and short-horizon.
- The "expert weights are fitted with supervision" claim is from the code/snippets;
  the LessWrong post itself was not opened → treat as needing re-read.

Applied to a biological model? NONE FOUND. Closest: GDTR (Cho et al., bioRxiv Jul 2026),
a training-free residual-stream "settling depth" lens on Evo 2 7B (logit-lens-style, not Jacobian).

## 1.2 BSF = Block-Sparse Featurizer
- Name: "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds",
  arXiv 2606.25234, 2026-06-23. First authors Fel, Kowal, Jacobs, Hazra, Bhalla;
  seniors incl. McGrath, Merullo, Lubana, Geiger (Goodfire/Harvard/Stanford/Brown).
  Blog: goodfire.ai/research/bsf-vision. Code: github.com/goodfire-ai/block-sparse-featurizer (MIT).
- Idea: each activation = sparse sum of points on low-dim manifolds; prior is sparse
  ACROSS blocks, dense WITHIN a block. Dictionary = n_groups blocks of group_size dirs
  (default 3), linear decoder. Each block → a norm (presence) + a coordinate z_g.
- Three variants (from code):
  - Vanilla: free encoder; block-TopK keeps l0 largest-norm blocks; MSE; unit-normed
    decoder. group_size=1 → absolute-TopK SAE.
  - Grassmannian: orthonormal frame per block (QR each fwd); tied encoder z_g = γ_g x D_g^T;
    block-TopK; MSE.
  - Group Lasso: free encoder; block-JumpReLU gate H(||a_g|| - θ_g), θ via STE; MSE + λ L0,
    λ by dual ascent to target L0.
- Training: Adam lr 4e-4, batch 2048, OneCycle, denoising. Preprocess: subtract per-position
  mean, scale E||x||^2 = d. Eval: MDL vs direction SAEs; recovered concepts typically 2–4 dim.
- Models shown: InceptionV1, DINOv3 ViT-B/16, SDXL. Needs only cached activations; no
  gradients through base model; no architectural assumption beyond per-token vectors.
  Cost = SAE of width n_groups x group_size.
- NOT published on language models by Goodfire. Robotics follow-up: arXiv 2610.20767 (UNVERIFIED).
- Critique — Jerpelea & Ananthram, "A Deeper Analysis of Block-Sparse Featurizers",
  arXiv 2608.27515: manifold splitting across blocks, seed-dependent, not fixed by capacity;
  raising k degrades recovery; mixed block sizes merge/misassign; naive block crosscoder
  collides features. Fix: Tournament Top-K (0.77→0.90 synthetic manifold recovery) +
  dedicated-column crosscoders. BSF still beats SAE+post-hoc grouping (R² 0.77 vs <=0.65).
- Related motivation: Bhalla et al., "Do Sparse Autoencoders Capture Concept Manifolds?"
  (arXiv Apr 2026). Within-block basis identified only up to rotation.

Applied to a biological model? NONE FOUND for BSF. Goodfire's own bio work uses ordinary
BatchTopK SAEs on Evo 2 (layer 26, with Arc). One third-party issue (speediedan/interpretune
#708) proposes BSFs on LM residuals / in J-space — title only.

## 1.3 Combination assessment (from methods agent)
- Neither method has a bio application; BSF has no published LM application; no published
  combination of the two except one third-party issue title.
- BSF ports easily (cached activations only). J++ does NOT port cleanly: needs causal
  decoder + rich token vocab + labelled intermediates. Evo 2 has 4 tokens; ESM3 is
  bidirectional multi-head; AlphaGenome has no unembedding.
- What transfers: J_l as an averaged linear transport map; readout must be redesigned
  (structure tokens / tracks / a BSF basis).
- Cleanest contribution = BSF on bio models with physical ground truth; Jacobian-transport
  is the riskier exploratory half. Scoop risk: Goodfire owns BSF + works on Evo 2 with Arc.

================================================================================
# PART 2 — MODEL AVAILABILITY (for interpretability use)
================================================================================
- ESM-2 (all sizes incl 3B/15B): open, permissive (MIT-style). Workhorse of the field.
- ESM3 (esm3-sm-open-v1, 1.4B, 48 layers): open weights, NON-COMMERCIAL (EvolutionaryScale
  Cambrian NC license); larger sizes API/Forge only. License specifics lightly UNVERIFIED.
- ESM C (300M/600M; 6B via API): open weights, Cambrian licensing; non-commercial caveat.
  300M-vs-600M license split UNVERIFIED.
- Evo 2 (1B/7B/40B): open weights, Apache-2.0 on Arc model cards (commercial OK).
  NVIDIA NIM container license differs. Public Goodfire BatchTopK SAE baseline exists.
- Nucleotide Transformer (50M–2.5B, incl v2): open weights; exact license UNVERIFIED.
- DNABERT-2, HyenaDNA, Caduceus, Enformer, Borzoi: open, usable. Enformer/Borzoi are the
  untouched regression targets (no SAE/circuit work found).
- AlphaGenome v1 (~450M): weights public but GATED + NON-COMMERCIAL (DeepMind AlphaGenome
  Model Terms); code Apache-2.0; released ~Jan 2026. Usable for interp under NC terms.
- AlphaGenome 2 (released 2026-11-11): API-ONLY, no weights; cell-state-conditioned,
  ~107M cell profiles; weights "to accompany a forthcoming paper" (UNVERIFIED, news only).
  NOT usable for mechanistic interp today.
- scGPT / Geneformer / scFoundation / Cell2Sentence: open, used throughout single-cell interp.
- RFdiffusion / RoseTTAFold3 / Boltz-1 / Boltz-2 / OpenFold / ESMFold: open, actively used.
  AlphaFold2/3 weights under their own gated NC terms (AF3 appears in Simon 2026 feature-death study).

## 2.1 "Bonbon" (the user's requested 4th model)
- Real model. Paper: "Emergent Biological Capabilities in a Foundation Model for Molecular
  Interactions", arXiv 2611.14504 (~10–12 Nov 2026), single author Adi Mashiach; affiliation
  surfaced only as "Output Labs" (UNVERIFIED). Reported NeurIPS 2026 Main Track Oral (UNVERIFIED).
- 1.6B params (ladder 42M–1.6B), ~3.2T tokens; decoder transformer + RoPE, pre-LN;
  self-supervised on paired protein–ligand "sequences". NO 3D structure, NO pair rep,
  NO diffusion head — pitched as an alternative to co-folding. Ligand tokenisation
  (SMILES/SELFIES/atoms) and objective (masked vs AR) UNVERIFIED — absent from reachable sources.
- Reported zero-shot capabilities: mechanism of action, binding-site (ortho/allosteric),
  bond type (covalent/non-covalent); wet-lab dose-response hit rate 61.5% (arXiv) / 70%
  (NeurIPS listing) over nine targets — figures conflict, UNVERIFIED.
- NO public weights/code/model card/API found; data+weights reported proprietary.
- NO interpretability work exists on it.
- VERDICT: sequence-only + closed-weight → cannot be hooked, and the ligand-geometry
  manifolds the thesis wants are not architecturally present. Recommend DROP. Open
  substitutes for a protein–ligand leg: Boltz-2 (already has PairSAE interp, arXiv 2606.27440),
  Chai / RoseTTAFold-All-Atom, or seq+SMILES binding-site models (e.g. "Pou25b"
  Pourmirzaei et al. — UNVERIFIED). Agent did not pick one.
- All Bonbon specifics are from search snippets/secondary indexes; arXiv PDF not opened.

================================================================================
# PART 3 — LITERATURE MAP (from the two completed 15-Nov deep searches)
================================================================================
Workspace: app.undermind.ai/projects/33899782-4f3e-48f1-9e4a-a3b6ac8a7af7
Every paper below was verified by the survey agent against a primary source (abstract or
full text via Undermind) or the model card. biorxiv.org / huggingface.co were unreachable;
bioRxiv items read via Undermind PDF corpus; some license details from search snippets.

## 3.1 Protein LMs — SAEs / dictionary learning
- InterPLM — Simon & Zou (Stanford). bioRxiv 2024-11 → Nature Methods 2025. ESM-2 8M–3B.
  L1 SAEs. ~2,548 interpretable latents/layer → ~143 concepts. DOI 10.1038/s41592-025-02836-7.
- InterProt — Adams, Bai, … AlQuraishi (Columbia). bioRxiv 2025-02 / ICML 2025. ESM-2 650M.
  TopK SAEs. Feature taxonomy point/motif/domain. 10.1101/2025.02.06.636901.
- Gujral … Berger (MIT). PNAS 2025-08. ESM-2 650M. TopK SAEs + transcoders → GO/families.
  10.1073/pnas.2506316122.
- Garcia & Ansuini (AREA). arXiv 2025-02. ESM-2. SAE + steering (zinc-finger gen). arXiv 2502.09135.
- Reticular (Parsan, Yang, Yang). arXiv 2025-03. ESM2-3B (ESMFold base), layers 8 & 36.
  MATRYOSHKA SAEs; steers ESMFold solvent accessibility. arXiv 2503.08764.
- Silberg, Simon, Zou (Stanford). bioRxiv 2025-10. ESM-2/InterPLM SAEs. Finds 491 missing
  CATH annotations; 8,077 metagenomic matches. 10.1101/2025.10.02.680154.
- MotifAE — Hou, Liu, Shen (Columbia). Nat Commun 2026-09. PLM dictionary. Unsupervised motifs.
- ProtSAE — Liu … Hu. arXiv 2025-08. Semantically-guided SAE.
- Tsui, Talreja, Aghazadeh (GT). arXiv 2025-08. ESM-2 650M. TopK SAE for low-N fitness/design. arXiv 2508.18567.
- Setlur, Mihajlovic, D. Lee. arXiv 2026-08. ESM-2 8M, all layers. SAE + Cα-geometry,
  FDR-controlled; ablation shifts contact maps. arXiv 2608.26419. [reviewer's "prior work" pick]
- Hu … Liu. preprint 2026-06. ESM C 6B, 16,384-dim SAE + GPT-5 auto-annotation; EC-number
  78.9% top-1 w/o task training. arXiv 2606.12209.
- Kulikova … Safavi-Hemami (Utah). bioRxiv 2026-08. InterPLM SAEs → neuropeptide precursors.
  10.64898/2026.08.20.746077.
- Wang … Ren. bioRxiv 2026-05. ESM-3 + ESM-2. SAE + differentiable steering → +VEP on 114 DMS.
  10.64898/2026.05.12.724472.
- Murakami & Matsunaga. Bioinformatics 2025-12. Fine-tuned nanobody PLM interp (thermostability).
- AuditPPI — Zhu … Pan. bioRxiv 2026-10. SAE pair-fingerprints; PPI predictors exploit
  colocalization shortcuts. 10.64898/2026.09.27.754758.
- Banerjee … Tam. arXiv 2025-07. Auto-labels EVERY neuron + steering (MW, α-helix, zinc). arXiv 2507.06458.
- Huang … Yao (Tencent). arXiv 2025-07. Activation steering (no SAE), lysozyme. arXiv 2509.07983.

## 3.2 Protein structure / folding / co-folding / generative
- Lu, Brinkmann … Bau, Wendler (Northeastern). arXiv 2026-02. ESMFold/OpenFold/Boltz-1 trunks.
  Causal interventions; shared 2-stage folding; steer charge/distance. arXiv 2602.06020.
- Liu … Zhang. arXiv 2026-10. ESMFold trunk (single+pair, recycles). SAEs + grad attribution
  + LLM auto-interp; single rep=semantics, pair rep=geometry. arXiv 2610.36226.
- PairSAE — Migliorini … Viessmann. arXiv 2026-06. Boltz-2 pairformer. N-mode SVD + SAE on
  pair tensors; features align UniProt, predict affinity. arXiv 2606.27440. [NO causal/seed/leakage controls]
- Jedryszek … Crook (Oxford). arXiv 2026-08. Boltz-1 trunk+diffusion. Probes+SAEs+interventions;
  LINEAR DECODABILITY != CAUSAL INFLUENCE (β-strand F1 0.82, no steering). arXiv 2608.11475.
- Jedryszek & Crook. arXiv 2026-08. Boltz-1. Cross-seed stability + intruder detection;
  stability selects structure over function features. arXiv 2608.27754.
- FoldSAE — Zarzecki … Deja. arXiv 2025-11. RFdiffusion. TopK SAE; SS features, steering. arXiv 2511.22519.
- VFUSE — Yu & Olson. bioRxiv 2026-06. RoseTTAFold3/RFDiffusion3 (all-atom diffusion). First
  SAE on all-atom diffusion; hazard features AUROC 0.84; biosecurity angle. 10.64898/2026.06.08.730928.
- Steenwyk. bioRxiv 2026-07. ESM3 1.4B, 48 layers. RSA per modality; modalities fuse at
  layers 25–35; function never fuses; absent in random-init. 10.64898/2026.07.08.737319.
- Steenwyk. bioRxiv 2026-05. ESM-2 650M + ESM3 1.4B. SAEs + attribution patching + steering;
  78% feature convergence; geometric head L0H7 is a structure bottleneck. 10.64898/2026.05.12.724593.

## 3.3 Protein circuit / causal / Jacobian / geometry
- ProtoMech — Tsui … Aghazadeh (GT). arXiv 2026-02. ESM-2. Cross-layer transcoders;
  82–89% perf, <1% latent; steering beats baselines 70%. arXiv 2602.12026.
- ProGenMech — Tsui … Aghazadeh. arXiv 2026-06. ProGen3 (MoE generative). CLT circuit tracing. arXiv 2606.16044.
- Nainani … Green (UMass/NEU). bioRxiv 2025-08. ESM-2. ACTIVATION PATCHING in SAE latent space;
  motif detectors gate domain detectors. 10.1101/2025.08.22.671739.
- Mohanty, Phutela, Green. bioRxiv 2026-09. ESM-2 650M. SAE + IG + patching; causal zero-shot VEP;
  40–75% circuit reuse. 10.64898/2026.08.28.747907.
- Zhang … Ovchinnikov (Harvard). PNAS 2024-01. ESM-2. CATEGORICAL JACOBIAN for coevolution/contacts.
- Thorstenson. arXiv 2026-06. ESM-2. Attention-head contact readout; single forward pass;
  beats categorical Jacobian on 650M; quantifies pretraining-overlap inflation. arXiv 2606.21876.
- Zhang, Akiyama … Ovchinnikov. bioRxiv 2025-11. ESM-2 up to 15B. Homo-oligomer interface contacts.
  10.1101/2025.11.16.688745.
- Vinod … K. Yang (Microsoft). PLOS Comp Biol 2025-12. 39 trainable subnetworks; structure
  factorization in weights. 10.1371/journal.pcbi.1013925.
- Wang … Wei. bioRxiv 2026-04. ESM-2. Latent-manifold geometry; conflates IDP/fold-switch/knotted.
  10.64898/2026.04.07.717117.
- Whitfield … Fournier (Mila). bioRxiv 2026-07. ESM2 & AMPLIFY (8 models), layerwise probing;
  concepts emerge by depth; data/compute > scale. 10.64898/2026.07.20.739599.
- Bigot … Shakhnovich (Harvard). arXiv 2026-06. ESM families; "nativeness axis". arXiv 2606.12609.
- Candido … Rives (EvolutionaryScale). bioRxiv 2026-06. Concept-space "world model of protein
  biology"; 6.8B-sequence map. 10.64898/2026.06.03.729735.

## 3.4 Genomic / DNA / RNA
- Evo 2 — Brixi … Hie (Arc + Goodfire). Nature 2026-03 / bioRxiv 2025-02. BatchTopK SAE;
  exon-intron, TF binding, prophage features. LAYER = 6 per Orlov review Table 3, but
  secondary reports say layer 26 / 32,768 features on HF — UNVERIFIED, resolve from primary.
  10.1038/s41586-026-10176-5.
- Evo 2 as classification machine. bioRxiv 2026-09. 10.64898/2026.09.30.755330.
- Decode-gLM — Maiwald, Jedryszek … Crook (Oxford). bioRxiv 2026-02. Nucleotide Transformer
  50M/500M. BatchTopK SAEs + Meta-SAEs; CMV enhancer feature → DB-contamination audit;
  steered A1408G resistance; random-init control. 10.1101/2025.10.31.685860.
- Ali. arXiv 2026-07. NT + DNABERT-2. TopK SAE + causal ablation; NAIVE PWM VALIDATION
  CONFOUNDED by GC/repeats → spurious TF features; composition-matched protocol cleans. arXiv 2607.19618.
- Kmiec, O'Brien, McCoy. bioRxiv 2026-05. NT v2 500M, all 24 layers. SAEs + knowledge graphs;
  weak distributed regulatory logic. 10.64898/2026.05.13.725007.
- Guan, He, Zhang. 2025-07. HyenaDNA-small-32k. L1 SAE; TF motifs. arXiv 2507.07486.
- SAE-RNA — Kim & Nam. arXiv 2025-10. RNA LM. arXiv 2510.02734.
- SPIRAL — Hossain … Rahman (BUET). bioRxiv 2026-08. BiRNA-BERT L0/5/11; tokenizer-aware SAE;
  44.3% layer-5 features → bpRNA SS. 10.64898/2026.08.11.744228.
- Xu et al. bioRxiv 2026-05. Disentangle RNA evolution vs thermodynamics. 10.64898/2026.05.28.728275.
- Tomaz da Silva … Gagneur (TUM). bioRxiv 2025-07. DNA LMs. NUCLEOTIDE-DEPENDENCY (Jacobian-style)
  maps; found 4 validated E. coli RNA structures. 10.1101/2024.07.27.605418.
- Kendiukhov. Comp Biol Chem 2026-10. Plant-DnaGemma/DNABERT/AgroNT. SAEs + ablation;
  "random background too weak a control"; GC confounds. 10.1016/j.compbiolchem.2026.109438.
- Con et al. bioRxiv 2025-06. Genomic transformer attention framework. 10.1101/2025.06.26.661544.
- Genome-Factory — Wu et al. 2025. Unified SAE/interp tooling (HyenaDNA/DNABERT-2/Caduceus/NT/Evo/GenomeOcean).
- Mathur & Sachidanandam (Mt Sinai). bioRxiv 2026-11. Evo2/DNABERT-2 VEP blind spots;
  flanking flips pathogenic tRNA calls. 10.64898/2026.03.10.710786.
- "Mechanistic Invariance Test". arXiv 2026. gLMs fail positional regulatory logic (negative). arXiv 2604.06549.
- Position paper: "Genomic Model Research Must Move Beyond Anecdotal Evaluation of
  Interpretability". arXiv 2026-06. arXiv 2606.07607.

## 3.5 Single-cell foundation models
- Pedrocchi … Boeva (ETH). bioRxiv 2026-06. scGPT/scFoundation/Geneformer. BatchTopK SAEs;
  suppress batch features → better integration; drug-feature steering. 10.1101/2025.10.22.681631.
- Kendiukhov. arXiv 2026-03. Geneformer V2-316M / scGPT. TopK SAEs all layers; 82k+24k atlases;
  only 3/48 TFs show regulatory-target-specific response vs CRISPRi. arXiv 2603.02952.
- Hosokawa … Takeda (Tokyo). arXiv 2025-09. Cell2Sentence. Transcoder circuits → pathways. arXiv 2509.14723.
- Claye … Duquesne. arXiv 2025-10. 2 scRNA-seq FMs. TopK SAE + counterfactual attribution. arXiv 2510.25807.
- Kendiukhov. Bioinformatics 2026-03. Geneformer/scGPT. SAE-feature ablation circuits; weak CRISPRi. 10.1093/bioinformatics/btag379.
- Kendiukhov. arXiv 2026-02. scGPT/Geneformer. Trivial gene baselines beat attention edges;
  ablating "regulatory heads" → no degradation (negative). arXiv 2602.17532.
- Kendiukhov. arXiv 2026-03. Exhaustive circuit mapping; redundancy, hub architecture. arXiv 2603.11940.
- Kendiukhov. arXiv 2026-03. scGPT hematopoietic manifold extraction. arXiv 2603.10261.
- Kendiukhov. PLOS ONE 2026. 141 geometric-structure hypotheses on bio FMs. 10.1371/journal.pone.0344826.
- Schuster. MLCB 2024-10. SAE on scVI-style latents.

## 3.6 Cross-cutting evaluation / negative-result / method papers
- Shahsavand … Mahdavi Sabet. bioRxiv 2026-10. Ground-truth SAE eval on spatial/single-cell/LM;
  reproducibility != meaningfulness; consensus-over-seeds SAE. 10.64898/2026.10.27.762296.
- Korznikov … Tutubalina. arXiv 2026-02. SAEs recover 9% of ground-truth at 71% EV; RANDOM
  BASELINES MATCH TRAINED SAEs. arXiv 2602.14111.
- Heap … Aitchison. 2025-01. Automated interp metrics don't distinguish trained vs random transformers.
- Paulo & Belrose (EleutherAI). arXiv 2025-01. SAEs on same data learn different features (seed instability).
- Simon, Adams, Zou. arXiv 2026-05. Activation outliers + feature death; >70% death on AlphaFold3;
  mean-centering fixes it. arXiv 2605.31518.
- Chanin. arXiv 2026-05. Are SAE benchmarks reliable? (critiques SAEBench).
- Ghosh … Agarwal. 2026-10. Diminishing returns of data in SAEs.
- Orlov … Nikitin. bioRxiv 2026-03. REVIEW "What Do Biological Foundation Models Compute?";
  3-level framework; catalogs circularity/null-model gaps. 10.64898/2026.03.04.709491.

================================================================================
# PART 4 — OPEN GAPS (survey agent, with confidence)
================================================================================
1. AlphaGenome / Borzoi / Enformer have ~no SAE/transcoder/circuit interp — only ISM/attention. HIGH.
2. No crosscoder / model-diffing on any bio model (Orlov review states this). HIGH.
3. Circular validation: protein-SAE features scored vs Swiss-Prot/InterPro/GO the models
   trained on; review says no study controls for it. Experimental-only validation (DMS/MPRA/
   CRISPRi) is the clearest opening. HIGH.
4. Random-init / composition-matched baselines rarely run; the few that do change conclusions. HIGH.
5. Causal (not correlational) validation is thin; existing causal protein work is 1–2-protein
   case studies; genomic/single-cell repeatedly find organized knowledge but little causal
   regulatory logic. HIGH.
6. "Linear decodability != causal steerability" shown once (Boltz-1); otherwise unexamined. MED-HIGH.
7. Feature reproducibility/stability barely measured for bio SAEs. MED-HIGH.
8. No standardized interp benchmark for bio models (a bio-SAEBench). HIGH.
9. ESM3 multimodal generative steering via dictionary learning underexplored. MED.
10. Feature death quantified (AF3) but downstream effect on discovery unquantified. MED.
11. RNA models very thin; no circuit/causal work. MED-HIGH.
12. Genuine novel-biology discovery (not re-finding annotations) is rare; standout examples:
    Decode-gLM, Silberg (491 CATH), Tomaz da Silva (4 RNA structures). HIGH.

================================================================================
# PART 5 — EVALUATION / POSITIONING (area-chair agent; dual-use section was cut by a safeguard)
================================================================================
Bottom line: four model families + two methods ≈ 15–20% main-track. Narrow to two models
with pre-registered ground-truth variables + causal dose-response → ~35–45%.

Corrections to the brief (from primary sources):
- Ostrow et al. does NOT critique J++ (critiques base J-lens; toy models only).
- "J++ expert weights fitted with supervision" UNVERIFIED (post not opened); re-read before related-work.
- PairSAE reports no causal interventions, no random/seed controls, no leakage checks → open lane.
- BSF primary PDF unreadable via Undermind; BSF details from the critique + snippets.

H1–H5 falsifiable hypotheses:
- H1 Existence: pre-registered continuous vars (φ/ψ, helix phase, rel-SASA, codon phase,
  exon position, pocket distance/angle) lie on low-dim manifolds of predicted topology in
  TRAINED models. Refute if trained−random-init circular-R² gap < 0.1 or topology no better than null.
- H2 Fragmentation: 1-D SAE latents tile these manifolds. Refute if <=d SAE latents reach
  restricted R² within 0.05 of supervised rank-d ceiling.
- H3 Recovery: BSF (Tournament Top-K) recovers each var in fewer units than alternatives at
  matched reconstruction AND matched bits. Refute if SAE+best post-hoc grouping within seed
  noise, or BSF advantage gone on MDL/Pareto frontier.
- H4 Causal manifold: moving along within-block coordinate changes outputs in predicted
  direction+amount. Refute if slope indistinguishable from random-direction control, or only
  magnitude matters.
- H5 Jacobian transport ranks blocks by output relevance. Refute if J-score doesn't beat
  activation-norm ranking, or mean-vs-per-input Jacobian alignment low at the layers used.
- If BSF does NOT beat SAEs: still publishable as a ground-truth benchmark paper (Datasets &
  Benchmarks / TMLR), carried by the released benchmark. Not a main-track oral.

Evaluation protocol highlights:
- Ground truth computed by deterministic code from experimental structures/annotation ABSENT
  from training — not curated labels, not model's own predictions.
- Supervised ceiling (rank-d linear + MLP probe) and floor FIRST per var/layer/model; drop low-ceiling sites.
- Recovery metrics: blocks/latents to reach R²>0.8; restricted R² at n=d + support size;
  held-out circular-circular correlation / toroidal R² after alignment; fraction of ceiling;
  block monosemanticity (MI gap to nuisance vars).
- Topology: persistent homology on within-block codes; expect Betti (1,1,0) circle, (1,2,1)
  torus; but φ/ψ is non-uniform (Ramachandran basins) → pre-register "torus vs basin clusters".
- Fair d-dim block vs d latents: report ALL matchings (matched total dim; matched params;
  matched active scalars; matched active units) + full NMSE/downstream-loss Pareto + MDL with
  identical code. Post-hoc grouping baselines get equal tuning budget.
- Rotation non-identifiability: evaluate invariants only (principal angles, CCA to (cos,sin),
  Procrustes-aligned R² with train/test split, topology); never interpret raw axes; Hungarian
  block-matching across seeds vs random-subspace baseline; report permuted-label version.
- Synthetic sanity: plant a known torus/circle into real activations at realistic norm first.

Controls — MANDATORY all modalities: (1) random-init same-arch model through full pipeline;
(2) token-shuffled + random-featurizer; (3) autocorrelation-preserving permuted-label nulls
(circular shifts, not i.i.d.); (4) PCA/ICA/varimax + supervised rank-d probe; (5) >=5 seeds +
stability statistic + plain vs Tournament Top-K; (6) dead/dense rates; (7) local-sequence
window baseline (helix/codon phase may be near-trivial from local seq); (8) sequence-cluster splits.
- Proteins (ESM3): post-cutoff structures, <=30% identity; experimental structures only;
  sequence-only vs structure-conditioned reported separately (structure input makes dihedral
  decoding trivial); AA- and SS-conditioned analysis.
- Genomics: dinucleotide-shuffled + GC-matched; k-mer baseline; held-out chromosomes (respect
  AlphaGenome folds); reverse-complement consistency; frame-shifted codon controls; species held out (Evo 2);
  resolution-matched ground truth + conv receptive-field positional control (AlphaGenome).
- Co-folding: time + scaffold(Tanimoto) split; PLINDER leakage categories; pocket-similarity
  control; apo vs holo; MSA vs no-MSA; fixed recycling/diffusion seeds as noise floor;
  hook single vs pair reps separately if Pairformer+diffusion.

Causal validation: manifold-traversal (encode→rotate/translate within-block code→decode→add
to residual→forward); dose-response over >=8 targets incl wrap-around for circular vars (the
discriminating test); matched-norm controls (random in-subspace, random orthogonal, equal-freq
block, top-d-PCA, supervised-probe direction); off-manifold checks (norm, pseudo-perplexity,
pLDDT/pTM); counterfactual-input comparison; pre-specified effect thresholds; paired tests over
clustered proteins/loci. "Output relevance" for J-score only if: Ostrow diagnostics per layer
+ H5 holds + stable across subsets/seeds + cluster-conditional vs global mean compared on
disjoint split. Else call it "first-order output sensitivity (averaged)", layer-restricted.
Naming: use "layer-to-output Jacobian transport"; add a table vs categorical Jacobian (Zhang
PNAS 2024) + nucleotide-dependency maps (Tomaz da Silva); cite Thorstenson arXiv 2606.21876.

Positioning — one-sentence contribution: "Using variables with exact geometric ground truth,
we show bio FMs encode periodic/continuous structure as low-dim manifolds that scalar SAE
latents fragment, test whether block-sparse featurizers recover them under random-init/leakage/
seed controls, and validate recovered manifolds by dose-response interventions on outputs."
Framings ranked: (1) ground-truth benchmark for featurizers [best, robust to negative BSF];
(2) controls-first rigour paper; (3) causal manifold steering [high risk]; (4) "first BSF/J-lens
on bio" [weakest, incremental].
Venues: ICML 2027 main — full paper reported 22 Jan 2027, abstract ~14–16 Jan (UNVERIFIED, icml.cc
not read); conf 31 Jul–6 Aug 2027 Rio. ICLR 2027 main CLOSED. ICLR 2027 workshops ~1 Feb 2027
(UNVERIFIED; accepted-workshop list due 29 Nov 2026). NeurIPS 2027 ~14–21 May 2027 (unconfirmed).
Recommendation: ICLR 2027 workshop paper on ONE model (~1 Feb) as a stake, then ICML main if
H3+H4 land by early Jan, else NeurIPS 2027.
Scope: four families too broad for 9 pages. KEEP ESM3-open; keep ONE of AlphaGenome (more novel,
no SAE baseline → train own) or Evo 2 (cheaper, Goodfire SAE exists); CUT Bonbon first; cut the
other genomic model to appendix.

Scoop & ethics: NO direct scoop found (through 19 Nov 2026). Closest:
- Cerebral Valley hackathon entry reportedly adapted J-lens to ESM1b masked encoder (one snippet,
  UNVERIFIED, could not re-find). Check manually.
- Setlur et al. arXiv 2608.26419 — geometric annotations for scalar SAE features on ESM-2;
  no circular/manifold/random-init → the reviewer's "prior work" pick; MUST cite.
- Goodfire "phylogeny manifold" (Aug 2025) frames Evo 2 reps as a manifold; Goodfire likeliest
  to run BSF on Evo 2; their Oct–Nov 2026 posts not readable (UNVERIFIED whether BSF-on-bio).
- Biohub ESMC world-model (bioRxiv Jun 2026) — SAE on ESMC-6B w/ random-init control.
- Also cite: Steenwyk x2; Liu ESMFold trunk SAE; Shahsavand (ground-truth SAE eval, overlap
  with framing 1); SASA (arXiv 2606.06333); Bhalla et al. (arXiv 2604.28119); Cho GDTR.
- Dual-use: section was cut by the model safeguard; not reconstructed here. General principle
  retained: restrict interventions to benign structural variables; note genomic/ligand steering
  is capability-adjacent. Full treatment deferred to a human/compliant-model pass.
