# Interpretability of deep-learning / foundation models in drug discovery (A) and protein language models (B)

A current map of recent interpretability work (2024 – Nov 2026, weighted toward 2026).
Compiled 2026-11-19 from the Undermind workspace `33899782-4f3e-48f1-9e4a-a3b6ac8a7af7`.

## Sourcing and method

- **Primary sources**: two Undermind deep searches plus iterative `search_papers`.
  - (A) `/Interpretability of deep learning models in drug discovery` — completed 2026-11-19, 163 ranked papers. Workspace path: `?path=/Interpretability%20of%20deep%20learning%20models%20in%20drug%20discovery`
  - (B) `/Protein model mechanistic interpretability SAE` — completed 2026-11-15, 72 ranked papers. Workspace path: `?path=/Protein%20model%20mechanistic%20interpretability%20SAE`
- Gap-filling `search_papers` runs covered: chemical-LM SAEs; co-folding/docking shortcut & leakage studies; molecular GNN explainability; molecular representation geometry; molecular generative/diffusion interpretability; antibody/peptide-design interpretability; and a recency pass (Oct–Nov 2026) for PLM work that post-dates the 15 Nov search.
- **No additional deep search was launched** — the two existing searches plus targeted semantic searches gave full coverage of every requested sub-area, so the shared one-search quota was left for the other agent.
- **Claims below are grounded in abstracts / deep-search summaries read from Undermind.** Items I did not read a PDF for are reported at abstract level; nothing is filled in from memory. A few flagged points are marked UNVERIFIED. All DOIs/links were fetched with `get_paper_info(show_doi=true)`.
- Scope limiter: this is a description of the published interpretability literature. It deliberately does not give guidance on steering models toward hazardous biological or chemical outputs.

## Headline reading of the field

1. **Two parallel maturity levels.** Protein-model interpretability has moved from *feature cataloguing* (SAEs on ESM-2) to *causal, cross-layer circuit tracing* and *folding-trunk mechanism*. Chemical-language-model (cLM) interpretability is roughly where PLMs were in 2024 — mostly representational (SAEs, probing, attention) with causal/circuit work just beginning (late-2025/2026).
2. **"Decodable ≠ causal" is now the central methodological theme** on both sides. Multiple 2026 papers show that probe accuracy or feature–label correlation does not establish causal use, and that untrained/random-weight controls can erase apparent "learned chemistry/biology" (GFlowNets, molecular models, SAE auto-interp metrics).
3. **Physics-vs-shortcut is the dominant critical thread in drug discovery.** Co-folding/affinity models (AlphaFold3, Boltz-1/2, RoseTTAFold-All-Atom, Chai, DiffDock) are repeatedly shown to memorize training poses, be insensitive to binding-site mutations, and lean on MSA/evolutionary "shortcuts" rather than interaction physics.
4. **The newest internal work targets the Pairformer/trunk–diffusion boundary** of AlphaFold3-class models (PairSAE, trunk partitioning, trunk-diffusion probing, MSA-shortcut circuits) — this is the most active 2026 frontier and the most directly relevant to a co-folding thesis.

---

# PART A — DRUG DISCOVERY

## A1. Chemical / molecular language models — SAEs, circuits, probing (the core of A)

Central, most recent:

- **[Ken26l] What Does a Chemical Language Model Know About Molecules?** — Kenneth, Adams, Bai, van Westen (Leiden / AlQuraishi-adjacent). arXiv, 2026-06-22. Model: **MolFormer** (encoder-only cLM). Method: **sparse autoencoders** across layers + InterMol visualizer. Finding: early layers use **position-tracking latents** to parse SMILES grammar; later layers encode **atom-in-substructure and pharmacologically relevant** features; non-canonical SMILES disrupt representations more than invalid SMILES via position-latent propagation. https://doi.org/10.48550/arXiv.2606.23443
- **[Var25] Circuits, Features, and Heuristics in Molecular Transformers** — Váradi, Marosi, Antal (Budapest). arXiv, 2025-12-10. Model: autoregressive SMILES transformers (drug-like molecules). Method: **mechanistic/circuit analysis + SAE feature dictionaries**. Finding: identifies low-level **syntactic parsing** circuits and more abstract **chemical-validity** constraints; SAE features transfer to downstream predictive gains. First real circuit-level cLM study. https://doi.org/10.48550/arXiv.2512.09757
- **[Coh25] Unveiling Latent Knowledge in Chemistry Language Models through Sparse Autoencoders** — Cohen, Hasson, Tanovic. arXiv, 2025-12-08. Model: **FM4M SMI-TED** chemistry foundation model. Method: SAEs on CLM activations. Finding: latent features map onto **structural motifs, physicochemical properties, and pharmacological drug classes**; generalisable SAE framework for CLMs. https://doi.org/10.48550/arXiv.2512.08077
- **[Li26h] From Syntax to Semantics: Emergence of Chirality in SMILES Translation Models** — Z. Li, Yoshikai, Nemoto, Kusuhara, Mizuno (Tokyo). arXiv, 2026-05-11. Model: Pan-CORE autoregressive encoder-decoder SMILES models. Method: high-time-resolution checkpoint analysis of **attention dynamics, residual-stream trajectories, latent geometry; attention-head ablation**. Finding: chirality learning shows an abrupt **"grokking"-style jump** after a plateau; a small set of **chiral-sensitive attention heads** is causally necessary; encoder-centered reorganization. https://doi.org/10.48550/arXiv.2605.09949
- **[Bal25c] A Large Scale Investigation of Scaling Limits in Chemical Language Models (NovoMolGen)** — Balaji, Chitsaz, Fournier, Bhatt, Chandar (Mila). 2025-08-19. Model: 30k+ CLMs, 0.5M–1B params, SMILES/SELFIES/SAFE. Method: **layer-wise probing + SAE analysis** under compute-controlled scaling. Finding: **chemical syntax saturates early, semantics emerge slowly**; representation gains **decouple from downstream goal-directed generation**. https://arxiv.org/abs/2508.13408
- **[Ela26] Molecules Meet Language: Confound-Aware Representation Learning and Chemical Property Steering in Transformer-VAE Latent Spaces** — Elabid, Andrzejewski, Brzoza, Cangi. arXiv, 2026-05-07. Model: autoregressive Transformer-VAE on SELFIES. Method: frozen-model linear probes as steering directions + **confound-aware residualization** (SELFIES length/branch/ring/entropy confounds). Finding: robust monotonic steering for cLogP, TPSA, etc. **only after** controlling for representation-level confounds — apparent property geometry partly reflects SELFIES artifacts. https://doi.org/10.48550/arXiv.2605.06303
- **[Zha26s] SLIM: Sparse Latent Steering for Interpretable and Property-Directed LLM-Based Molecular Editing** — M-X. Zhang, …, Y. Sun. arXiv, 2026-05-11. Method: sparse latent steering for property-directed molecular editing in LLM latent space. https://doi.org/10.48550/arXiv.2605.10831
- **[Li26k] Controllable Molecule Generation via Sparse Representation Editing** — Z-R. Li, …, W-Y. Lin. 2026-11-06. Interpretability-driven sparse representation editing for controllable generation. https://arxiv.org/abs/2611.09235
- **[Zha25f] AtomDisc: Atom-level Tokenizer that Boosts Molecular LLMs and Reveals Structure-Property Associations** — M-X. Zhang, Shen, Sun. arXiv, 2025-11-28. Atom-level discrete tokenizer; surfaces structure–property associations. https://doi.org/10.48550/arXiv.2512.03080
- **[Lee25b] Task-Specific Sparse Feature Masks for Molecular Toxicity Prediction with Chemical Language Models** — K.S. Lee, …, D.D. Wang. ICBCB, 2025-12-12. Sparse feature masks for interpretable toxicity prediction. https://doi.org/10.1109/ICBCB69424.2026.11621727

Probing / "does pretraining teach chemistry":

- **[Kar26] Probing Chemical Language Models: Effects of Pre-training and Fine-tuning** — Karnysheva, Klakow, Lee (Saarland). arXiv, 2026-07-02. Method: probe 78 molecular substructures across 8 pretrained + 6 random CLMs. Finding: pretraining improves structure awareness in **upper layers**; random models already encode **rings in layer 1**; fine-tuning selectively reshapes **task-relevant** substructures. Strong random-baseline design. https://doi.org/10.48550/arXiv.2607.02140
- **[Sie26] Does Pretraining Teach Chemical Language Models Chemistry?** — Sieben, Zimmermann. 2026-10-14. Probing study questioning how much chemistry CLM pretraining imparts. https://arxiv.org/abs/2610.18651
- **[Ngu26c] Functional-Group Knowledge Is Structured in Language Models Before It Is Spoken** — T. Nguyen, …, J. Ock. 2026-11-03. Functional-group knowledge is internally structured prior to being expressed in output. https://arxiv.org/abs/2611.03684
- **[Jin26] SMolLM: Small Language Models Learn Small Molecular Grammar** — Jindal, Ju. arXiv, 2026-05. Small models learn molecular grammar; interpretable grammar acquisition. https://doi.org/10.48550/arXiv.2605.06322
- **[Fen25b] Beyond performance: how design choices shape chemical language models** — Fender, Gut, Lemmin (Bern). J. Cheminformatics, 2025. Representation/tokenizer/design-choice analysis of CLMs. https://doi.org/10.1186/s13321-025-01099-w
- **[Cin25] Property-driven localization and characterization in deep molecular representations** — Cintas, …, Speakman (IBM). Scientific Reports, 2025. Localizes property-associated subspaces in molecular representations. https://doi.org/10.1038/s41598-025-09717-1
- **[Pin25] Superior Molecular Representations from Intermediate Encoder Layers** — L. Pinto. 2025-06-06. Finding: **intermediate** encoder layers give better molecular representations than final layers (layer-wise probing). https://arxiv.org/abs/2506.06443
- **[Rot25] Unraveling learning characteristics of transformer models for molecular design** — Roth, Bajorath (Bonn). Patterns, 2025-10-01. Model: sequence-based generative compound design transformers. Finding: predictions depend on **compound memorization** and train/test similarity; models **did not learn target-sequence information relevant to ligand binding** — statistical pattern-association, cautioning against over-interpretation. (Pairs with [Baj26].) https://doi.org/10.1016/j.patter.2025.101392
- **[Baj26] Transformer Learning in Sequence-Based Drug Design Depends on Compound Memorization and Similarity of Sequence–Compound Pairs** — Bajorath. Molecular Informatics, 2026. Follow-up confirming memorization/similarity dependence. https://doi.org/10.1002/minf.70016

Robustness / notation-shortcut / attribution (chemical LMs):

- **[Yos23] Difficulty in chirality recognition for Transformer architectures learning from string representations** — Yoshikai, …, Kusuhara. Nature Communications, 2023 (anchor paper). Transformers struggle with chirality from SMILES strings. https://doi.org/10.1038/s41467-024-45102-8
- **[Gan25b] Measuring Chemical LLM robustness to molecular representations** — Ganeeva, …, Tutubalina. J. Cheminformatics, 2025. SMILES-variation framework; CLM predictions unstable to representation changes. https://doi.org/10.1186/s13321-025-01079-0
- **[Kik25] Molecular notation can act as a shortcut in AI benchmark evaluation** — Kikuchi, …, Mizuno. 2025. Notation itself is a benchmark shortcut. https://arxiv.org/abs/2505.07139
- **[Mis26] Notation matters: cross-representation inconsistency in chemistry language models and its mechanistic origins** — A. Mishra. Digital Discovery, 2026. Mechanistic origins of cross-representation inconsistency. https://doi.org/10.1039/d6dd00309e
- **[Raj26] Rethinking Molecular Text Representations for LLMs** — Raja, Morris, Chai (Oxford). arXiv, 2026. Empirical study of molecular text representations in LLMs. https://doi.org/10.48550/arXiv.2606.03057
- **[Saa26] Same molecule, Different Knowledge: Cross-Identifier Binding Failures in LLMs for chemistry** — Saadi, Wang. 2026-10. LLMs give inconsistent knowledge across molecular identifiers. https://arxiv.org/abs/2610.30903
- **[Hod24] What can attribution methods show us about chemical language models?** — Hödl, …, Robinson. Digital Discovery, 2024-07. Early attribution-method study on CLMs (no PDF in corpus). https://doi.org/10.1039/d4dd00084f
- **[Ala26b] Which chemical features are captured by ChemBERTa's attention?** — Alavi, Zare-Mirakabad, Gharaghani. J. Comput.-Aided Mol. Design, 2026. Attention analysis of ChemBERTa. https://doi.org/10.1007/s10822-026-00889-0

## A2. Molecular GNNs & molecular foundation models — explainability + representation geometry

GNN explainability (mature sub-field; representative selection — these are mostly attribution/concept methods rather than mechanistic internals):

- **[Pan24c] FragNet: A GNN for Molecular Property Prediction with Four Levels of Interpretability** — Panapitiya, …, Saldanha (PNNL). JACS, 2024-10. Built-in atom/bond/fragment/molecule-level interpretability. https://doi.org/10.1021/jacs.5c22620
- **[Urr26] Explainable AI reveals divergent learning in pharmacophore-based hierarchical pooling GNNs** — Urrutia, Mastropietro, Bajorath. Scientific Reports, 2026-06. XAI shows pharmacophore GNNs learn divergently from intended chemistry. https://doi.org/10.1038/s41598-026-59947-0
- **[Kom25c] Explainable GNNs in Chemistry: Combining Attribution and Uncertainty Quantification** — Komissarov, …, Sach-Peltason (Roche). JCIM, 2025-07. https://doi.org/10.1021/acs.jcim.5c01003
- **[Hom26] Interpreting GNNs with Myerson values for cheminformatics** — Homberg, …, Koch. Digital Discovery, 2026. Game-theoretic (Myerson/Shapley-on-graphs) attribution. https://doi.org/10.1039/d5dd00577a
- **[Mus25] Fragment-Wise Interpretability in GNNs via Molecule Decomposition and Contribution Analysis** — Musiał, Zieliński, Danel. arXiv, 2025-08. https://doi.org/10.48550/arXiv.2508.15015
- **[Riv26] GlassMol: Interpretable Molecular Property Prediction with Concept Bottleneck Models** — Rivera, …, Ding. PAKDD, 2026. Concept-bottleneck molecular models. https://doi.org/10.48550/arXiv.2603.01274
- **[Teu24] Global Concept Explanations for Graphs by Contrastive Learning** — Teufel, Friederich (KIT). xAI, 2024. https://doi.org/10.48550/arXiv.2404.16532
- **[Wan24e] Explainable Molecular Property Prediction: Aligning Chemical Concepts with Predictions via Language Models** — Z-Z. Wang, …, K. Tan. IEEE TPAMI, 2024. https://doi.org/10.1109/TPAMI.2026.3664098

Molecular representation geometry / "what physics is encoded":

- **[Eda25] Universally Converging Representations of Matter Across Scientific Foundation Models** — Edamadaka, …, Gómez-Bombarelli (MIT). arXiv, 2025-12-03. Finding: different molecular/materials foundation models **converge on a shared representation of matter** (Platonic-representation-style). https://doi.org/10.48550/arXiv.2512.03750
- **[Ste26c] Information Routing in Atomistic Foundation Models: How Task Alignment and Equivariance Shape Linear Disentanglement** — J. Steier. arXiv, 2026-03. How equivariance/task-alignment shape linear decodability in atomistic FMs. https://doi.org/10.48550/arXiv.2603.03155
- **[Sul26] Structural Hierarchy and Geometry in Molecular Representation Learning** — Sulu, Di Fruscia, Weber (Delft). arXiv, 2026-08. Hierarchy/geometry of learned molecular representations. https://doi.org/10.48550/arXiv.2608.29886
- **[Fal24] Pretraining graph transformers with atom-in-a-molecule quantum properties for improved ADMET modeling** — Fallani, …, Chernichenko. J. Cheminformatics, 2024. Quantum-property pretraining improves ADMET; interpretable spectral perception (see also [Fal24b]). https://doi.org/10.1186/s13321-025-00970-0

## A3. Generative molecular models — GFlowNets, diffusion, VAE (interpretability + control)

- **[Ami25b] Interpreting GFlowNets for Drug Discovery: What probes can and cannot show** — AmirthaVarshiniA, Ranasinghe, Tam. 2025-11-24. Model: **SynFlowNet** (synthesis-aware GFlowNet, QED reward). Method: gradient saliency + counterfactual edits + under/overcomplete + **BatchTopK SAE**, with shuffled-label, RDKit-descriptor, scaffold-disjoint, cross-seed, and **architecture-matched untrained** controls. Finding (important): physicochemical decodability is **as high in an untrained net** — reflects architecture/featurization, not learned policy; surviving claims are narrow (overcomplete SAE beats undercomplete; some per-halogen/boron detectors emerge but are seed-unstable). A reusable control protocol. https://arxiv.org/abs/2511.19264  (companion: **[Ami25]** "Extracting Actionable Insights for Medicinal Chemistry," https://doi.org/10.48550/arXiv.2511.19264)
- **[Mas26b] Explaining a molecular diffusion model** — Mastropietro, Bajorath (Bonn). Cell Reports Physical Science, 2026-04. Interpretability of a molecular **diffusion** generator (one of the very few mechanistic treatments of molecular diffusion). https://doi.org/10.1016/j.xcrp.2026.103270
- (VAE latent steering: see **[Ela26]** in A1.)

## A4. Co-folding / docking / affinity models — INTERNAL interpretability

This is the frontier most relevant to a protein–ligand co-folding thesis.

- **[Mig26] PairSAE: Mechanistic Interpretability from Pair Representations in Protein Co-Folding** — Migliorini, Rontogiannis, Guitchounts, Franklin, Elaldi, Viessmann (Flagship/Prescient-style team). arXiv, 2026-06-25. Model: **Boltz-2** on PLINDER protein–ligand complexes. Method: **N-mode SVD of the pairwise tensor → token-wise interaction roles → shared SAE** decoding into both sequence and pair reps (solves pairformer quadratic blow-up). Finding: interpretable features align with UniProt annotations and **predict Boltz-2 affinity**. https://doi.org/10.48550/arXiv.2606.27440
- **[Par26] How Do Co-folding Models Organize Structural Information?** — Park, Kim, Moon, Kim, Jeon, W.Y. Kim (KAIST). bioRxiv, 2026-09-08. Model: **Boltz-1 trunk**. Method: decompose pair rep into **intra-/inter-chain** blocks; ablation, layer-wise geometry, activation patching, apo–holo patching. Finding: three info streams (single = token-internal atomic detail; intra-chain pair = chain geometry; inter-chain pair = relative arrangement); **Mix–Compress–Refine** depth trajectory; ligand-induced backbone changes **reconciled inside the diffusion module**, not the trunk. https://doi.org/10.64898/2026.09.06.749739
- **[Jed26b] Probing and steering biology across Boltz-1's trunk–diffusion boundary** — Jedryszek, Xie, Winnifrith, Hasson, Ślesak, Wicks, …, Crook (Oxford/OMass-adjacent). arXiv, 2026-08-11. Model: **Boltz-1** Pairformer trunk + diffusion module. Method: linear probes + SAEs + causal interventions. Finding: trunk makes geometry and sequence-chemistry linearly decodable; in diffusion, **secondary structure transfers but sequence chemistry attenuates**; helix/coil steering works dose-dependently but a highly predictive **β-strand direction does not steer** → **decodable ≠ causal**. Releases trunk+diffusion SAEs and activations. https://doi.org/10.48550/arXiv.2608.11475
- **[Fel26] AlphaInterp: Mechanistic Interpretability of AlphaFold 3** — Feldman, Skolnick (Georgia Tech). bioRxiv, 2026-04-23. Model: **AlphaFold 3**. Method: probe 4 checkpoints + causal activation patching across the forward pass. Finding: Pairformer compresses co-evolutionary input into a compact latent geometry; predicted **confidence is manipulable/transferable**; representational coherence **requires comparative evolutionary context** (collapses without MSA) — AF3 behaves like advanced fold-recognition. https://doi.org/10.64898/2026.04.22.720175
- **[Fel26b] A Shortcut to Structure in AlphaFold 3** — Feldman, Skolnick. arXiv, 2026-10-06. Model: AF3; causal interventions + probing every Pairformer block. Finding: the **MSA is a structural shortcut**; restoring MSA-enriched pair rep at **only 40 residues** recovers global topology; fold becomes recoverable by **block ~9/48**; an alignment from a *different* protein sharing the fold raises median TM-score 0.44→0.72 — AF3 reads **"a description of the fold itself," transferable between homologs**, not the query's own evolution. https://arxiv.org/abs/2610.08937
- **[Lu26] Two Stages of Folding: Convergent Mechanisms in AI Protein Folding Trunks** — Lu, Brinkmann, Huber, Mueller, Belinkov, Bau, Wendler (Northeastern/Technion). 2026-02-05. Models: **ESMFold, OpenFold, Boltz-1** trunks. Method: causal interventions. Finding: shared **two-stage** computation (early: charge/biochemical signals into pair rep; late: distance/contact), causally verified by steering; pairwise states **linearly interchangeable across models**. https://arxiv.org/abs/2602.06020
- **[Liu26] Structure, Not Sequence: Tracking and Validating Folding Mechanisms in ESMFold's Trunk with SAEs** — B-H. Liu, …, A-D. Zhang. 2026-10-26. Model: **ESMFold trunk**. Method: SAEs + LLM-assisted concept interpretation + gradient attribution across blocks. Finding: single rep carries inherited semantics, **pair rep computes geometry**; concept activations **forecast the model's own fold quality**. https://arxiv.org/abs/2610.36226
- **[Roz26] DEEP: Extracting Latent Energetic Landscapes from Protein Structure Predictors** — Rózsa, Erdős. bioRxiv, 2026-10-15. Extracts energetics-like signals from structure predictors. https://doi.org/10.64898/2026.10.15.759389
- **[Suz26] Biasing Conformational Sampling in AlphaFold 3 and Boltz-2 via Pair Representation Scaling** — Suzuki, Amagasa. JCIM, 2026. Scaling the pair representation biases conformational sampling — mechanistic lever on AF3/Boltz-2. https://doi.org/10.64898/2026.01.23.701250
- **[Jun26] Boltz-Perturb: Training-Free Conditioning Perturbation** — Jung, Lee, Cheng (Merck). bioRxiv, 2026-08. Perturbation of conditioning to improve co-folding diversity/accuracy; interpretability-adjacent control. https://doi.org/10.64898/2026.08.05.742877
- **[Jan26b] A Systematic Evaluation of Co-folding Model Representations for Small-Molecule Learning** — Jang, …, Ahn (KAIST). arXiv, 2026-02. Probes whether co-folding **internal representations** transfer to small-molecule property learning. https://arxiv.org/abs/2602.13249
- **[Sar25] Can We Extract Physics-like Energies from Generative Protein Diffusion Models?** — Sarma, …, J.J. Gray (JHU). bioRxiv, 2025-11-28 (also PRX Life 2026, **[Ano26]**). Tests whether protein diffusion models encode physics-like energy functions. https://doi.org/10.1101/2025.11.28.690021

## A5. Co-folding / docking / affinity — PHYSICS-vs-SHORTCUT, memorization, leakage-aware evaluation

The most-cited critical thread; directly tests "physics vs shortcut learning."

- **[Mas25] Investigating whether deep learning models for co-folding learn the physics of protein–ligand interactions** — Masters, Mahmoud, Lill (Basel). **Nature Communications, 2025-10-06; 118 citations.** Models: **RoseTTAFold-All-Atom, AlphaFold3**, etc. Method: adversarial perturbations grounded in physical/chemical/biological principles. Finding: co-folding models produce **physically unrealistic** structures and are **insensitive to significant binding-site modifications** → overfitting to training-data features, not physics. The anchor paper. https://doi.org/10.1038/s41467-025-63947-5  (preprint **[Mas24]**: https://doi.org/10.1101/2024.06.03.597219)
- **[Skr26] Evaluating generalization in protein–ligand cofolding methods** — Škrinjar, Eberhardt, Studer, Tauriello, Schwede, Durairaj (SIB). **Nature Structural & Molecular Biology, 2026-05; 37 cit.** Introduces **Runs N' Poses** (2,600 post-cutoff complexes). Finding: leading all-atom co-folding methods **largely memorize ligand poses** from training data. https://doi.org/10.1038/s41594-026-01797-5  (preprint **[Skr25]** "Have protein-ligand cofolding methods moved beyond memorisation?", 101 cit.: https://doi.org/10.1101/2025.02.03.636309)
- **[Bre26] Assessing Boltz-2 Performance for the Binding Classification of Docking Hits** — Bret, Sindt, Rognan (Strasbourg). JCIM, 2026-01. Finding: Boltz-2 affinity classification is **insensitive to binding-site mutations and even target shuffling** in some cases — "hidden features" govern predictions, not interaction physics. https://doi.org/10.1021/acs.jcim.5c02630
- **[Lyu26] Physics-Grounded Evaluation to Guide Accurate Biomolecular Prediction** — Lyu, …, Herschlag (Yale). bioRxiv, 2026. Physics-grounded evaluation protocol for biomolecular predictors. https://doi.org/10.1101/2025.06.30.662466
- **[Li25h] Decoding the limits of deep learning in molecular docking for drug discovery** — Y. Li, …, D-S. Cao. Chemical Science, 2025-08; 48 cit. Systematic limits of DL docking. https://doi.org/10.1039/d5sc05395a
- **[Men26b] AlphaFold3 for Structure-guided Ligand Discovery** — Menon, …, Lyu (UCSF). bioRxiv, 2026-06. Prospective assessment of AF3 for ligand discovery. https://doi.org/10.64898/2025.12.04.692352
- **[She25] Unlocking the application potential of AlphaFold3-like approaches in virtual screening** — Shen, …, Hou (Zhejiang). Chemical Science, 2025-12. https://doi.org/10.1039/d5sc06481c
- **[Kim25b] Large-scale prospective evaluation of co-folding across hundreds of Mac1–ligand complexes and three virtual screens** — Kim, …, Fraser (UCSF). eLife, 2025-12. Prospective (not retrospective) co-folding evaluation. https://doi.org/10.64898/2025.12.25.696505
- **[Obe26b] Benchmarking Protein-Ligand Cofolding Models: Correct Ligand Placement Is Decoupled From Accurate Protein Conformation** — Obendorf, …, Wolber. JCIM, 2026. https://doi.org/10.1021/acs.jcim.6c02169
- **[Sun26d] A curated benchmark for cofolding models on kinase conformational states** — Sun, Head-Gordon (Berkeley). npj Drug Discovery, 2026. https://doi.org/10.1038/s44386-026-00068-z
- **[Xu26h] Multiparadigm Benchmark of Molecular Docking: From Physics to Co-Folding and Hybrid Models** — Xu, …, Katritch (USC). JCIM, 2026. https://doi.org/10.1021/acs.jcim.6c01293

Binding-affinity data leakage / bias (quantifies the shortcut source):

- **[Mat26c] Identifying and Addressing Systematic Data Leakage in Protein-Ligand Affinity Benchmarks** — Mattsson, W.P. Walters (Relay). bioRxiv, 2026-06. https://doi.org/10.64898/2026.06.29.735309
- **[Li26f] Leak Proof PDBBind: A Reorganized Dataset for More Generalizable Binding Affinity Prediction** — J. Li, …, Head-Gordon. J. Phys. Chem. B, 2026-01; 25 cit. https://doi.org/10.1021/acs.jpcb.5c08598
- **[Gra25] Resolving data bias improves generalization in binding affinity prediction** — Graber, …, Buller. **Nature Machine Intelligence, 2025; 68 cit.** https://doi.org/10.1038/s42256-025-01124-5  (follow-up **[Gra26c]** "How Bias Shapes the Leaderboard," bioRxiv 2026: https://doi.org/10.64898/2026.07.27.740774)
- **[Wan26s] Leakage-controlled benchmarking reveals generalization limits of DL for protein–ligand binding affinity** — Wang, Cheng. bioRxiv, 2026-09. https://doi.org/10.64898/2026.09.17.752433
- **[Kop25] Generalization Beyond Benchmarks: Evaluating Learnable Scoring Functions on Unseen Targets** — Kopko, …, Sivic. arXiv, 2025-12. https://doi.org/10.48550/arXiv.2512.05386
- **[Yaz26] MIRAGE: Measuring Interpolation and Redundancy in Affinity GEneralization** — Yazdani-Jahromi, Padhi, Garibay. 2026-09. https://arxiv.org/abs/2609.14491
- **[Avd25] All That Glitters Is Not Gold: Rigorous Evaluation of Proteochemometric Models** — Avdiunina, …, Isayev (CMU). JCIM, 2025. https://doi.org/10.1021/acs.jcim.5c00395
- **[Err24] Assessing interaction recovery of predicted protein–ligand poses** — Errington, …, Dreyer (Exscientia). J. Cheminformatics, 2024; 36 cit. PLIF-based interaction-recovery metric beyond RMSD. https://doi.org/10.1186/s13321-025-01011-6
- **[Vol22] On the Frustration to Predict Binding Affinities from Protein–Ligand Structures with Deep Neural Networks** — Volkov, …, Rognan. J. Med. Chem., 2022; 177 cit. Foundational "models exploit ligand/protein bias, not interactions." https://doi.org/10.1021/acs.jmedchem.2c00487
- **[Wan26aa] On the Reliability of AI Methods in Drug Discovery: Evaluation of Boltz-2** — Wan, …, Coveney (UCL). arXiv, 2026-03. https://doi.org/10.48550/arXiv.2603.05532
- **[Mir26] AI-guided competitive docking for virtual screening and compound efficacy prediction** — Mirgaux, …, Wintjens. npj Drug Discovery, 2026. https://doi.org/10.1038/s44386-026-00039-4
- Docking generalization methods (context): **[Cor24]** Deep Confident Steps to New Pockets (Corso et al., DiffDock generalization, 112 cit., https://doi.org/10.48550/arXiv.2402.18396); **[Jai24]** Fair comparisons of DL vs conventional docking (Jain, Cleves, Walters, https://doi.org/10.48550/arXiv.2412.02889).

## A6. Antibody / peptide / binder design — interpretability

Genuinely thin as *mechanistic* interpretability (most design papers do not open the model). The real interpretability entries:

- **[Haq25] Mechanistic Interpretability of Antibody Language Models Using SAEs** — Haque, Turnbull, A. Parsan, N. Parsan, Yang, Beukenhorst, Deane (Oxford OPIG). arXiv, 2025-12-05. Model: autoregressive **antibody LMs**. Method: **TopK and Ordered SAEs** + generative steering. Finding: TopK SAEs reveal biologically meaningful latents but high feature–concept correlation **does not guarantee causal control**; **Ordered SAEs** give reliable steerability at the cost of interpretability. The key antibody-LM interp paper. https://doi.org/10.48550/arXiv.2512.05794
- **[Mur25] Mechanistic interpretability of fine-tuned protein language models for nanobody thermostability prediction** — Murakami, Hashidate, Matsunaga. Bioinformatics, 2025-12. Interpreting a fine-tuned PLM for **nanobody** thermostability. https://doi.org/10.64898/2025.12.22.695878
- **[Sar26d] MoE-Bind: Guiding De Novo Protein Binder Generation with Sparse Experts** — Sarkar, Sarkar. bioRxiv, 2026-06. Sparse-expert (interpretable) guidance of binder generation. https://doi.org/10.64898/2026.06.13.732043
- **Gap note:** I found **no** paper combining mechanistic-interpretability feature steering of a protein/antibody model with **prospective wet-lab validation** of the designed binders. Design frameworks with wet-lab validation (Germinal, Latent-X, contrastive peptide binders) exist but treat the model as a black box; conversely interp papers (Haq25, InterPLM) steer generation but validate in-silico only. This is an open niche.

---

# PART B — PROTEIN LANGUAGE & STRUCTURE MODELS

(The 15 Nov deep search covered this thoroughly; below is the structured map, then §B7 on what it missed / what appeared since.)

## B1. SAE feature discovery (foundational)

- **[Sim24] InterPLM: Discovering Interpretable Features in PLMs via SAEs** — Simon, Zou (Stanford). bioRxiv 2024-11 → **Nature Methods 2025**; 168 cit. SAEs on **ESM-2** recover thousands of interpretable latents (binding sites, motifs, domains) vs weak neuron alignment (superposition); auto-interp + annotation-filling + steering; interPLM.ai. https://doi.org/10.1038/s41592-025-02836-7
- **[Ada25] From Mechanistic Interpretability to Mechanistic Biology** — Adams, Bai, Lee, Yu, AlQuraishi (Columbia). bioRxiv, 2025-02; 64 cit. SAEs on ESM-2 residual stream; generic vs family-specific features; linear probing recovers thermostability/localization determinants. https://doi.org/10.1101/2025.02.06.636901
- **[Guj25] Sparse autoencoders uncover biologically interpretable features in PLM representations** — Gujral, Bafna, Alm, Berger (MIT). **PNAS, 2025-08; 77 cit.** SAEs + transcoders on protein- and residue-level reps; GO-enriched interpretable features; more interpretable than neurons. https://doi.org/10.1073/pnas.2506316122
- **[Par25] Towards Interpretable Protein Structure Prediction with SAEs** — N. Parsan, D. Yang, J.J. Yang (Reticular). arXiv, 2025-03. Scales SAEs to **ESM2-3B** (ESMFold base) + **Matryoshka SAEs**; steers ESMFold (solvent accessibility). https://doi.org/10.48550/arXiv.2503.08764
- Extensions: **[Ste26]** Structure tokens sharpen the feature vocabulary (ESM-2 vs ESM-3; 78% feature convergence; single geometric head L0H7 bottleneck; https://doi.org/10.64898/2026.05.12.724593); **[Liu25]** ProtSAE semantically-guided SAEs (https://doi.org/10.48550/arXiv.2509.05309); **[Hou26]** MotifAE unsupervised functional sequence patterns, Nat. Commun. 2026 (https://doi.org/10.1038/s41467-026-77333-2); **[Set26]** geometric annotations for latent features (https://doi.org/10.48550/arXiv.2608.26419); **[Hu26]** SAE features of **ESM-C** for enzyme function (https://arxiv.org/abs/2606.12209); **[Yu26b]** VFUSE virulence-factor SAEs (https://doi.org/10.64898/2026.06.08.730928); **[Kul26]** InterPLM features predict neuropeptide precursors (https://doi.org/10.64898/2026.08.20.746077); **[Sil25]** functional annotation with latent features (https://doi.org/10.1101/2025.10.02.680154); **[Zhu26b]** "Auditing Protein-Protein Interaction Signals with Sparse Autoencoder Fingerprints," Zhu et al., bioRxiv 2026-10 (DOI not fetched; cite key resolvable in the workspace).

## B2. Circuits, transcoders, causal feature analysis

- **[Tsu26] Protein Circuit Tracing via Cross-layer Transcoders (ProtoMech)** — Tsui, Talreja, Saeedi, Aghazadeh (Georgia Tech). arXiv, 2026-02. **Cross-layer transcoders** on ESM-2; recovers 82–89% performance; compressed circuits (<1% latent space) for binding/signaling/stability; steering → high-fitness design. https://doi.org/10.48550/arXiv.2602.12026
- **[Tsu26b] Circuit Tracing in Autoregressive PLMs (ProGenMech)** — Tsui, Deinzer, Saeedi, Aghazadeh. arXiv, 2026-06. CLTs on **ProGen3** (MoE); zero-shot circuit discovery for generation + fitness. https://doi.org/10.48550/arXiv.2606.16044
- **[Nai25] Motif-gated domain recognition drives contact prediction in PLMs** — Nainani, Reimer, Watts, Jensen, Green (UMass). bioRxiv, 2025-08. First **circuits-style causal** pLM study: activation patching in SAE-latent space; two-step motif→domain computation; <0.03% of latent–token pairs suffice. https://doi.org/10.1101/2025.08.22.671739
- **[Moh26] Towards Sparse Causal Features for Zero-shot Mutation Effect Prediction** — Mohanty, Phutela, Green. bioRxiv, 2026-09. SAE + integrated-gradients + activation patching on **ESM-2 650M**; first causal feature-level account of zero-shot mutation effects; indirect-effect selection > activation magnitude. https://doi.org/10.64898/2026.08.28.747907
- **[Kes26] Induction Meets Biology: Mechanisms of Repeat Detection in PLMs** — Kesten-Pomeranz, Nikankin, Reusch, Tsaban, Schueler-Furman, Belinkov (HUJI/Technion). arXiv, 2026-02. **Induction heads** + biologically specialized neurons for exact/approximate repeat detection. https://doi.org/10.48550/arXiv.2602.23179
- **[Jed26c] Retrieval and competition: how a protein foundation model starts a protein** — Jedryszek, Crook. arXiv, 2026-05. Circuit for the "proteins start with Met" rule in **ESM2-8M**: positional-prior retrieval, not biological recognition; fails where biology diverges. https://doi.org/10.48550/arXiv.2605.16331
- **[Nor23] Identification of Knowledge Neurons in PLMs** — Nori, Singireddy, Ten Have. arXiv, 2023. Early knowledge-neuron localization. https://doi.org/10.48550/arXiv.2312.10770
- **[Tsu24] On Recovering Higher-order Interactions from PLMs** — Tsui, Aghazadeh. arXiv, 2024. Higher-order epistatic interactions in PLM outputs. https://doi.org/10.48550/arXiv.2405.06645

## B3. Folding trunks & co-folding structure models

(Primary entries given in A4 — they are both drug-discovery and PLM interpretability.) Add:
- **[Hay25] Quantifying the Role of OpenFold Components in Protein Structure Prediction** — Hayes, Krishnan. arXiv, 2025-11. Component-ablation of OpenFold. https://doi.org/10.48550/arXiv.2511.14781
- **[Gut24b] Dissecting AlphaFold2's capabilities with limited sequence information** — Gut, Lemmin. Bioinformatics Advances, 2024. https://doi.org/10.1093/bioadv/vbae187
- **[Gut26] Synthetic sequence alignments as programmable probes of learned conformational landscapes** — Gut, Kleinschmidt, Lemmin. Bioinformatics, 2026. https://doi.org/10.1093/bioinformatics/btag605
- **[Meh26] Neural spectroscopy of AlphaFold2 reveals encoded protein conformational landscapes** — Mehta. arXiv, 2026-07. https://doi.org/10.48550/arXiv.2607.16087
- **[Zar25] FoldSAE: Learning to Steer Protein Folding Through Sparse Representations** — Zarzecki, …, Deja. 2025-11. https://arxiv.org/abs/2511.22519

## B4. Attention, coevolution, structural mechanisms

- **[Zha24] Protein language models learn evolutionary statistics of interacting sequence motifs** — Zhang, Wayment-Steele, Brixi, Wang, Kern, Ovchinnikov. **PNAS, 2024; 181 cit.** "Categorical Jacobian" shows **ESM-2** stores coevolutionary pairwise statistics (like MRF/Potts); contacts stored as motifs. Anchor for coevolution analysis. https://doi.org/10.1101/2024.01.30.577970
- **[Ell25] Transformers trained on proteins can learn to attend to Euclidean distance** — Ellmen, …, Deane. TMLR, 2025. Attention can encode 3D distance. https://doi.org/10.48550/arXiv.2502.01533
- **[Nay24] Paying attention to attention: high attention sites as indicators of protein family/function** — Nayar, Tartici, Altman (Stanford). PLOS Comput. Biol., 2024. https://doi.org/10.1101/2024.12.13.628435
- **[Moy26] Topological attention asymmetry in ESM-2 encodes the allosteric hierarchy of the A2A receptor** — Moya-García. bioRxiv, 2026. https://doi.org/10.64898/2026.04.24.720732

## B5. Steering & design applications of interpretable features

- **[Gar25] Interpreting and Steering PLMs through SAEs** — Garcia, Ansuini (AREA Trieste). arXiv, 2025-02; 27 cit. https://doi.org/10.48550/arXiv.2502.09135
- **[Hua25] Steering Protein Language Models** — L-K. Huang, …, Yao (Tencent). arXiv, 2025-09. https://doi.org/10.48550/arXiv.2509.07983
- **[Ban25] Automated Neuron Labelling Enables Generative Steering and Interpretability in PLMs** — Banerjee, …, Tam. arXiv, 2025-07. https://doi.org/10.48550/arXiv.2507.06458
- **[Hin26] Generative Latent Priors for Interpreting and Steering PLMs** — Hinjos, Urcelay. 2026-11-06. https://arxiv.org/abs/2611.09818
- **[Wan26] Improving Variant Effect Prediction by Steering Sparse Mechanistic Features in PLMs** — M-Q. Wang, …, Z-X. Ren. bioRxiv, 2026. https://doi.org/10.64898/2026.05.12.724472
- **[Tsu25] Sparse Autoencoders for Low-N Protein Function Prediction and Design** — Tsui, Talreja, Aghazadeh. arXiv, 2025-08. https://doi.org/10.48550/arXiv.2508.18567
- **[Ism24] Concept Bottleneck Language Models For protein design** — Ismail, …, Frey (Genentech/Prescient). arXiv, 2024. Intervenable concept bottlenecks in protein design. https://doi.org/10.48550/arXiv.2411.06090

## B6. Critical evaluation, validation frameworks, SAE methodology

- **[Orl26] What Do Biological Foundation Models Compute? SAEs from Feature Recovery to Mechanistic Interpretability** — Orlov, Makus, Ashniev, Orlova, Nikitin. bioRxiv, 2026-03. **Systematic review**; warns annotation-matching validation is **circular**; proposes representational/computational/causal three-level framework + experimental validation (DMS, MPRA). https://doi.org/10.64898/2026.03.04.709491
- **[Cho26] Mechanistic Interpretability for PLMs: A Validation Framework** — Chon, Andreopoulos (SJSU). bioRxiv, 2026-05. https://doi.org/10.64898/2026.05.29.727021
- **[Hea25] Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers** — Heap, …, Aitchison (Bristol). 2025; 35 cit. **Random-model baseline** critique of SAE auto-interp. https://arxiv.org/abs/2501.17727
- **[Sim26] On the Relationship Between Activation Outliers and Feature Death in SAEs** — Simon, Adams, Zou. arXiv, 2026. SAE training pathologies. https://doi.org/10.48550/arXiv.2605.31518
- **[Son26] Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs** — X-C. Song, …, K. Zhang. ACL 2026. Cross-cutting SAE methodology (feature consistency across seeds). https://doi.org/10.18653/v1/2026.acl-long.99
- **[Whi26] High-resolution dissection of concept acquisition in different families of PLMs** — Whitfield, …, Fournier. bioRxiv, 2026. https://doi.org/10.64898/2026.07.20.739599
- **[Vin25] Trainable subnetworks reveal structure-knowledge organization in PLMs** — Vinod, …, K.K. Yang. PLOS Comput. Biol., 2025. https://doi.org/10.1371/journal.pcbi.1013925
- **[Bes25] Towards Understanding the Shape of Representations in PLMs** — Beshkov, Malthe-Sørenssen. arXiv, 2025-09. Representation geometry. https://doi.org/10.48550/arXiv.2509.24895
- **[Jed26] Efficient Auto-Interpretability of AI Models in Biology** — Jedryszek, Crook. arXiv, 2026-08. https://doi.org/10.48550/arXiv.2608.27754
- **[Hun26] Towards the explainability of protein language models** — Hunklinger, Ferruz (Barcelona). **Nature Machine Intelligence, 2026.** Review/perspective. https://doi.org/10.1038/s42256-026-01232-w
- **[Can26] Language Modeling Materializes a World Model of Protein Biology** — Candido, …, Rives (EvolutionaryScale, ESM3 team). bioRxiv, 2026-06; 99 cit. Argues ESM-family models build an internal "world model." https://doi.org/10.64898/2026.06.03.729735
- **[Ste26b] A geometric atlas of how ESM3 organizes modalities across depth** — Steenwyk. bioRxiv, 2026. https://doi.org/10.64898/2026.07.08.737319

## B7. What the 15-Nov PLM search missed or that appeared since (surfaced by the 19-Nov drug search + recency pass)

These are PLM/structure-model interpretability papers **not in** the 72-paper 15 Nov list:

- **[Wan26ab] A germline shortcut in protein language model retrieval of adaptive immune receptors** — S-Y. Wang, …, Standley. bioRxiv, 2026-07. PLMs retrieve immune receptors via a **germline shortcut** rather than true recognition — a shortcut-learning result for antibody/TCR contexts. https://doi.org/10.64898/2026.07.21.739963
- **[Zha25b] Hit or Miss: Emergence and Absence of Homo-oligomeric Contacts in PLMs** — Zhang, …, Ovchinnikov. bioRxiv, 2025-11-16. When/why PLMs encode homo-oligomeric contacts (categorical-Jacobian lineage). https://doi.org/10.1101/2025.11.16.688745
- **[Liu25h] Exploring Protein Language Model Architecture-Induced Biases for Antibody Comprehension** — M. Liu, Y-X. Zhang, Y-M. Zhang. arXiv, 2025-12. Architecture-induced biases in antibody PLMs. https://doi.org/10.48550/arXiv.2512.09894
- **[Che26f] Probing Physical Intuition in LLMs through Partner-Dependent Mutation Prediction** — L. Cheng, …, Z-S. Yu. bioRxiv, 2026-10. Tests physics-like intuition via partner-dependent mutation effects. https://doi.org/10.64898/2026.10.09.757856
- **[Baz26] Systematic Profiling of Encoder Architectures and Multimodal Embeddings in Protein Foundation Models (allosteric/orthosteric binding-site prediction)** — Bazarova, Verkhivker. JCIM, 2026-10. Probing binding-site separability across protein FMs. https://doi.org/10.1021/acs.jcim.6c03114
- **[Li26e] A geometry-over-coevolution principle governs protein complex assembly in AlphaFold** — S. Li, Mu, Yan. bioRxiv, 2026-04. Geometry dominates coevolution in AF complex assembly. https://doi.org/10.64898/2026.04.03.716280
- **[Cha26c] When Does Structure Help? The Information Bonus of AlphaFold2 Representations over PLMs** — Chauhan. arXiv, 2026-06. Information-theoretic comparison of AF2 vs PLM representations. https://doi.org/10.48550/arXiv.2606.04228
- **[Rah26b] One Mistake, Three Layers: Persistence Rules Under Depth Correlation** — Rahimimovassagh, …, Yousefi. 2026-11-02 (sibling of ORBIT-FMIB **[Rah26]**, https://arxiv.org/abs/2610.00672). Depth-correlated error persistence in ESM-2. https://arxiv.org/abs/2611.03968
- **[Naw26] Mechanistic Interpretability of PLMs Reveals Encoded Properties of Intrinsically Disordered Proteins** — Naworski, …, Knowles (Cambridge). bioRxiv, 2026-09. IDP structure/function in PLM internals. https://doi.org/10.64898/2026.09.10.750691
- **[Wan26b] PLMs Encode Evolutionary Grammar but Conflate Topological and Thermodynamic Phases** — Y. Wang, …, K. Wei. bioRxiv, 2026-04. https://doi.org/10.64898/2026.04.07.717117
- **[Nam25] PLMs Capture Structural and Functional Epistasis in a Zero-Shot Setting** — Nambiar, …, Maslov. bioRxiv, 2025-09. https://doi.org/10.1101/2025.09.14.676130
- **[Mol25] How Much Does Protein Structure Really Help? (mutation-induced stability)** — Moldwin, Shehu. bioRxiv, 2025-12. Probes marginal value of structure vs sequence. https://doi.org/10.64898/2025.12.22.695878

Also newly surfaced but out-of-scope for A/B (belong to the genomic/single-cell searches): a plant genomic FM mechanistic-interp paper (Kendiukhov, Comput. Biol. Chem. 2026) and histopathology-FM SAEs (Smida & Benesova, 2026). TCR-epitope explainability **[Li26j]** (arXiv 2026) sits at the immuno-drug boundary.

---

# Cross-cutting themes, gaps, and relevance to a co-folding/BSF thesis

1. **"Decodable ≠ causal" is now empirically load-bearing on both sides.** Jed26b (β-strand direction decodes F1=0.82 but won't steer), Ami25b (untrained GFlowNet matches trained), Sri26 ("Probing isn't Causality in Molecular Models" — see below), Hea25 (random transformers pass auto-interp metrics), Orl26/Cho26 (annotation-matching circularity). Any manifold/BSF claim needs an **architecture-matched untrained control + a causal intervention**, exactly the critique already anticipated in the local technical plan.
   - **[Sri26] Probing isn't Causality in Molecular Models** — Srivastava, Chundru, Singh, Singh. bioRxiv, 2026-10. Models: **Uni-Mol, MolCLR**. Method: replace a concept-associated scalar at an intermediate block, measure downstream readout change, vs random directions and architecture-matched random encoders. Finding: in Uni-Mol, block-12 edits predict energy-change sign for 97.6% of held-out pairs; proposes an "evidence ladder" separating recoverability, intervention selectivity, and pretraining advantage. https://doi.org/10.64898/2026.10.05.756781

2. **The Pairformer/trunk–diffusion boundary is the hot 2026 object** and the closest prior art to interpreting a protein–ligand co-folding model: PairSAE [Mig26] (Boltz-2 pair rep, SAE, affinity), Par26 (Boltz-1 intra/inter-chain partition + apo-holo patching), Jed26b (Boltz-1 trunk→diffusion probing/steering), Fel26/Fel26b (AF3 MSA shortcut circuits), Lu26 (ESMFold/OpenFold/Boltz-1 two-stage trunk). A BSF/Jacobian-transport study of a co-folding model would sit directly on top of this cluster — strong novelty pressure to differentiate from PairSAE and Par26 specifically.

3. **Physics-vs-shortcut is settled enough to cite as motivation** (Mas25 Nat Commun 118 cit; Skr25/Skr26 memorization; Bre26 mutation-insensitivity; Gra25 data bias). The newer *mechanistic* papers (Fel26b MSA shortcut, Par26 apo-holo) explain *where in the network* the shortcut lives — that is the bridge from "co-folding models memorize" to "here is the circuit."

4. **Chemical-LM interpretability is the least-mature leg** and the most open: only Ken26l, Var25, Coh25 do SAE/circuit work on molecular transformers, all late-2025/2026; Li26h is the one solid attention-circuit/grokking study. If a thesis wants a ligand-side internal analysis, this is comparatively uncrowded (but note cLMs are sequence-only — no pair geometry, matching the local plan's concern about sequence-only ligand models).

5. **Confirmed genuine gaps:** (a) interpretability-guided design with **prospective wet-lab validation** (none found — see A6); (b) **manifold/continuous-variable SAE recovery** with capacity-matched controls on bio models (the BSF thesis's own niche — not yet occupied, though Set26 "geometric annotations," Bes25/Sul26 representation-shape, and Ste26c equivariant-disentanglement touch the edges); (c) **Jacobian/transport-style cross-layer relevance** on bio models is essentially absent — closest are the categorical-Jacobian coevolution line [Zha24, Zha25b] and cross-layer transcoders [Tsu26, Tsu26b], but none do input→output Jacobian transport with an output-relevance readout.

## Deep-search pointers (for the user, in the Undermind app)
- (A) `app.undermind.ai/projects/33899782-4f3e-48f1-9e4a-a3b6ac8a7af7?path=/Interpretability%20of%20deep%20learning%20models%20in%20drug%20discovery` — 163 papers, finished 2026-11-19.
- (B) `app.undermind.ai/projects/33899782-4f3e-48f1-9e4a-a3b6ac8a7af7?path=/Protein%20model%20mechanistic%20interpretability%20SAE` — 72 papers, finished 2026-11-15.

*All findings above are from Undermind abstracts/summaries; items without a resolvable DOI are given with their arXiv link. No claims were added from model memory.*
