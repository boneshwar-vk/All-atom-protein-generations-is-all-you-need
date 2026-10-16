# Steering head + atom-level interaction module for RFD3 — literature compilation

_Compiled 2026-10-16. Sources: 2 Undermind deep searches (139 + 241 ranked papers) and 6 Claude web-search agents (confidence heads, interaction modules, steering/guidance, atom-level masked pretraining, all-atom interaction modeling, training-data generation, engine-generated-complex training, and a Baker-lab author sweep)._

**Legend.** `[Xxx00]` markers are Undermind workspace cite keys — look them up in the Undermind "New Research Project" workspace for the linked PDF + metadata. Papers found only via web search are given a direct arXiv/bioRxiv/DOI link and should be **dereferenced before citing** — several 2026 IDs are recent preprints with self-reported numbers. Flags from the scouts are preserved.

---

## 0. Project framing

Two modules are in view, both wrapped around **RFdiffusion3 (RFD3)** (`[Wat23]` is RFD1; RFD3 = Butcher et al. 2025, see §7):

1. **A multi-property "steering head"** — a module that reads a target+binder complex (ideally RFD3's own internal/latent representation) and predicts several properties (binding, interface quality, affinity, stability, developability). Used as guidance for *partial / conditional* generation so RFD3 better "understands" how complexes form. → §1, §2, §3.
2. **An atom-level masked-interaction module** — self-supervised, atom-wise masked / denoising pretraining over proteins + ligands + mini-proteins + nucleic acids, learning "is this how atom–atom interactions work?", then generalized to all-atom PPI scoring. → §4, §5. Training data for both, including engine-generated complexes (Rosetta, Protenix-v2, OpenDDE, self-distillation). → §6.

The two clearest **open gaps** (both scouts independently found them) are summarized in §9.

---

## 1. Confidence heads (pLDDT / PAE / pTM / ipTM) and binder ranking

**How the heads work**
- `[Jum21b]` AlphaFold2 — origin of per-residue pLDDT and pairwise PAE, from which pTM is computed.
- `[Eva21]` AlphaFold-Multimer — introduces **ipTM** and the 0.8·ipTM + 0.2·pTM ranking for complexes.
- `[Abr24]` AlphaFold3 — carries pLDDT/PAE/pTM/ipTM into an all-atom diffusion co-folding pipeline.
- `[Woh24]` Boltz-1 and `[Pas25e]` Boltz-2 — open AF3-class confidence modules; Boltz-2 adds an **affinity module** (PairFormer dual head: binding-likelihood + affinity regression) and mixes pLDDT/ipTM into overall confidence.
- `[Tea25]` PXDesign (Protenix) — confidence-filtered binder design on an open AF3 reproduction.

**Reliability of these metrics for PPIs / designed binders** (central to the scoring question)
- `[Ove25]` Predicting Experimental Success in De Novo Binder Design — meta-analysis of 3,766 tested binders; **AF3 ipSAE_min beats ipAE/ipTM**, performance is strongly target-dependent. _Single most on-topic scoring paper._
- `[Dun25]` "Rēs ipSAE loquuntur: what's wrong with AlphaFold's ipTM and how to fix it" — diagnoses ipTM's failure mode, introduces **ipSAE**.
- `[Nor25]` BindEnergyCraft — casts predictors as energy-based models; **pTMEnergy** (from pAE logits) gives denser interface gradient (~98–100% of target residues engaged vs ~22–35% for ipTM) and beats ipTM as a design objective.
- `[Var25]` actifpTM (Varga & **Ovchinnikov**) — refines ipTM so disordered/flexible flanks don't depress interface confidence.
- `[Gen25c]` Assessing scoring metrics for AF2/AF3 complexes — ipLDDT/ipTM/iPAE/pDockQ2 vs DockQ; interface-specific scores beat global ones.
- `[Smo26]` Structural Plausibility Without Binding Specificity — confidence can stay high for plausible but **non-cognate** antibody–antigen pairs. Cautionary.
- `[Liu25ag]` **AF3Score** — a *score-only* adaptation of AF3: scores existing coordinates without re-diffusing; improves binder enrichment. Directly the "predictor-as-scoring-head" pattern.
- `[Gao25e]` CONFIDE — hallucination assessment for reliable structure prediction and design.
- `[Vog26]` Analysing open-source folding models for nanobody binding; `[Uns26]` AntiConf (antibody–antigen precision metric); `[Fie26b]` Benchmarking Boltz-2 for therapeutic Ab–Ag; `[Lan26c]` auto-discovery of interpretable binder filter rules; `[Tan25k]` GeoFlow-V3 antibody design.
- Web: [pDockQ (Bryant 2022, Nat Commun)](https://www.nature.com/articles/s41467-022-28865-w); [actifpTM arXiv](https://arxiv.org/pdf/2412.15970); [AlphaJudge tool (ipTM/pDockQ/pDockQ2/ipSAE/LIS + physics)](https://github.com/KosinskiLab/AlphaJudge); [calibrating AF confidence to DockQ probability (2610.17108)](https://arxiv.org/html/2610.17108); [Pinc probabilistic interaction score (Protein Science 2026)](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.70760); [Boltz-2 reliability evaluation (2603.05532)](https://arxiv.org/html/2603.05532v1) _(verify)_.

---

## 2. Interaction modules (inter-chain pair representation, triangle, co-folding, affinity heads)

- `[Abr24]` AlphaFold3 **Pairformer** — the reference co-folding backbone; pair representation reasons over chain relationships before structure generation.
- `[Ouy25b]` **"Triangle Multiplication Is All You Need" / Pairmixer** — drops triangle *attention*, keeps triangle *multiplicative* updates; up to 4× faster at matched folding/docking accuracy. Directly on the triangle-module design choice.
- `[Par26b]` How Do Co-folding Models Organize Structural Information? — interpretability of what the pair/co-folding stack encodes.
- `[Jan26c]` Co-folding with a Soup of Representations.
- `[Jin26d]` **Promera** — a unified model for structure prediction, *filtering*, and design (Berger lab); conceptually a prediction+steering model.
- Affinity / interface heads: `[Qia25b]` PPAP (interfacial contact-aware attention); `[Par26]` **PreFold-dG** — predicts PPI affinity **from intermediate representations of a folding model** (the "read the co-folding latent" idea); `[Kin25]` fine-tuning Boltz-2 for PPI affinity.
- `[Jin23b]` **DSMBind** — SE(3) **denoising score-matching** for *unsupervised* binding-energy prediction and nanobody design. Closest existing analog to the atom-level denoising-interaction idea (see also §4).
- `[Gai23]` MaSIF / learned surface fingerprints (Gainza & **Correia**) — surface-based interaction design.
- Web (residue-level PPI/contact architectures worth adapting to atoms): [MSA Pairformer (Cell 2026, **Ovchinnikov**)](https://www.sciencedirect.com/science/article/pii/S009286742600749X) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.02.668173v1); [PPLM paired-sequence inter-protein transformer (Nat Commun 2026)](https://www.nature.com/articles/s41467-026-70457-5); [Geometric Transformers for interface contact prediction (ICLR 2022)](https://arxiv.org/abs/2110.02423); [CrossPPI cross-fusion affinity (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.11.25.690371v1.full); [CrossAffinity (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.02.22.707318v1.full).

---

## 3. Steering heads / property-guided conditioning of generators (for RFD3)

**Guidance machinery (how a learned predictor steers the reverse process)**
- [Protein Design with Guided Discrete Diffusion — NOS / LaMBO-2 (Gruver et al., NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/29591f355702c3f4436991335784b503-Paper-Conference.pdf) — **the archetype multi-property steering head**: a regression head on the denoiser's hidden states supplies gradients steering generation toward multiple objectives (affinity + developability).
- [SGPO — steering generative models with experimental data (2505.15093)](https://arxiv.org/abs/2505.15093) — time-dependent predictor p(property | noisy input, t) as the guidance term; plug-and-play beats fine-tuning.
- [ProteinGuide — on-the-fly property guidance (2505.04823)](https://arxiv.org/abs/2505.04823); [Simple Guidance Mechanisms for Discrete Diffusion (2412.10193)](https://arxiv.org/pdf/2412.10193).
- [Context-Guided Diffusion for OOD molecular/protein design (2407.11942)](https://arxiv.org/pdf/2407.11942) — makes the guidance predictor robust **off-distribution** (key failure mode when a head drives generation into novel regions).

**Conditioning / partial diffusion in the RFdiffusion family** (what a head would hook into)
- [RFdiffusion repo — partial diffusion + fold/hotspot conditioning](https://github.com/RosettaCommons/RFdiffusion); [β-pairing-targeted RFdiffusion + partial-diffusion refinement (Nat Commun 2025)](https://www.nature.com/articles/s41467-025-67866-3).
- [Constrained Diffusion with hard structural constraints (2510.14989)](https://arxiv.org/pdf/2510.14989) _(verify)_; [Monte Carlo Tree Diffusion with multiple experts (2509.15796)](https://arxiv.org/pdf/2509.15796) _(verify)_.

**Reward / energy / EBM guidance**
- `[Nor25]` BindEnergyCraft / pTMEnergy (see §1).
- [DRAKES — fine-tuning discrete diffusion via reward optimization (ICLR 2025)](https://arxiv.org/pdf/2410.13643); [Reward-guided iterative refinement at test time (ICML 2025)](https://icml.cc/virtual/2025/poster/46204); [Uncertainty-aware multi-objective RL-guided 3D diffusion (2510.21153)](https://arxiv.org/html/2510.21153).

**Most on-point recent steering work (2025–2026)**
- [Steering Generative Models for Protein Design — survey (Stocco, Garibbo, **Ferruz**, 2511.21476)](https://arxiv.org/abs/2511.21476) — best taxonomy (weight-changing vs. model-fixed/inference-time steering). **Read first for framing.**
- [Proteína-Complexa — scaling atomistic binder design with generative pretraining + test-time compute (2603.27950, ICLR 2026 oral)](https://arxiv.org/html/2603.27950v1) · [code](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa) — inference-time reward steering (beam / Feynman–Kac / MCTS) of an all-atom binder generator (built on La-Proteina); ships the **Teddymer** synthetic complex dataset. _Single most on-point paper across all three of your threads._
- `[Sta25c]` **BoltzGen** (Stark & Jaakkola) — universal all-atom binder design steered at inference by a design-specification language (constraints, binding sites, SS, masks).
- [EmbedOpt — steer a frozen Protenix/AF3 by optimizing its conditional embedding (2602.05285)](https://arxiv.org/html/2602.05285v2) · [code](https://github.com/rs-station/embedopt) — novel "where to inject the steering signal."
- [Controllable protein design with particle-based Feynman–Kac steering of RFdiffusion (2511.09216)](https://arxiv.org/abs/2511.09216) · [code](https://github.com/ErikHartman/FK-RFdiffusion) — +89.5% binder designability via interface-ΔG steering. **Directly steers RFdiffusion.**
- `[Liu26v]` **TorchCraft** — unified binder design by inverting a frozen all-atom predictor (AF3 weights).
- [ProHiFlo — hierarchical flow matching with functional guidance (2606.11243)](https://arxiv.org/pdf/2606.11243) _(verify)_; [ZetaDial — inference-time net-charge steering (2609.05451)](https://arxiv.org/pdf/2609.05451) _(verify)_; [BindMecNet — multi-task affinity + interface interaction (2025)](https://www.biorxiv.org/content/10.64898/2025.12.29.696866.full.pdf) _(protein–ligand; verify title)_.

---

## 4. Atom-level masked / denoising self-supervised pretraining

**Masked-atom / masked-coordinate modeling on 3D structures**
- [Uni-Mol (ICLR 2023)](https://chemrxiv.org/engage/chemrxiv/article-details/6402990d37e01856dc1d1581) — SE(3) transformer, masked-atom prediction + coordinate denoising on 209M conformers + ~3M protein pockets. Canonical pocket-aware atom-level pretraining.
- [Uni-Mol2 at scale (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/53923bb44655a7defb31c7744c01b62b-Paper-Conference.pdf) — 15% atom-token masking + coordinate denoising.
- [SpaceFormer — masked 3D-space occupancy/atom/position (2503.10489)](https://arxiv.org/pdf/2503.10489); [Localized Geometric Generation (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.09.10.612249v1.full); [Masking-design study (2512.07064)](https://arxiv.org/pdf/2512.07064); [3D-GSRD selective re-mask (2510.16780)](https://arxiv.org/html/2510.16780v2); [Equivariant Masked Position Prediction (2502.08209)](https://arxiv.org/pdf/2502.08209).

**Denoising ≈ learning an implicit force field / interaction energy** (the theory behind the "is this how atoms interact?" objective)
- Zaidi et al., "Pre-training via Denoising" (ICLR 2023) — foundational: coordinate-noise prediction ≈ learning the score of the equilibrium distribution (a force field). _Referenced ubiquitously; confirm direct link._
- [Frad — fractional denoising (ICML 2023, 2307.10683)](https://arxiv.org/pdf/2307.10683); [SliDe — sliced denoising (ICLR 2024)](https://openreview.net/pdf?id=liKkG1zcWq); [AniDS — anisotropic noise (2510.22123)](https://arxiv.org/html/2510.22123); [Denoise pretraining on nonequilibrium molecules → transferable potentials (2303.02216)](https://arxiv.org/pdf/2303.02216); [global-structure denoising (2506.13174)](https://arxiv.org/pdf/2506.13174); [MolSpectra (2502.16284)](https://arxiv.org/pdf/2502.16284).

**SSL specifically for interactions / interfaces at atomic resolution**
- `[Jin23b]` **DSMBind** — SE(3) denoising score-matching for unsupervised binding energy (see §2). Closest match to the proposed module.
- [BindNet — protein–ligand binding representation learning (2311.16160)](https://arxiv.org/pdf/2311.16160) — atomic pairwise distance-map prediction + masked ligand reconstruction. Closest fit for atom-level masked *interaction* pretraining.
- [SableBind (BMC Bioinformatics 2025)](https://link.springer.com/article/10.1186/s12859-025-06064-w); [Span-mask on 3D protein chains (2402.01481)](https://arxiv.org/pdf/2402.01481); [MpbPPI — multi-task PPI pretraining for ΔΔG (PMC10516393)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10516393/); GeoPPI (side-chain perturbation reconstruction; _link unconfirmed_); [Joint masked reconstruction + contrastive for PPI (2503.04650)](https://arxiv.org/pdf/2503.04650); [PLM-interact (sequence-level, bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.11.05.622169.full.pdf).

---

## 5. All-atom interaction modeling architectures & learned-potential scorers

**Atom-level / cross-biomolecule encoders**
- [Void-X — generative void-filling of masked interface atoms (PNAS 2026)](https://www.pnas.org/doi/10.1073/pnas.2607035123) — 172M params on ~8M PDB atomic clusters; 78.3% intra-chain / 68.2% inter-chain atomic-cluster accuracy. **Closest existing analog to masked atom–atom interface modeling.**
- `[Abr24]` AlphaFold3 atom-attention / Atom Transformer; [Protenix AF3 reproduction (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.01.08.631967v1.full.pdf) — best open source for atom-encoder implementation detail.
- [Pearl — all-atom structure foundation model (2510.24670)](https://arxiv.org/html/2510.24670v2); [IntFold — controllable AF3-style model (2507.02025)](https://arxiv.org/pdf/2507.02025); [ATMOS — AF3-style atom transformer, init from Protenix Pairformer (2603.17633)](https://arxiv.org/pdf/2603.17633) _(verify)_; [EPT — equivariant transformer, block/atom-level denoising (Nat Commun 2026)](https://www.nature.com/articles/s41467-026-69185-7); [PRIME — physics-informed multiscale equivariant (2605.01625)](https://arxiv.org/pdf/2605.01625) _(verify)_; [ProteinWorkshop benchmark (ICLR 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213157/).
- [FoldBench — benchmark of AF3/Protenix/Chai-1/Boltz on interfaces (Nat Commun 2025)](https://www.nature.com/articles/s41467-025-67127-3); [Latent-X — atom-level binder+target co-generation (2507.19375)](https://arxiv.org/abs/2507.19375); [ODesign — world model for biomolecular interaction design (2510.22304)](https://arxiv.org/pdf/2510.22304); [SeedProteo — all-atom binder design (2512.24192)](https://arxiv.org/pdf/2512.24192).

**Equivariant / geometric interaction models**
- [EquiCPI — SE(3)-equivariant compound–protein interaction (JCIM 2025)](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00773); [VN-EGNN — virtual-node equivariant binding-site ID (J Cheminform 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12837241); [SE(3)-equivariant ternary-complex prediction (2502.18875)](https://arxiv.org/pdf/2502.18875); [EquiBind (ICML 2022)](https://arxiv.org/pdf/2202.05146); [DiffBindFR (2311.15201)](https://arxiv.org/pdf/2311.15201).

**Learned interatomic potentials / physical-plausibility scorers**
- [Representing local protein environments with ML force fields (2505.23354, ICLR 2026)](https://arxiv.org/pdf/2505.23354) — MLFF (MACE/AIMNet) latent embeddings of a residue + atoms within 5 Å as transferable atom-level descriptors, zero-shot transfer. **Strong fit for a reusable atom-level encoder.**
- [MACE-OFF (2312.15211)](https://arxiv.org/pdf/2312.15211); [Allegro at biomolecular scale (2304.10061)](https://arxiv.org/pdf/2304.10061); [universal MLIPs w/ long-range terms in biomolecular sims (2508.10841)](https://arxiv.org/pdf/2508.10841); [MLIP-in-biomolecular-modeling review (Biophys Rev 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12847590/); [OMol25 dataset+models (2505.08762)](https://arxiv.org/pdf/2505.08762).

---

## 6. Training-data generation & augmentation (incl. engine-generated complexes)

**Decoys / hard negatives for interface scoring**
- [EuDockScore / -Ab (Bioinformatics 2024)](https://academic.oup.com/bioinformatics/article/40/11/btae636/7833366); [GDockScore (bioRxiv 2022)](https://www.biorxiv.org/content/10.1101/2022.12.02.518908.full.pdf); [LambdaDockScore / LambdaLoss ranking (2610.00191)](https://arxiv.org/html/2610.00191v1) _(verify)_; [DeepCoy property-matched decoys (Bioinformatics 2021)](https://academic.oup.com/bioinformatics/article/37/15/2134/6126797); [DUD-E hidden-bias caution (PLOS ONE 2019)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220113); [UPNA-PPI topology-driven ~3M hard negatives (Bioinformatics 2025)](https://academic.oup.com/bioinformatics/article/41/5/btaf148/8107765).

**Engine-generated complexes → train a module** (the Rosetta / Protenix-v2 / OpenDDE paradigm)
- **Rosetta:** [DeepAccNet (Nat Commun 2021, **Baker lab**)](https://www.nature.com/articles/s41467-021-21511-x) — trained on ~1M Rosetta decoys to predict per-residue accuracy, then steers Rosetta refinement. **Canonical "engine generates decoys → train a scoring/accuracy head."** Also [decoy-ranking via XGBoost (BMC Bioinformatics 2020)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-3523-9); [NanoBinder (J Cheminform 2025)](https://link.springer.com/article/10.1186/s13321-025-01040-1).
- **Protenix:** [Protenix v1 (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.01.08.631967v1.full) uses an MGnify distillation set; [**Protenix-v2** — prediction + target-conditioned design (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.04.10.717613v1.full) uses **training-free guidance** at sampling (not a trained head); [**AlphaRank** — trains a lightweight head on **Protenix PairFormer** single/pair representations for affinity ranking (bioRxiv 2025)](https://www.biorxiv.org/content/10.1101/2025.05.27.656341v1.full). **AlphaRank is the clearest "read the AF3-class engine's internal reps → train a scoring head."**
- **OpenDDE = Open Drug Discovery Engine** (Aureka Research): [arXiv 2607.03787](https://arxiv.org/abs/2607.03787) · [code](https://github.com/aurekaresearch/OpenDDE) — ~655M-param open all-atom co-folding+design engine; downstream modules (design, affinity, active learning) explicitly **future work**. It is an engine, not a dataset. Nearby data effort: [OpenBind](https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/).
- **Self-distillation:** `[Pas25e]` Boltz-2 (AF2 monomer + high-confidence Boltz-1 complex predictions as added training data); [RoseTTAFold2 — +1.2M AF2 distillation (bioRxiv 2023)](https://www.biorxiv.org/content/10.1101/2023.05.24.542179.full.pdf); [SimpleFold — iterated self-distillation (Apple, 2509.18480)](https://arxiv.org/pdf/2509.18480); [AFDistill (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-00436-1).
- **Training directly on model-generated complexes:** [Hsu et al. — scorer trained on Boltz-1x co-folded complexes (JCIM 2025, 2507.07882)](https://arxiv.org/abs/2507.07882) — **key caveat: augmentation helps only when generated-structure quality is high**; [IARA — GNN trained entirely on BindCraft-on-RFdiffusion trajectories (bioRxiv 2026)](https://www.biorxiv.org/content/10.64898/2026.04.16.718848v1.full); [Backenköhler et al. — E(3)-GNN trained on template-docked kinase–ligand complexes (JCIM 2024)](https://univerlag.sulb.uni-saarland.de/handle/20.500.11880/38269); [Mac1 557-ligand prospective co-folding caveat (bioRxiv 2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12776374/).

**MD / physics augmentation & corruption-based augmentation**
- [MISATO — ~20k PL complexes + ~170 µs MD + QM (Nat Comp Sci 2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11136668/); [PLAS-5k MM/PBSA (Sci Data 2022)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9451116/); [PLACER conformational ensembles (PNAS 2025, **Anishchenko/Baker**)](https://www.pnas.org/doi/10.1073/pnas.2427161122); [ensemble generation w/ physical feedback (ICML 2025, 2505.24203)](https://arxiv.org/pdf/2505.24203); [Manifold Restore Mixing augmentations (2606.22307)](https://arxiv.org/pdf/2606.22307); [diffused-decoy refinement pretraining (2302.10888)](https://arxiv.org/pdf/2302.10888); [to-augment-or-not (2510.01349)](https://arxiv.org/pdf/2510.01349).

---

## 7. Baker-lab & ecosystem — recent sweep (newest first)

Theme tags: **(a)** steering/guidance · **(b)** atom-level interaction · **(c)** complex training data. _Several entries verified via secondary sources only; flags preserved._

**2026**
- `[Liu26v]` / [TorchCraft — invert a frozen all-atom predictor for binder design (2609.19770)](https://arxiv.org/abs/2609.19770) **(a)**. _NOT Baker (Changping Lab); on-theme._
- Designed **proteases** that cleave disease proteins (TDP-43, amyloid-β) — Baker lab/IPD, reported ~Oct 14 2026, Nature per [unite.ai](https://www.unite.ai/baker-lab-designs-ai-proteases-that-cleave-disease-linked-proteins/) / [Cornell NE-CAT](https://necat.chem.cornell.edu/2026/10/15/ai-used-by-baker-lab-to-design-proteases/) **(b)**. _Primary DOI unconfirmed; RFdiffusion/RFD3 atom-level active-site design._
- [MSA Pairformer — PPI interface contacts from a monomer-trained PLM (Cell, Aug 2026, **Ovchinnikov**)](https://www.sciencedirect.com/science/article/pii/S009286742600749X) **(b)**.
- [Proteína-Complexa (2603.27950, ICLR 2026 oral)](https://arxiv.org/html/2603.27950v1) **(a)(b)(c)** — _NVIDIA+Mila; Ovchinnikov-linked via La-Proteina; the single most on-point paper for all three threads; ships the Teddymer synthetic complex dataset._
- [The past, present and future of de novo protein design — review (**Baker** corresp., Nature Apr 2026)](https://www.nature.com/articles/s41586-026-10328-7) — "how to design" largely solved; open question is "what to design."

**2025**
- [RFdiffusion3 (RFD3) — all-atom biomolecular interaction design (bioRxiv Sep 2025, Butcher…Bennett, Krishna, **Baker**)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12458353/) **(b)(c)** — drops RFD2's hybrid residue-atom scheme; ~10× faster; AME (atomic-motif-enzyme) 41-case benchmark. **Your base model.**
- [RFdiffusion2 — atom-level active-site scaffolding (bioRxiv Apr 2025 → Nat Methods Dec 2025; Ahern, Yim, …, Krishna, **Baker**)](https://www.biorxiv.org/content/10.1101/2025.04.09.648075v1) **(b)**.
- [LigandMPNN — atomic-context-conditioned sequence design (Nat Methods Apr 2025, **Dauparas**…**Baker**)](https://www.nature.com/articles/s41592-025-02626-1) **(b)**. _(FAMPNN "full-atom MPNN" referenced in ProteinDJ but primary paper unverified.)_
- [PLACER — protein–small-molecule conformational ensembles (PNAS Nov 2025, **Anishchenko**…**Baker**)](https://www.pnas.org/doi/10.1073/pnas.2427161122) **(b)**.
- `[Pac25]` / [BindCraft (Nature Aug 2025; Pacesa…**Ovchinnikov**, **Correia**)](https://www.nature.com/articles/s41586-025-09429-6) **(a)**.
- [Designed pMHC-I-binding proteins (Science Jul 2025, Baker lab)](https://www.bakerlab.org/wp-content/uploads/2025/08/science.adv0185.pdf); [de novo GPCR antibodies via test-time compute (bioRxiv May 2025)](https://www.biorxiv.org/content/10.1101/2025.05.28.656709) _(Baker authorship to confirm)_.
- [La-Proteina — partially-latent flow matching (2507.09466, **Ovchinnikov** co-author)](https://arxiv.org/html/2507.09466v2) **(b)**; [ProteinDJ pipeline (bioRxiv Sep 2025)](https://www.biorxiv.org/content/10.1101/2025.09.24.678028v2.full).
- Possu Huang lab: Protpardelle-1c (all-atom conditional generation, MotifBench 4.97→28.16 with sidechain conditioning); Caliby (ensemble-conditioned sequence design); DiffTopo. _URLs to confirm on bioRxiv._

**2024 (anchors)**
- [RoseTTAFold All-Atom (Science 2024, Krishna…**Baker**)](https://www.science.org/doi/10.1126/science.adl2528) **(b)** — the atom-level predictor RFD3/TorchCraft-type inversion builds on.
- `[Bae21]` RoseTTAFold 3-track (Science 2021, Baek & Baker); `[Wat23]` RFdiffusion (Nature 2023); `[Ben22]` Improving de novo binder design with deep learning (Bennett…Baker, Nat Commun 2022); `[Cho25h]` BoltzDesign1 (Cho & **Ovchinnikov**); `[Dau22]` ProteinMPNN.

_Per-author latest (confirmed): Baker → RFD3 + proteases + Nature review; Ovchinnikov → MSA Pairformer (Cell 8/2026); Dauparas → LigandMPNN; Ahern → RFdiffusion2; Anishchenko → PLACER. No 2025–26 RoseTTAFold successor from Baek/DiMaio surfaced (latest = RFAA 2024); Cao/Coventry — no confirmed recent preprint (manual check advised)._

---

## 8. Datasets (for either module)

- [PINDER — ~2.3M (up to ~25M augmented) dimeric PPI systems, holo/apo/predicted, leakage-free splits (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.07.17.603980v1) — the protein–protein corpus.
- [PLINDER — ~449k PL systems, leakage-aware splits (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.07.17.603955.full.pdf) — the protein–ligand corpus.
- [SAIR — ~5.24M PL structures w/ activity (NeurIPS 2025 ML4PS)](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_170.pdf); [PoseX docking benchmark (2505.01700)](https://arxiv.org/pdf/2505.01700); [PPInterface PP interface structures (Biophys J 2025)](https://www.cell.com/biophysj/fulltext/S0006-3495(24)01915-5); [PSBench complex-model accuracy labels (2505.22674)](https://arxiv.org/pdf/2505.22674); [DataSAIL leakage-aware splitting (Nat Commun 2025)](https://www.nature.com/articles/s41467-025-67495-w); [OpenProteinSet — >16M MSAs (NeurIPS 2023)](https://arxiv.org/abs/2308.05326v1); [SPICE2 QM dataset](https://zenodo.org/records/10835749); [AI3 PL affinity dataset (AWS)](https://registry.opendata.aws/ai3/); Uni-Mol corpus (209M conformers, ~3M pockets); **Teddymer** (synthetic domain–domain complexes, in Proteína-Complexa §7).
- _Gap: dedicated protein–nucleic-acid synthetic/augmented interaction data did not surface — needs a targeted follow-up._

---

## 9. Gaps & how they map to the proposed module

Both independent scouts converged on two **open niches** this project could claim:

1. **No published single multi-task head** jointly predicts {binding, interface quality, affinity, stability, developability} for protein–protein complexes **and uses it to steer generation.** Closest precedents are LaMBO-2/NOS (affinity+developability, sequence space) and BindMecNet (affinity+interface, protein–ligand). → a **multi-property steering head on RFD3's pair/latent representation** (à la PreFold-dG / AlphaRank reading internal reps, driven like LaMBO-2 or Feynman–Kac steering) is genuinely novel.
2. **No SSL objective targets atomic-resolution PPI interface contacts specifically.** The nearest works are protein–ligand (BindNet, atomic distance-map + masked-ligand) and interface-geometry reconstruction (Void-X masked interface atoms, MpbPPI, DSMBind denoising-energy). → an **atom-wise masked / denoising "is-this-a-valid-interaction" module** pretrained across proteins+ligands+nucleic acids, then transferred to all-atom PPI, is open.
3. **No Baker-lab learned steering head exists** — the ecosystem steers RFdiffusion by *inference-time search* (Feynman–Kac, test-time compute, Proteína-Complexa) or by *inverting a frozen predictor* (TorchCraft, BoltzDesign1), not by a trained property head feeding guidance gradients. That is the lane this idea occupies.

**Suggested first reads, by thread**
- Steering head: Steering survey (2511.21476) → LaMBO-2/NOS → AlphaRank → PreFold-dG → FK-RFdiffusion → Proteína-Complexa.
- Atom-level masked module: DSMBind `[Jin23b]` → BindNet → Void-X → Zaidi denoising → Uni-Mol2 → local-environment MLFF (2505.23354).
- Data generation: DeepAccNet → AlphaRank (Protenix reps) → Boltz-2 self-distillation `[Pas25e]` → IARA → Hsu et al. (JCIM 2025, quality caveat) → PINDER/PLINDER.

---

## 10. Live Undermind deep searches (ranked, with PDFs in the Undermind workspace)

- Confidence heads — structure prediction & binder ranking (139 papers): https://app.undermind.ai/projects/b8b4f51e-1128-409d-8fd0-2da3d1dc4ee7?path=/confidence%20heads%20protein%20structure%20prediction%20binder%20ranking
- Modules modeling target↔de novo binder PPI scoring (241 papers): https://app.undermind.ai/projects/b8b4f51e-1128-409d-8fd0-2da3d1dc4ee7?path=/modules%20modeling%20target%20de%20novo%20binder%20protein%20protein%20interaction%20scoring
