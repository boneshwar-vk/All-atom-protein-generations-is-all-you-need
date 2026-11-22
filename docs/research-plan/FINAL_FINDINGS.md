# FINAL FINDINGS — Interpretability of biological foundation models, and a novelty assessment for a BSF/Jacobian project
Consolidated 2026-11-19. Supersedes evidence_base.md where they conflict (all three review agents' corrections applied).

Source files in this scratchpad:
- evidence_base.md — raw consolidation of the first four reports
- technical_plan.md — per-model experiment plan (hook sites, configs, compute)
- deepsearch_drug_plm.md — drug-discovery + protein-LM reading list (163+72 papers screened)
- deepsearch_singlecell.md — genomic + single-cell reading list (102+92 papers screened)
- review_proteins.md / review_genomics.md / review_positioning.md — the three verification/red-team passes
Undermind workspace (all papers + the 4 deep searches): https://app.undermind.ai/projects/33899782-4f3e-48f1-9e4a-a3b6ac8a7af7

"UNVERIFIED" = not confirmed against a primary source. arxiv.org / biorxiv.org / huggingface.co / goodfire.ai
were network-blocked this session; bioRxiv read via Undermind PDFs; some licence/date facts from web snippets.

================================================================================
## 0. BOTTOM LINE
================================================================================
- The two methods are real: **J++ Lens** (Anthropic, github.com/safety-research/jpp_lens, Oct 2026) built on
  the **Jacobian lens** (Gurnee et al. arXiv 2607.15495), and **BSF = Block-Sparse Featurizer**
  (Goodfire, arXiv 2606.25234). Neither has ever been applied to a biological model. [J++ identity UNVERIFIED
  via Undermind — it is a LessWrong post + repo, not indexed; BSF & Jacobian-lens identities CONFIRMED.]
- Verdict on the project as the user framed it (BSF + Jacobian on ESM3 + AlphaGenome(2) + Evo 2 + Bonbon):
  **RESHAPE.** The core idea is real and currently unoccupied; the scope is too broad, scoop-exposed, and the
  Jacobian half is the weakest link.
- **Bonbon must be dropped** (sequence-only, closed-weight, and its very architecture figures are UNVERIFIED).
- **AlphaGenome 2 is unusable** (released 2026-11-11, API-only, no weights).
- Single most defensible contribution: **a ground-truth benchmark showing that scalar SAE latents fragment
  known continuous/periodic geometric manifolds (backbone dihedrals, helix phase, codon phase, SASA) that
  block-sparse dictionaries recover at matched capacity — with a toroidal wrap-around dose-response as the
  causal test a 1-D latent structurally cannot pass.** Protein-first (ESM3-open), one genomic generality panel.
- Acceptance odds: ~15% main-track as planned; ~35–45% at a NeurIPS-class main track after two changes
  (drop the Jacobian half to an appendix; reframe as the benchmark); near-certain at a Datasets & Benchmarks
  or workshop track. ICML 2027 (full paper ~late Jan, UNVERIFIED) is not realistic in ~9 weeks.

================================================================================
## 1. THE TWO METHODS (verified)
================================================================================
### J++ Lens / Jacobian lens
- Jacobian lens: J_l = corpus-average of d h_final / d h_l, read out through the unembedding; ~100 prompts
  saturate; needs full weights + autograd; assumes single residual stream + causal decoder + token unembed.
- J++ adds k-means "expert" Jacobians (claimed supervised weight-fitting — UNVERIFIED, post not opened),
  LRP backward rules, and readout filtering; stays one linear map per layer.
- **CORRECTION:** Ostrow et al. ("A Mathematical Theory of the Jacobian Lens", arXiv **2610.31955**, Oct 2026)
  critiques the BASE J-lens, NOT J++ (J++/expert-weights never appear in it); it proposes a different ω-J-lens.
  Its load-bearing point stands: the MEAN Jacobian need not resemble any individual one, and selected
  directions can explain arbitrarily little activation variance.
- Applied to a bio model: NONE. Closest is GDTR (Cho et al., Evo 2 7B settling-depth lens) — logit-lens-style.

### BSF = Block-Sparse Featurizer
- SAE variant: dictionary = n_groups blocks x group_size (default 3) directions; sparse ACROSS blocks, dense
  WITHIN. Three variants (Vanilla / Grassmannian / Group-Lasso). Needs only cached activations; group_size=1
  reduces to a TopK SAE. Recovered concepts typically 2–4 dimensional. Identity CONFIRMED (Fel et al. 2606.25234).
- Critique CONFIRMED: Jerpelea & Ananthram (arXiv 2608.27515) — manifold splitting, seed dependence; fix =
  Tournament Top-K. Within-block basis identified only up to rotation.
- Applied to a bio model: NONE. Goodfire's own Evo 2 work is a plain BatchTopK SAE (see §3). The
  "phylogeny manifold" is a Goodfire blog post, not a BSF paper (web-checked).

================================================================================
## 2. MODEL AVAILABILITY (corrected)
================================================================================
- ESM-2 (to 15B): open, permissive. ESM3-open (esm3-sm-open-v1, 1.4B, 48 layers): open weights, NON-COMMERCIAL.
  ESM C (300M/600M open; 6B API): non-commercial caveat.
- Evo 2 (1B/7B/40B): open, Apache-2.0 (NVIDIA BioNeMo packaging differs). Public Goodfire BatchTopK SAE baseline.
- Nucleotide Transformer, DNABERT-2, HyenaDNA, Caduceus, Enformer, Borzoi: open.
- AlphaGenome v1 (~450M): weights GATED + NON-COMMERCIAL; code Apache-2.0; docs CC-BY. ("~Jan 2026" date UNVERIFIED.)
- **AlphaGenome 2 (2026-11-11): API-ONLY, no weights, cell-state conditioned (<=256 cells/call). NOT usable for
  mechanistic interp.** ("107M profiles" UNVERIFIED.)
- scGPT / Geneformer / scFoundation / Cell2Sentence: open. Boltz-1/2, ESMFold, RFdiffusion, RoseTTAFold3: open.
- **Bonbon (arXiv 2611.14504, "Emergent Biological Capabilities in a Foundation Model for Molecular
  Interactions"): UNVERIFIED end to end** — not in Undermind, direct lookup failed, arXiv blocked. Reported
  sequence-only, 1.6B, decoder+RoPE, NO structure/pair/diffusion head, NO public weights, single author. Every
  architecture/capability figure (1.6B, 3.2T tokens, 61.5% vs 70% hit-rate) is unconfirmed. **DROP.** If a
  protein–ligand leg is wanted, the review advises NOT substituting Boltz-2 (already crowded by PairSAE/Par26/
  Jed26b); a seq+SMILES binding-site model exists as a substitute (Pou25b, Pourmirzaei, 10.1101/2025.09.28.679103)
  but adds scope.

================================================================================
## 3. STATE OF THE FIELD (verified reading list — headline items; full tables in the review_* files)
================================================================================
### Proteins — SAEs / dictionary learning
- InterPLM (Simon & Zou) Nature Methods 2025, ESM-2, L1 SAEs — the foundational paper. 10.1038/s41592-025-02836-7.
- "From Mechanistic Interpretability to Mechanistic Biology" (Adams… AlQuraishi) bioRxiv 2025-02 / ICML 2025,
  ESM-2 650M TopK SAEs. 10.1101/2025.02.06.636901. [NOTE: "InterProt" is the viewer tool, not the paper title —
  evidence_base listed them as two papers; they are one.]
- Gujral… Berger, PNAS 2025-08, ESM-2 650M, TopK SAE + transcoders. 10.1073/pnas.2506316122.
- Reticular (Parsan et al.) arXiv 2503.08764, Matryoshka SAEs on ESM2-3B/ESMFold.
- **Setlur et al. [Set26] — geometric annotations (Cα pseudo-torsion/curvature/contacts) on ESM-2 SAE features,
  FDR-controlled, ablation shifts contacts. THE closest prior art; MUST cite and differentiate.** (arXiv 2608.26419.)
### Proteins — circuits / causal / Jacobian
- ProtoMech (Tsui… Aghazadeh) arXiv 2602.12026 — cross-layer transcoders on ESM-2. ProGenMech (2606.16044) on ProGen3.
- Nainani… Green bioRxiv 2025-08 — activation patching in SAE latent space (motif→domain gating).
- Mohanty… Green bioRxiv 2026-09 — causal feature-level zero-shot VEP.
- **Zhang… Ovchinnikov, "categorical Jacobian" (ESM-2, coevolution/contacts)** — the input→output-logit Jacobian
  that OWNS the "Jacobian on proteins" name. Published; evidence_base cited only the bioRxiv preprint.
- Thorstenson arXiv 2606.21876 — attention-head contact readout; quantifies pretraining-overlap inflation.
### Proteins — structure / co-folding (the crowded 2026 frontier; closest architectural prior art)
- Lu… Bau/Wendler arXiv 2602.06020 — ESMFold/OpenFold/Boltz-1 trunks, two-stage folding, causal steering.
- **PairSAE (Migliorini… Viessmann) arXiv 2606.27440 — SAE on Boltz-2 pair tensors. CORRECTION: it predicts
  Boltz-2's OWN predicted affinity (test R²=0.528), not experimental affinity; no causal/seed/leakage controls
  (confirmed from PDF) → the causal co-folding lane is open.**
- **Jedryszek… Crook arXiv 2608.11475 — Boltz-1 trunk+diffusion; DECODABLE ≠ CAUSAL (β-strand probe F1 0.82 but
  strand-steering effect ≈ −0.001 vs helix +0.039/coil +0.061). The empirical backbone of the "controls" case.**
  [CORRECTION: arXiv 2608.27754 is Jedryszek & Crook "Efficient Auto-Interpretability of AI Models in Biology";
  evidence_base misattributed a "cross-seed stability + intruder detection" description to it — drop that description.]
- Fel… AF3 MSA-shortcut [Fel26b] — 40-residue restore recovers 71% of the TM gap; localizes where the shortcut lives.
- Steenwyk x2 (ESM3 geometry atlas + ESM3/ESM-2 SAE steering, structure head L0H7) — an UNDER-WEIGHTED collision
  with an ESM3-centric plan; must be differentiated.
- VFUSE (Yu & Olson) — first SAE on an all-atom diffusion model (RoseTTAFold3/RFDiffusion3); biosecurity framing.
### Genomics
- **Evo 2 (Brixi… Hie, Arc + Goodfire) Nature 2026-03 — BatchTopK SAE on LAYER 26 of Evo 2 7B** [CORRECTED from
  "layer 6"; the Orlov review's Table 3 has an internal error; Goodfire's repo is "Evo-2-Layer-26-Mixed"].
  32,768 features = UNVERIFIED. 10.1038/s41586-026-10176-5.
- Decode-gLM (Maiwald, Jedryszek… Crook) — NT 50M/500M, BatchTopK + Meta-SAEs; CMV-enhancer feature exposed
  training-DB contamination; steered A1408G resistance; random-init control. CONFIRMED in full. 10.1101/2025.10.31.685860.
- Ali arXiv 2607.19618 — NT + DNABERT-2; naive PWM validation confounded by GC/Alu (440/6576 spurious CTCF);
  composition-matched + ablation gives causally-clean CTCF/GATA1/REST. CONFIRMED in full.
- Tomaz da Silva nucleotide-dependency maps — **published Nature Genetics 2025 (10.1038/s41588-025-02347-3)**;
  found 4 validated E. coli RNA structures. [evidence_base cited only the preprint.]
- Mathur & Sachidanandam — Evo 2 VEP blind spots (flanking flips pathogenic tRNA calls).
- Position paper arXiv 2606.07607 — demands non-anecdotal interp evaluation in genomics.
### Single-cell
- Pedrocchi… Boeva (ETH) — scGPT/scFoundation/Geneformer BatchTopK SAEs; drug-feature steering (in-silico, not
  wet-lab); independent group. 10.1101/2025.10.22.681631.
- **Kendiukhov — ~17 single-author bio-FM-interp papers (2026)** [CORRECTED up from ~11]. The three central
  single-cell NEGATIVES all rest on this one author: only ~3/48 TFs show target-specific SAE responses vs
  CRISPRi; attention adds nothing over gene-level baselines; scGPT interventions AUROC≈0.50 vs real perturbation.
  Direction IS corroborated independently (Ahlmann-Eltze/EMBL Nat Methods; Csendes; PertEval; Qiu; Tiwari), but
  single-group replication of the mechanistic negatives is a real gap. (Genomic-SAE subfield is NOT one-author.)
- **DeepSCENIC [Par26e] — transfers Enformer/Borzoi INTERNAL embeddings into a CRISPRi-benchmarked causal GRN
  (dS-Enformer AUC 0.83); SOX10-KD → melanoma MEL/MES state shift. The strongest perturbation-validated
  sequence-model-internals result. MISSING from evidence_base — add it.**
### Cross-cutting evaluation / negatives
- **Korznikov et al. — CORRECTION: the "SAEs recover 9% at 71% EV / random baselines match" result is a SYNTHETIC
  toy, not bio and not even an LM; the "matching" baselines are constrained/partially-frozen, not random, on
  Gemma/Llama. Cite as a caution, not as "random beats SAE on real models".**
- Heap & Aitchison — automated interp metrics don't separate trained vs random transformers.
- Simon et al. arXiv 2605.31518 — feature death (>70% on AlphaFold3), fixed by mean-centering.
- Shahsavand et al. (Oct 2026) — ground-truth SAE eval on spatial/single-cell/LM; reproducibility ≠ meaning.
  A framing collision with contribution (1); must cite.
- Orlov et al. review (bioRxiv 2026-03) — "What Do Biological Foundation Models Compute?"; catalogs the
  circularity and null-model gaps. 10.64898/2026.03.04.709491.

================================================================================
## 4. OPEN-GAP LEDGER (red-team verdicts; 12 original + 2 new)
================================================================================
GENUINELY OPEN:
- #8 No standardized interpretability benchmark for bio models. HIGH. ← anchor the paper here.
- #3b The CONTINUOUS/periodic slice of validation (circular variables with exact geometric ground truth, not
  database annotations). The periodic/toroidal angle is the least-scoopable atom. HIGH.
PARTIAL / CLOSING:
- #1 manifold-geometry existence (Steenwyk, Bigot already frame axes/geometry).
- #3 annotation circularity (acknowledged widely; Ali/Decode-gLM already attack it).
- #5 scalable causal validation (DeepSCENIC, Mohanty advancing).
- #7 seed stability (Shahsavand, Jedryszek).
- #9 ESM3 multimodal steering (Steenwyk).
CLOSED / TABLE-STAKES (do them, don't claim them):
- #4 random-init baselines; #6 decodable ≠ causal (Jedryszek did it first).
NARROWED (evidence_base overstated it):
- #2 "no crosscoder on bio" — holds. #1-genomic "no SAE on Enformer/Borzoi/AlphaGenome" must be narrowed to
  "no SAE/dictionary-learning on their INTERNALS": DeepSCENIC (embedding transfer) and Enformation-Theory
  (embedding PCA) DO use internals, so "untouched / only ISM+attention" is wrong.
SIDE-STEPPED (a vulnerability, not a win):
- #12 novel-biology discovery — the plan validates against known geometry, so reviewers can call it "re-finding
  physics". Have a discovery angle ready (e.g. a feature with no annotation that predicts a held-out property).
NEW GAPS (from the single-cell deep search):
- Virtual-cell / perturbation-native models (STATE, CellFM, Tahoe-X1, scGPT+STATE) are essentially un-audited.
- SAE/circuit decomposition of the fate-BIFURCATION computation itself, tested against lineage-tracing
  (LARRY-style) ground truth, is essentially nonexistent.

================================================================================
## 5. RECOMMENDED PROJECT (reshaped)
================================================================================
CONTRIBUTION (one sentence): Using variables with exact geometric ground truth, show that biological foundation
models encode periodic/continuous structure as low-dimensional manifolds that scalar SAE latents fragment, that
block-sparse dictionaries recover them at matched capacity under random-init/leakage/seed controls, and that
moving along a recovered manifold's coordinate causally shifts model outputs — with the circular wrap-around
dose-response as the signature test a 1-D latent cannot pass.

SCOPE:
- Methods: **BSF only** in the main paper. Move the Jacobian-transport idea to an appendix / Evo-2-only
  exploration / follow-up (H5 is the weakest link — see §6).
- Models: **ESM3-open primary and load-bearing** + **one genomic generality panel — lean AlphaGenome v1**
  (most novel, off Goodfire's turf; you must train your own SAE baseline there). Use **Evo 2 only for the
  cheap pilot** (its public SAE is the ready baseline). **Drop Bonbon; do not substitute Boltz-2.**

GROUND-TRUTH VARIABLES (computed by code from experimental structures/annotations ABSENT from training):
- Proteins: φ/ψ (2-torus), ω, χ1, helix phase (index mod 3.6 within DSSP helices), relative SASA, contact number.
- DNA: codon phase (Z3), exon relative position, distance to splice site, GC, ~10.5 bp nucleosome phasing.

STAGED PILOTS WITH GO/NO-GO GATES:
1. Evo 2, codon phase: BSF vs the public SAE vs a group_size=1 control AND a local-sequence window baseline
   (run the positional/local-sequence confound FIRST — helix/codon phase may be trivially positional and could
   kill the periodic claim). GO if BSF recovers periodicity above all baselines.
2. ESM3-open, φ/ψ torus, one layer, ~5k proteins, SEQUENCE-ONLY input (structure input makes dihedrals trivial).
   GO if a block decodes the torus where <=d SAE latents fragment it.
3. Causal: within-block coordinate traversal with wrap-around dose-response (>=8 targets), matched-norm random-
   direction controls, readout = the model's OWN output head (not a same-variable probe). GO if the output
   follows the coordinate and returns to baseline at 2π.
4. Full ESM3 sweep (Grassmannian + Group-Lasso + Vanilla, >=5 seeds, Tournament Top-K) + AlphaGenome codon/exon.

COMPUTE (order-of-magnitude, UNVERIFIED): one 24 GB GPU → pilots 1–2 (~150–300 GPU-h); 1–2×80 GB → full ESM3 +
Evo 2 + AlphaGenome tower (~800–1500 GPU-h); small cluster → seed ensembles + all variants (~4–8k GPU-h).
BSF training is cheap (runs on cached activations); cost is dominated by activation caching + interventions.

================================================================================
## 6. HYPOTHESES (fixed after red-team) and RISKS
================================================================================
- H1 Existence — was unfalsifiable as written (torus-vs-basins decided post hoc). FIX: pre-commit per-variable
  Betti numbers; require trained−random-init circular-R² gap > 0.1; run the local-sequence confound control first.
- H2 Fragmentation — well-specified but is Bhalla-ported-to-bio; make the topological "you cannot tile a circle
  with intervals" argument explicit so it is more than a metric.
- H3 Recovery (BSF beats SAE) — the likely-NEGATIVE headline. Enforce the win at matched BITS/MDL, not just
  matched units. Keep the Datasets & Benchmarks fallback: if BSF does not beat SAE, the benchmark + the
  fragmentation result + the negative are still a paper.
- H4 Causal — needs a readout-INDEPENDENCE control (model's own output head, not a same-variable probe). The
  circular wrap-around dose-response is the crown jewel.
- H5 Jacobian transport — DROP from the main paper. It likely fails its own mean-vs-per-input (Ostrow) clause on
  bidirectional ESM3, is ill-defined on ESM3/AlphaGenome, is only clean on the de-emphasized Evo 2, collides
  with the categorical-Jacobian name, and must beat attribution patching (not just activation norm) to matter.

SCOOP RISK (ranked): (1) Goodfire+Arc — maximal on the "BSF beats SAE" headline (they own BSF AND the Evo 2 SAE);
(2) EvolutionaryScale on ESM3; (3) Oxford/Crook on causal-controls steering; (4) Stanford/Zou; (5) Aghazadeh
(could scoop the Jacobian half via transcoders). LIVE UNKNOWN: Goodfire's Oct–Nov 2026 posts were unreadable —
a human should check goodfire.ai/research before submission.

VENUE / TIMELINE (ALL DEADLINES UNVERIFIED — confirm on venue sites):
- ICLR 2027 workshop (~Feb) — an ESM3-only stake in the ground. Near-certain if the benchmark is clean.
- NeurIPS 2027 main (~May) — the realistic main-track target after the reshape; NeurIPS Datasets & Benchmarks
  is the floor.
- ICML 2027 main (full paper ~late Jan) — NOT realistic in ~9 weeks from 2026-11-19; aspirational only.

THE TWO CHANGES THAT RAISE ODDS MOST: (a) drop the Jacobian half to an appendix; (b) reframe the paper as the
ground-truth manifold-recovery benchmark whose signature is the toroidal wrap-around causal test.

================================================================================
## 7. OPEN ITEMS FOR A HUMAN / COMPLIANT-MODEL PASS
================================================================================
- Dual-use / biosecurity section: repeatedly cut by a safety classifier when an agent tried to write it. Needs a
  human or policy-compliant pass. General principle retained: restrict interventions to benign structural
  variables (dihedrals, phase, SASA); note that genomic/ligand steering is capability-adjacent.
- Open arxiv.org in the session network to finish verifying Bonbon (2611.14504) and the J++ post.
- Confirm the Evo 2 SAE feature count (32,768), AlphaGenome v1 release date / "107M profiles", and all venue
  deadlines from primary pages.
- Pre-submission: check goodfire.ai/research and the Cerebral Valley hackathon gallery (a reported J-lens-on-ESM1b
  entry, UNVERIFIED) for scoops.
