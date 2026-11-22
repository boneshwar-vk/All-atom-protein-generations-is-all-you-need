# Proteins, Co-folding, Drug Discovery & Methods — verification + formalisation pass

Reviewer: senior mech-interp reviewer (bio × AI). Date: 2026-11-19.
Scope: protein LMs, protein structure/co-folding models, drug-discovery/molecular models,
and the two methods (J++ / Jacobian lens; BSF Block-Sparse Featurizer).
NOT genomics or single-cell (owned by another agent).

This is a VERIFICATION + FORMALISATION pass over two untrusted drafts:
`evidence_base.md` and `deepsearch_drug_plm.md`. Every "CONFIRMED" below was checked
against a primary source through the Undermind MCP (workspace
`33899782-4f3e-48f1-9e4a-a3b6ac8a7af7`): either a metadata resolution to the real
paper record, a semantic-search hit on title+authors, or a full-text PDF read.
Confidence tags: **CONFIRMED-PDF** (read full text) / **CONFIRMED-REC** (resolved to the
real paper record / abstract, details not re-read) / **UNVERIFIED** (could not resolve).

Hard constraint honoured: this is a description of published interpretability work only.
No how-to for steering models toward hazardous outputs is given anywhere below.

---

## 0. TL;DR of the verification (what changed)

- Method identity for the two *published* methods is **solid**: Jacobian lens = Gurnee et al.
  arXiv 2607.15495 (CONFIRMED-REC), BSF = Fel et al. arXiv 2606.25234 (CONFIRMED-REC).
- **J++ Lens and Bonbon are the two unverifiable load-bearing items.** Neither is indexed in
  Undermind; arxiv.org is blocked in this session's network, so their abstracts could not be
  fetched. All J++-specific and all Bonbon-specific claims remain UNVERIFIED. The prior
  agents' own "UNVERIFIED" flags on these were correct and are retained.
- **The "Ostrow critiques the BASE J-lens, not J++" claim is CONFIRMED by reading the PDF.**
  Ostrow et al. never mention J++, Jacobian Filtering, or fitted expert weights; they analyse
  the base J-lens and propose a different extension (ω-J-lens). Recovered its missing arXiv id:
  **2610.31955**.
- **One real contradiction between the two files:** arXiv **2608.27754** is confirmed as
  Jedryszek & Crook, *"Efficient Auto-Interpretability of AI Models in Biology"* — NOT the
  "cross-seed stability + intruder detection" paper that `evidence_base.md` §3.2 attaches to
  that number. See §2(B).
- **One overclaim corrected:** PairSAE predicts **Boltz-2's own predicted affinity** (self-
  consistency, test R²=0.528), not experimental binding affinity. Its absence of causal /
  random-init / leakage controls is CONFIRMED-PDF — the "open lane" framing holds.
- All other top protein / co-folding / drug claims checked out, several with tightened numbers.

---

## 1. Claim-by-claim verification (the ~20 load-bearing items)

### Methods

| # | Claim as written in drafts | Verdict | Evidence / fix |
|---|---|---|---|
| 1 | Jacobian lens (J-lens) = Gurnee et al., *"Verbalizable Representations Form a Global Workspace in Language Models,"* arXiv 2607.15495 | **CONFIRMED-REC** | Resolves to [Gur26], DOI 10.48550/arXiv.2607.15495. Title/number/author match. |
| 2 | BSF = *"Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds,"* Fel, Kowal, Jacobs, Hazra, Bhalla…, arXiv 2606.25234 | **CONFIRMED-REC** | Resolves to [Fel26c], DOI 10.48550/arXiv.2606.25234. |
| 3 | J++ Lens = Ayonrinde & Lindsey (Anthropic), LessWrong Oct 2026 + github safety-research/jpp_lens; "expert weights fitted with supervision"; Oct 2026 Circuits Update | **UNVERIFIED** | Not indexed in Undermind (it is a blog post + code repo, not an arXiv paper); arxiv.org blocked. Could not independently confirm authorship, the supervised-fitting step, or the Circuits-Update claim. Prior agent's UNVERIFIED flag retained. |
| 4 | Ostrow, Qian, Liu, Fiete, *"A Mathematical Theory of the Jacobian Lens"* critiques the **base J-lens, not J++** | **CONFIRMED-PDF** | [Ost26], arXiv **2610.31955** (number recovered; drafts left it blank). PDF: J++/Jacobian-Filtering/expert-weights never appear; it proposes a distinct ω-J-lens. Core math CONFIRMED: ‖J_H v‖ = ρ_c(v)·ρ_τ(v)·a(v) (context coherence × lag coherence × amplification); mean Jacobian's singular vectors ≠ members'; prominence ⟂ variance explained (Cor. 1.1). Validation is toy (HMM Bayes filter + 2-layer transformers); it *does* inspect real Gemma/Llama/Qwen Jacobians, but only for the individual-vs-mean point — minor nuance to the drafts' "toy only." |
| 5 | BSF critique = Jerpelea & Ananthram, *"A Deeper Analysis of Block-Sparse Featurizers,"* arXiv 2608.27515 | **CONFIRMED-REC** | Resolves to [Jer26], DOI 10.48550/arXiv.2608.27515. |
| 6 | Bhalla et al., *"Do Sparse Autoencoders Capture Concept Manifolds?"* (BSF motivation) arXiv 2604.28119 | **CONFIRMED-REC** | Resolves to [Bha26], DOI 10.48550/arXiv.2604.28119. |

### Bonbon (the user's requested 4th model)

| # | Claim | Verdict | Evidence / fix |
|---|---|---|---|
| 7 | Bonbon = arXiv 2611.14504, *"Emergent Biological Capabilities in a Foundation Model for Molecular Interactions,"* single author A. Mashiach; 1.6B params / 3.2T tokens; decoder + RoPE, pre-LN; paired protein–ligand sequences; **no 3D / no pair rep / no diffusion**; MoA + binding-site + bond-type zero-shot; dose-response hit rate 61.5% (arXiv)/70% (NeurIPS); no public weights/code | **UNVERIFIED** | Not in Undermind corpus; semantic search for the exact abstract returned unrelated DTI/co-folding models (not Bonbon). Direct `arXiv:2611.14504` metadata lookup failed. arxiv.org fails DNS in this session → abstract could not be fetched. **Every architecture/capability figure remains unverified, including the 61.5% vs 70% hit-rate conflict.** The DROP recommendation does not depend on these numbers and stands (see §4). |

### Protein LMs — SAE / dictionary learning

| # | Claim | Verdict | Evidence / fix |
|---|---|---|---|
| 8 | InterPLM — Simon & Zou, Nature Methods 2025, DOI 10.1038/s41592-025-02836-7, SAEs on ESM-2 | **CONFIRMED-REC** | [Sim24], DOI matches; Nature Methods record, 168 cit. |
| 9 | "InterProt" — Adams … AlQuraishi, 10.1101/2025.02.06.636901, TopK SAEs on ESM-2 650M | **CONFIRMED-REC w/ NAMING FIX** | DOI resolves to [Ada25] *"From Mechanistic Interpretability to Mechanistic Biology…"* — that is the paper's real title. "InterProt" is the AlQuraishi lab's SAE **viewer/tool name**, not the paper title. `deepsearch_drug_plm.md` labels it correctly as [Ada25]; `evidence_base.md` §3.1 conflates tool-name and paper. Same DOI, so it is one paper, double-listed across files. |
| 10 | Gujral … Berger, PNAS 2025, 10.1073/pnas.2506316122, SAEs + transcoders | **CONFIRMED-REC** | [Guj25], DOI matches; PNAS record, 77 cit. |
| 11 | Setlur, Mihajlovic, D. Lee — geometric annotations of ESM-2 SAE features, arXiv 2608.26419 (reviewer's "prior work"/MUST-cite pick) | **CONFIRMED-REC** | [Set26] *"Interpreting Latent Protein Language Model Features with Geometric Annotations."* Authors confirmed. This is the single closest "geometry-on-protein-SAE-features" prior art; see §3. |

### Protein structure / co-folding

| # | Claim | Verdict | Evidence / fix |
|---|---|---|---|
| 12 | PairSAE — Migliorini … Viessmann, arXiv 2606.27440, Boltz-2 pairformer; N-mode SVD + shared SAE; features align UniProt, "predict affinity"; **NO causal/seed/leakage controls** | **CONFIRMED-PDF w/ OVERCLAIM FIX** | [Mig26], DOI 10.48550/arXiv.2606.27440. PDF: Boltz-2 pairformer, layers 33 & 64, recycle 3, PLINDER. Features predict **Boltz-2's own predicted affinity** (LASSO, test R²=0.528, 291 features) — *not* experimental affinity. Controls audit CONFIRMED: **no** feature interventions/steering (future work), **no** random-init/cross-seed stability, and **only partial** holdouts (no systematic PLINDER↔Posebusters leakage check). The "open lane" claim is correct. |
| 13 | Jedryszek … Crook, arXiv 2608.11475, Boltz-1 trunk+diffusion; LINEAR DECODABILITY ≠ CAUSAL (β-strand F1 0.82, no steering) | **CONFIRMED-PDF** | [Jed26b]. PDF: β-strand F1 **0.82** in both trunk and diffusion; strand direction predictive (0.78 held-out, prec ≥0.93) but steering effect **−0.001, 95% CI [−0.003,+0.001]** (null). Helix **+0.039**, coil **+0.061** at dose k=16, dose-dependent vs matched-norm random controls. Interventions on the single rep *s* (not pair *z*). Releases trunk+diffusion SAEs, per-residue activations, code. The flagship "decodable ≠ causal" datapoint is solid. |
| 14 | Lu, Brinkmann … Bau, Wendler, arXiv 2602.06020, ESMFold/OpenFold/Boltz-1 trunks; shared 2-stage folding; causal steering; cross-model linear interchange | **CONFIRMED-REC** | [Lu26], arXiv 2602.06020. Title/model list/author match. |
| 15 | Feldman & Skolnick, *"A Shortcut to Structure in AlphaFold 3,"* arXiv 2610.08937; MSA shortcut; 40-residue restore recovers topology; fold recoverable by block ~9/48; cross-protein alignment 0.44→0.72 | **CONFIRMED-PDF** | [Fel26b]. PDF: AF3, causal interventions + probes across all 48 Pairformer blocks. 40 anchor residues recover **median 71%** of the lost TM-score gap (long-range-contact retention 28%→73%); linear probe reaches **median TM 0.51 by block 9** (52% > 0.5) — *probe*-recoverable, not diffusion-output; donor-protein alignment raises median TM **0.442→0.717** (0.568→0.762 post-cutoff), register-dependent. Numbers match the drafts. |
| 16 | Masters, Mahmoud, Lill — co-folding models don't learn physics, Nature Communications 2025, 10.1038/s41467-025-63947-5 | **CONFIRMED-REC** | [Mas25], DOI matches; the field's anchor physics-vs-shortcut paper. |

### Drug / causal / methodology

| # | Claim | Verdict | Evidence / fix |
|---|---|---|---|
| 17 | Srivastava et al., *"Probing isn't Causality in Molecular Models,"* bioRxiv 2026, 10.64898/2026.10.05.756781 (Uni-Mol/MolCLR; evidence ladder) | **CONFIRMED-REC** | [Sri26], DOI matches. Supports the "decodable ≠ causal" thread on the molecular side. |
| 18 | Haque … Deane (OPIG), antibody-LM SAEs, arXiv 2512.05794; TopK vs Ordered SAEs; high corr ≠ causal control | **CONFIRMED-REC** | [Haq25], DOI 10.48550/arXiv.2512.05794. |
| 19 | Zhang … Ovchinnikov, "categorical Jacobian," PNAS 2024 — ESM-2 stores Potts-like coevolutionary pairwise statistics | **CONFIRMED-REC w/ DOI NOTE** | [Zha24]. DOI on record is the **bioRxiv preprint** 10.1101/2024.01.30.577970, not a PNAS DOI; both drafts cite the preprint DOI while calling it "PNAS 2024." Cosmetic; the paper is real and correctly described. Key naming-collision prior art for any "Jacobian on proteins" framing. |
| 20 | arXiv **2608.27754** = Jedryszek & Crook "cross-seed stability + intruder detection; stability selects structure over function" (`evidence_base.md` §3.2) | **CORRECTED** | 2608.27754 resolves to [Jed26] *"Efficient Auto-Interpretability of AI Models in Biology."* `deepsearch_drug_plm.md` §B6 labels 2608.27754 correctly; `evidence_base.md` mis-describes it. Whether a *separate* Jedryszek "cross-seed stability/intruder" paper exists is **UNVERIFIED** — do not cite that description against 2608.27754. |
| 21 | Open protein–ligand substitute "Pou25b" Pourmirzaei et al. (seq+SMILES binding-site) — flagged UNVERIFIED in drafts | **CONFIRMED-REC** | [Pou25b] *"Zero-Shot Protein–Ligand Binding Site Prediction from Protein Sequence and SMILES,"* bioRxiv, DOI 10.1101/2025.09.28.679103. Real; usable as a seq+SMILES substitute leg. |

**Not re-resolved this pass (lower load-bearing, left as drafts have them):** Hu et al. ESM-C 6B
SAE (arXiv 2606.12209), Thorstenson attention-head contact readout (arXiv 2606.21876),
InterProt/ProtSAE/MotifAE secondary entries. No reason to doubt them; simply not re-verified.

---

## 2. Duplicates, contradictions, overclaims (within scope)

**(A) Duplicate — same paper, two names/cite-conventions.**
`evidence_base.md` §3.1 "InterProt (Adams…AlQuraishi, 10.1101/2025.02.06.636901)" ==
`deepsearch_drug_plm.md` [Ada25] "From Mechanistic Interpretability to Mechanistic Biology."
One paper. Use [Ada25] as the canonical title; keep "InterProt" only as the tool/viewer name.

**(B) Contradiction — arXiv 2608.27754.** The two files attach *different papers* to this
number. Confirmed record: *"Efficient Auto-Interpretability of AI Models in Biology"* (Jed26).
`evidence_base.md`'s "cross-seed stability + intruder detection" description is misattributed.
Action: trust `deepsearch_drug_plm.md` here; treat the cross-seed-stability paper as
unconfirmed until a real identifier is found.

**(C) Overclaim — PairSAE "predicts affinity."** Both files can be read as "predicts binding
affinity." Primary text: predicts **Boltz-2's own predicted affinity values** (a self-
consistency / model-distillation result), R²=0.528. State it as such in related work, or a
reviewer will flag it.

**(D) Unverifiable foundations flagged as if near-settled.** Bonbon's architecture (decoder+
RoPE / 3.2T tokens / objective / tokenization) and J++'s supervised expert-weight fitting are
each sourced only to a blog/snippet/closed record. They read in the drafts as near-fact with a
trailing "(UNVERIFIED)". Keep them explicitly unverified; do not build related-work prose that
*asserts* them. The Bonbon 61.5%-vs-70% hit-rate conflict is itself a sign the figures are
second-hand.

**(E) Minor — categorical-Jacobian DOI.** Cited as "PNAS 2024" but with the bioRxiv-preprint
DOI (10.1101/2024.01.30.577970). Harmless; fix the DOI if the PNAS version is cited as venue.

**(F) Minor — Ostrow "toy-model only."** True for the *selection theory*, but the paper also
inspects real Gemma/Llama/Qwen Jacobians for the individual-vs-mean claim. Phrase as "theory
validated on toy models; real-model analysis limited to the mean-Jacobian pathology."

No fabricated DOIs were found in scope. No DOI was invented here; unresolved items left blank.

---

## 3. Closest prior art the BSF + Jacobian-transport project must differentiate from

Ranked by how directly each threatens novelty.

1. **[Set26] Setlur, Mihajlovic, Lee — geometric annotations of ESM-2 SAE features** (arXiv
   2608.26419). The nearest neighbour: assigns geometric/structural meaning to protein-SAE
   latents with FDR control, ablations shift contact maps. Differentiator required: it labels
   **scalar** latents post hoc; it does *not* test low-dim **manifold** recovery, capacity-
   matched block-vs-scalar comparison, random-init controls, or causal dose-response. MUST cite.
2. **[Mig26] PairSAE** (Boltz-2 pair reps, 2606.27440). Closest *co-folding* interpretability.
   Differentiate on: manifold/continuous-variable recovery, capacity-matched featurizer
   comparison, and the three controls it lacks (causal intervention, seed stability, leakage).
3. **[Jed26b] Jedryszek & Crook, Boltz-1 trunk→diffusion** (2608.11475). The causal-protocol
   bar: it already shows decodable ≠ causal with matched-norm random controls and dose curves.
   Any manifold-steering claim must clear this bar and cite it as the protocol antecedent.
4. **BSF method lineage — [Fel26c] (2606.25234) + [Jer26] critique (2608.27515) + [Bha26]
   manifold-SAE motivation (2604.28119).** The "first BSF on a biological model / on an LM"
   angle is only novel *relative to vision*; frame as a port, and adopt Jer26's Tournament
   Top-K as the fair baseline (Jer26 shows vanilla BSF splits manifolds seed-dependently).
5. **Folding-trunk mechanism cluster — [Lu26] (two-stage trunk), [Liu26] (single=semantics /
   pair=geometry, ESMFold), Park et al. Boltz-1 intra/inter-chain (bioRxiv 2026-09).** These
   own "where geometry is computed"; a manifold study must add the *representational-geometry*
   layer they don't, not re-derive their stage story.
6. **Naming-collision / Jacobian-on-proteins prior art — [Zha24] categorical Jacobian +
   Thorstenson attention-head contact readout (2606.21876).** A "Jacobian lens on proteins"
   contribution must include an explicit table separating **layer-to-output Jacobian transport**
   (the J-lens construct) from the **categorical Jacobian** (input-token coevolution). Reuse the
   drafts' naming guidance ("layer-to-output Jacobian transport").
7. **Cross-layer causal protein work — [Tsu26] ProtoMech transcoders, [Nai25] SAE-latent
   activation patching, [Moh26] causal features for VEP.** These are the causal-protein
   baselines; differentiate on *continuous-manifold* causality, not discrete feature circuits.

Scoop watch (confirmed as real, unresolved as threats): Goodfire owns BSF *and* runs SAEs on
Evo 2 — they are the most likely party to put BSF on a bio model first. [Fel26c]/[Jer26]/[Bha26]
are all Goodfire-adjacent. No published BSF-on-bio or Jacobian-transport-on-bio paper was found
through 2026-11-19.

---

## 4. Genuinely open gaps in this scope (confidence)

1. **BSF has never been applied to any biological model, nor to any language model** (only
   InceptionV1 / DINOv3 / SDXL in the original). Jacobian-transport (J-/J++-lens) has never been
   applied to a bio model. The *combination* exists only as a third-party GitHub issue title.
   **HIGH.**
2. **Manifold / continuous-variable recovery with capacity-matched (block-vs-scalar) controls on
   bio models is unoccupied.** [Set26] annotates scalar latents; representation-shape work
   (Beshkov, Sulu) and equivariant-disentanglement (Steier) touch edges only. **HIGH.**
3. **No featurizer study on co-folding pair reps carries all three of {causal intervention,
   random-init control, leakage-controlled split}.** PairSAE has none; Jed26b has causal+random
   but on single rep, not a featurizer-recovery benchmark. **HIGH.**
4. **Interpretability-guided design with prospective wet-lab validation: none found** across
   protein/antibody/ligand models (design papers black-box the model; interp papers validate in
   silico). **HIGH.** (Out of this project's likely scope but worth one sentence.)
5. **Protein–ligand (ligand-geometry) manifold interpretability is thin.** Bonbon is sequence-
   only + closed → unusable; open substitutes with actual pair geometry are Boltz-2 (already has
   PairSAE), Chai/RoseTTAFold-All-Atom; seq+SMILES leg = [Pou25b]. No manifold/BSF study of a
   ligand-conditioned pair rep exists. **MED-HIGH.**
6. **Decodable ≠ causal is shown on exactly one co-folding model (Boltz-1, Jed26b) and one
   molecular pair (Sri26).** Generalising it with a *continuous-manifold* dose-response (incl.
   wrap-around for circular variables) is open. **MED-HIGH.**
7. **Jacobian-transport as an output-relevance ranker on bio models is unexamined**; the only
   bio "Jacobian" work is categorical-Jacobian coevolution ([Zha24], Thorstenson), which is a
   different object (input→logit, not layer→output). **HIGH.**

---

## 5. Net recommendation to the planning stage (scope-limited)

- Keep ESM3-open (sequence model) + one co-folding model as the two legs; the co-folding leg is
  where the differentiated prior art is densest (PairSAE / Jed26b / Lu26 / Park) and therefore
  where novelty must be argued hardest.
- **DROP Bonbon** — unverifiable and architecturally unsuitable (sequence-only, no pair rep /
  diffusion, closed weights; the ligand-geometry manifolds the thesis wants are not present).
  This verdict is independent of the unverifiable 1.6B/3.2T/hit-rate figures.
- **Lead with BSF-on-bio under Jer26's Tournament Top-K + capacity-matched controls** as the
  robust-to-negative-result core; treat Jacobian-transport as the exploratory half and name it
  "layer-to-output Jacobian transport" with the Zha24/Thorstenson disambiguation table.
- MUST-cite-and-differentiate set: [Set26], [Mig26], [Jed26b], [Fel26c], [Jer26], [Bha26],
  [Lu26], [Zha24]. MUST-fix before related-work: the InterProt/Ada25 naming, the 2608.27754
  attribution, and the PairSAE "predicted-affinity" wording.
- **Blocked verifications to hand to a human/next pass:** open arxiv.org in the session network
  to confirm (a) Bonbon arXiv 2611.14504 abstract + architecture, (b) the J++ LessWrong post +
  github supervised-fitting claim. Both are currently unresolvable here.
