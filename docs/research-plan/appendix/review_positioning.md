# Red-team review: novelty, scoop, hypotheses, scope, venue

Role: skeptical ICML/ICLR/NeurIPS area chair at the bio x AI boundary. Prepared 2026-11-19.
Inputs: evidence_base.md, technical_plan.md, deepsearch_drug_plm.md, deepsearch_singlecell.md.
Scope: positioning/novelty only. Domain-citation accuracy is being verified by two parallel agents;
I take the reports' factual claims as given and attack the *framing*. Deadlines marked UNVERIFIED
are not confirmable from here and must be checked on the venue sites before anyone relies on them.
Spot-check performed: no public BSF-on-bio application found as of today (Goodfire's Evo 2 work is
BatchTopK SAE on layer 26; the "phylogeny manifold" is a blog observation, not a BSF paper).

================================================================================
# 1. NOVELTY STRESS-TEST
================================================================================

## 1a. Per-gap verdicts (evidence_base Part 4)

Verdict key: OPEN = genuinely unoccupied and the project can claim it; PARTIAL = being
filled, project must differentiate and cannot claim it clean; CLOSED/TABLE-STAKES = done
often enough that running it is a requirement, not a contribution.

1. **AlphaGenome/Borzoi/Enformer have ~no SAE/circuit interp.** PARTIAL (over-stated as
   written). FALSE that they have "no interp": CREME/SEAM (Koo), Enformation Theory (Rob25),
   DeepSCENIC (Par26e, CRISPRi-validated), Tu26 modality gap, Lia26 in-silico ablation all
   exist. What is OPEN is *SAE/BSF dictionary learning* on these embeddings specifically
   (singlecell gap 8 confirms). The project can claim "first manifold-aware dictionary on
   AlphaGenome," not "first interpretability of AlphaGenome." Narrow the claim or a reviewer
   kills it in one sentence.

2. **No crosscoder/model-diffing on any bio model.** OPEN but NOT this project's claim (the
   plan does no crosscoders). Note: Jerpelea's BSF follow-up extends blocks to crosscoders,
   so Columbia could occupy this adjacent lane. Leave it; don't gesture at it.

3. **Circular validation (features scored vs training annotations).** PARTIAL. The framework-
   level fix is already claimed by Orlov (review), Shahsavand (ground-truth eval), Ali
   (composition-matched). The project's *specific* instantiation -- deterministic physics
   ground truth for continuous/periodic geometric variables from post-cutoff experimental
   structures -- is still a defensible, concrete contribution. This is one of the two real
   openings. Claim the instantiation, cite the framework owners, do not claim the idea.

4. **Random-init / composition-matched baselines rarely run.** CLOSED / TABLE-STAKES. Now
   standard: Ami25b, Kar26, Decode-gLM, Ali, Korznikov, Heap. The project MUST run them and
   gets zero novelty credit. A submission without them is desk-rejectable at this boundary.

5. **Causal (not correlational) validation is thin.** PARTIAL, closing fast. Dose-response
   steering already exists: Ped26 (concentration-dependent drug features), Jed26b (dose-
   dependent helix/coil), Mai26 (steered A1408G), Ken26e (differentiation direction), Ali
   (decoder ablation), Sri26 (intervention selectivity ladder). "Causal dose-response" per se
   is NOT novel. The OPEN sliver is dose-response *along a within-manifold continuous/periodic
   coordinate with wrap-around* for circular variables -- nobody has done manifold-coordinate
   traversal with a toroidal wrap-around test. That is the distinctive causal move.

6. **"Linear decodability != causal steerability" shown once (Boltz-1).** CLOSED as a finding.
   Now replicated: Jed26b (beta-strand decodes F1 0.82, won't steer), Haq25 (antibody TopK not
   causal), Sri26, Ami25b. It is an established constraint the project must respect, not claim.

7. **Feature reproducibility/stability barely measured for bio SAEs.** PARTIAL -> closing.
   Shahsavand (consensus over seeds, 27-46% reproduce), Song (ACL 2026 feature consistency),
   Jed26 (cross-seed + intruder detection on Boltz-1), Paulo & Belrose. Project must report
   >=5-seed stability + Tournament Top-K; it is table stakes, lightly differentiating only
   because BSF seed-stability on bio is unmeasured (Jerpelea shows BSF is seed-fragile).

8. **No standardized interp benchmark for bio models (bio-SAEBench).** OPEN. This is the
   project's STRONGEST framing. Shahsavand and the position papers (Zhou, Orlov, Nag26b) push
   toward it but no accepted benchmark exists. A released ground-truth continuous/periodic-
   variable benchmark for featurizers is a genuine, citable artifact that survives a negative
   BSF result. Lead with this.

9. **ESM3 multimodal generative steering via dictionary learning underexplored.** PARTIAL.
   Steenwyk x2 (geometric atlas; modality fusion + attribution patching + steering on ESM3
   1.4B), Wang (ESM3+ESM2 differentiable steering, +VEP on 114 DMS). ESM3 is actively worked.
   The project overlaps Steenwyk directly and must differentiate (he already does geometry +
   steering on ESM3 48 layers). This is a real collision risk, under-weighted in the plan.

10. **Feature death downstream effect unquantified.** OPEN, not the project's claim. Ignore.

11. **RNA models very thin.** OPEN, out of scope. Ignore.

12. **Genuine novel-biology discovery is rare.** OPEN, and the project SIDESTEPS rather than
    fills it. The project deliberately re-finds *known* geometric variables (that is the point:
    exact ground truth). Vulnerability: a reviewer will say "you recover dihedrals/SASA you
    already knew -- where is the discovery?" Pre-empt by framing the deliverable as a
    *benchmark + a causal test SAEs fail*, not as biological discovery. If you pitch it as
    discovery you lose; if you pitch it as measurement rigor you win.

## 1b. Single most defensible novel contribution

> **The first ground-truth benchmark showing that scalar SAE latents fragment known
> continuous/periodic geometric manifolds (phi/psi torus, helix phase, SASA, codon/exon phase)
> in biological foundation models, and that manifold-aware (block-sparse) dictionaries recover
> them in fewer units at matched capacity -- validated causally by dose-response traversal
> along the recovered coordinate, including the toroidal wrap-around test that a 1-D latent
> cannot pass.**

The load-bearing, least-scoopable atoms are: (i) **physically-computed geometric ground truth**
(not annotations), and (ii) the **periodic/toroidal manifold + wrap-around dose-response**. The
circular-variable angle is the single most distinctive thing in the whole plan: it is the one
place where "manifold vs scalar latent" is not a matter of degree but of topology (you cannot
tile a circle with a bounded number of monotone 1-D features without a seam), and it gives a
causal test (wrap-around) with no scalar-SAE analogue. Build the paper's spine around circular
variables; treat scalar variables (SASA) as the easy control.

## 1c. What is NOT novel (do not claim; cite instead)

- SAEs on ESM/Evo 2 (dozens; InterPLM, InterProt, Decode-gLM, Goodfire Evo 2).
- Geometric annotation of SAE features -- **Setlur et al. (2608.26419, Aug 2026)** already
  annotates ESM-2 SAE features with Ca-geometry, FDR-controlled, with contact-map ablation.
  This is the closest existing paper; a reviewer can frame the project as "Setlur + BSF."
- Random-init / composition-matched controls (standard).
- Dose-response / concentration-dependent steering (Ped26, Jed26b, Mai26).
- "Decodable != causal" (established).
- Manifold framing of bio reps (Goodfire phylogeny manifold; Candido/Rives world model;
  Bes25; Wang 2604.07117; Ken26m spectral geometry).
- BSF and Tournament Top-K (Goodfire own BSF; Jerpelea/Columbia own Tournament Top-K).
- "Jacobian on bio models" -- the NAME and the coevolution object are taken: categorical
  Jacobian (Zhang PNAS 2024), nucleotide dependency (Tomaz da Silva, Nat Genet 2025),
  Thorstenson (2606.21876). See section 3 on the collision.

================================================================================
# 2. SCOOP RISK
================================================================================

No direct scoop found through 2026-11-19 (confirmed by my spot-check: BSF-on-bio does not
exist publicly yet). Ranked by probability of publishing this first:

1. **Goodfire + Arc (HIGHEST).** They own BSF, own the deployed Evo 2 BatchTopK SAE (layer 26),
   and already published the "phylogeny manifold" framing of Evo 2. If they point BSF at Evo 2
   or any protein model with a ground-truth variable, they scoop the headline *verbatim*.
   Exposure by sub-claim: BSF-beats-SAE-on-bio = MAXIMAL; Evo 2 manifold existence = maximal;
   ESM3 protein leg = lower (they are genomics-leaning). Their Oct-Nov 2026 posts were
   unreadable to the survey agents -- this is the one live unknown that could already moot the
   project. **Mitigation: go protein-FIRST on ESM3-open (off their turf), lead with physical
   geometric ground truth + toroidal wrap-around dose-response (not their natural framing).**

2. **EvolutionaryScale (ESM3 team).** Candido/Rives already frame ESM reps as a concept-space
   "world model of protein biology" (6.8B-sequence map). If they add dictionary learning with
   geometric ground truth they take the protein leg. Exposure: H1 (ESM3 manifold existence) =
   high; the BSF-vs-SAE methodological comparison and the controls-first posture are not their
   style, and they do not own BSF. Medium-high overall.

3. **Oxford / Crook (Jedryszek, Maiwald; Decode-gLM, Boltz-1 trunk/diffusion, auto-interp).**
   Fast, prolific, rigorous; they already do steering + random-init + cross-seed + Meta-SAEs.
   Most likely to publish the "controls-first causal manifold steering on a bio model" paper
   that overlaps framings 2/3. Exposure: causal/steering/controls methodology and dose-response
   = high; BSF-specific manifold recovery = lower (they use BatchTopK/Meta-SAE, not BSF). A
   Meta-SAE could pre-empt the "manifold fragmentation" observation without naming it BSF.

4. **Stanford / Zou (InterPLM, Silberg, Simon feature-death/outliers).** Own ESM-2 SAE
   infrastructure + annotation-filling + discovery. Could bolt geometry ground truth onto
   InterPLM. Exposure: moderate -- annotation/discovery-leaning, less periodic/manifold. Simon's
   outlier/feature-death work overlaps the controls section only.

5. **Aghazadeh (GT; ProtoMech, ProGenMech, Tsui low-N).** Circuit tracing / cross-layer
   transcoders / steering-for-design. Exposure on the manifold-recovery claim = low. BUT their
   cross-layer transcoder IS a transport-like object -- they are the group most able to scoop
   the **Jacobian-transport half** (a transcoder already does layer-to-layer relevance). Another
   reason to drop that half (section 3/4).

**Closest items in the last ~3 months:**
- **Setlur et al. 2608.26419 (Aug 2026)** -- closest overall. Geometry-annotated ESM-2 SAE
  features + ablation. Must cite prominently and differentiate (scalar SAE + discrete geometric
  labels; NO manifold/circular/toroidal framing, NO capacity-matched BSF comparison, NO
  continuous dose-response).
- **Shahsavand et al. (2026-10-29)** -- occupies the "ground-truth SAE eval" framing with
  planted concepts + consensus-over-seeds. Overlaps framing 1. Differentiate: they do
  reproducibility and discrete concepts, not continuous/periodic geometric manifolds, not
  manifold-aware dictionaries, not causal traversal.
- **Jaf26 / Microsoft (2026-11-02)** -- attribution diagnostics; "conclusions depend on
  reference/baseline/null." Raises the controls bar you must clear.
- **Goodfire Oct-Nov 2026 blog/posts (UNREADABLE)** -- the genuine live risk. Have a human open
  goodfire.ai/research before submission.

================================================================================
# 3. HYPOTHESIS + METHOD CRITIQUE (H1-H5)
================================================================================

## Weakest hypothesis: H5 (Jacobian transport). Recommend DROP from the main paper.

**H1 (existence).** Falsifiable IF topology is committed per variable *before* results.
As written ("pre-register torus vs basin clusters") it risks being unfalsifiable -- choosing
torus-or-basins after seeing data is a free pass. FIX: commit specific Betti numbers per
variable up front (phi/psi -> expect basin-clusters on T^2, NOT a uniform torus, because
Ramachandran is non-uniform; helix phase -> circle (1,1,0); codon phase -> circle), and treat
a different realized topology as refutation of that prediction. Biggest hidden landmine: the
cleanest periodic targets (helix phase = residue mod 3.6; codon phase = pos mod 3) may be
near-trivially recoverable from position/local-sequence. The local-sequence-window baseline
(control 7) may KILL the periodic claim. Run that control FIRST, before any BSF training; if a
3-residue window predicts helix phase, the "manifold" is a positional artifact. Also: on ESM3,
structure-conditioned input makes dihedral decoding trivial -- sequence-only is mandatory, not
optional. (The plan says this; elevate it to a gating pre-condition.)

**H2 (fragmentation).** Best-specified hypothesis. Capacity-matched control present (restricted
R^2 at n=d vs supervised rank-d ceiling). Falsifiable. Caveat: it is Bhalla's "do SAEs capture
concept manifolds" ported to bio -- incremental unless the *periodicity* makes it qualitatively
different (it does: a scalar latent cannot cover a circle without a seam). Make that topological
argument explicit so H2 is not read as a domain transfer of a known result.

**H3 (BSF recovery win).** The headline methodological claim and the one most likely to come
back NEGATIVE. Jerpelea's critique bites directly: BSF manifold recovery is seed-dependent, not
fixed by capacity, and degrades as k rises. Deeper risk (technical_plan decision 3): group_size
>= 2 has strictly more capacity per active unit, so "fewer units" is partly definitional. The
matched-bits/MDL control is the correct answer and IS in the plan (good) -- but enforce that the
win must survive matched-bits, not only matched-units, and must use an identical MDL code for
BSF and SAE. If the advantage only appears at matched-units, it is an artifact and reviewers
will say so. The plan's D&B fallback (publish the benchmark even if BSF loses) is the right
insurance; keep it explicit.

**H4 (causal manifold traversal).** Strongest causal hypothesis; good matched-norm controls
(random in-subspace, random orthogonal, supervised-probe direction). Two fixes:
 (a) **Missing readout-independence control.** For ESM3 the "output" must be the model's own
 generative/structure-token head, NOT a probe for the same variable -- traversing a dihedral
 coordinate and reading a dihedral probe is circular. State that the causal readout is
 probe-independent.
 (b) "Predicted amount" is fragile: direction may transfer while magnitude does not (Jed26b:
 helix steers, beta-strand does not). Keep the "only magnitude matters -> refute" clause; also
 pre-register that direction-correct-but-magnitude-wrong is a PARTIAL pass, not a full one.
 The toroidal wrap-around version of H4 is the crown jewel -- foreground it.

**H5 (Jacobian transport ranks blocks by output relevance). DROP from main; appendix on Evo 2
only, or spin to a follow-up.** Reasons:
 1. **Ostrow survives against it, not for it.** Mean Jacobian need not resemble any individual
    Jacobian; prominence is unrelated to variance explained. On a *bidirectional* ESM3 over a
    heterogeneous protein corpus, mean-vs-per-input alignment is likely LOW -- which is exactly
    H5's own refutation clause. H5 probably fails its own test on the primary model.
 2. **Ill-defined where it matters.** The causal "current-and-future target" sum is undefined on
    bidirectional ESM3; the square J_l is ill-posed across the AlphaGenome U-Net neck; only Evo 2
    (causal, 4 tokens) is a clean fit -- and that is the model being de-emphasized. The 4-token
    vocab also makes the readout thin.
 3. **Naming collision.** "Jacobian on bio models" already means categorical Jacobian (Zhang)
    and nucleotide dependency (Tomaz da Silva, Nat Genet). Reviewers at this boundary will
    conflate them. The "layer-to-output Jacobian transport" rename + comparison table is correct
    mitigation but is pure tax for a secondary contribution.
 4. **Weak baseline bar.** H5 as written only must beat activation-norm ranking. It must also
    beat attribution patching, which already exists in bio (Mohanty, Steenwyk) and is the
    obvious stronger baseline. Against that bar H5 is not clearly a contribution.
 5. **Dilution.** It is a separable second contribution that doubles the attack surface of a
    9-page paper and is owned, in spirit, by Aghazadeh's transcoders. Cut it.

Net: H1 needs topology pre-commitment + position-confound gating; H2 needs the topological
argument made explicit; H3 must survive matched-bits (keep D&B insurance); H4 needs
readout-independence; H5 should leave the main paper.

================================================================================
# 4. SCOPE + VENUE
================================================================================

## Recommended scope
- **Methods: BSF only.** Drop the Jacobian-transport half from the contribution (appendix on
  Evo 2 only, or a separate follow-up). This single cut removes the Ostrow liability, the
  naming collision, the ill-definedness on ESM3/AlphaGenome, and the Aghazadeh overlap, and
  lets the paper be one tight story.
- **Models: ESM3-open primary and load-bearing + ONE genomic generality panel.** Build the
  paper so it stands on ESM3 alone (richest ground-truth set: phi/psi torus, chi1, SASA,
  contact number). The genomic model is a one-model "it is not just proteins" panel using
  codon phase (the cleanest periodic target), NOT a second full study.
  - Choice of genomic model: **AlphaGenome v1** if you want maximum novelty and minimum scoop
    exposure (weights available, NC-research OK, largest open SAE gap, OFF Goodfire's turf) --
    accept that you train your own SAE baseline and restrict any Jacobian to the transformer
    tower. **Evo 2** if compute/time is tight (reuse Goodfire corpus + deployed SAE for a clean
    BSF-vs-real-SAE head-to-head on codon phase) -- but that is Goodfire's home turf and the
    highest scoop risk. Given the scoop analysis, **lean AlphaGenome** for the main submission;
    use Evo 2 only for the cheap go/no-go pilot.
  - Drop Bonbon entirely (sequence-only, closed-weight, no pair/diffusion -- cannot be hooked,
    wrong manifolds). Already the plan's conclusion; endorse it. Do NOT substitute a co-folding
    model (Boltz-2) -- PairSAE/Par26/Jed26b already crowd that space and it reopens the pocket-
    geometry leg you do not have time for.
- **Protein-only vs +genomic:** add the one genomic panel for generality, but do not let a
  weak genomic result sink the paper. ESM3 carries it.

## Venue + timeline (ALL DEADLINES UNVERIFIED -- confirm on icml.cc / iclr.cc / neurips.cc)
- The evidence_base's dates are UNVERIFIED and I cannot confirm them from here. Treat the
  following as UNVERIFIED and have a human check them:
  - ICML 2027 main: full paper reported ~22 Jan 2027, abstract ~14-16 Jan -- UNVERIFIED.
  - ICLR 2027 main: CLOSED. ICLR 2027 workshops ~1 Feb 2027 -- UNVERIFIED (accepted-workshop
    list reportedly due 29 Nov 2026).
  - NeurIPS 2027 abstract ~mid-May 2027 -- UNVERIFIED.
- **Realistic plan of record:**
  1. **ICLR 2027 workshop (~Feb, UNVERIFIED) -- ESM3-only, H1/H2 + partial H3 + first H4
     dose-response.** Priority stake + feedback vehicle. This is the honest near-term output.
  2. **NeurIPS 2027 main track (~May abstract, UNVERIFIED) as the real target**, with the
     **NeurIPS 2027 Datasets & Benchmarks track as the floor** if BSF does not beat SAEs (the
     released ground-truth benchmark carries a D&B paper regardless of H3's sign).
- **ICML 2027 main (late-Jan full paper) is NOT a realistic primary.** From 2026-11-19 that is
  ~9 weeks for: ESM3 sweep + genomic panel + >=5 seeds + Tournament Top-K + causal dose-response
  + all mandatory controls, at the rigor this boundary now demands. Do not make it the plan of
  record; it is aspirational-only.

================================================================================
# 5. BOTTOM LINE
================================================================================

- **Main-track acceptance odds as currently planned (4 models in play, both methods, Jacobian
  included, aiming ICML main in Jan): ~15%.** Consistent with the area-chair agent's 15-20%.
  Drivers: over-broad for 9 pages; Jacobian half fragile and collision-prone; BSF core maximally
  scoop-exposed to Goodfire; closest prior art (Setlur, Shahsavand, Steenwyk) not yet
  differentiated; timeline forces a rushed submission.
- **Two changes that raise odds the most:**
  1. **Drop the Jacobian-transport half** (appendix/Evo 2 only). Halves the attack surface and
     removes the Ostrow + naming-collision + ill-definedness liabilities in one move.
  2. **Reframe as a ground-truth manifold-recovery BENCHMARK whose signature result is the
     toroidal wrap-around dose-response that scalar SAEs structurally cannot pass** -- pre-
     register per-variable topology and run the position/local-sequence confound control FIRST.
     This differentiates from Setlur/Shahsavand/PairSAE, survives a negative BSF result (D&B
     fallback), and moves the paper from "another SAE-on-bio" to "the measurement + the causal
     test SAEs fail."
  (Close third: protein-first/ESM3-heavy + AlphaGenome as the genomic panel, to dodge Goodfire's
  Evo 2 turf.)
- With those changes + ESM3-primary + one genomic panel: **~35-45% at NeurIPS main, near-certain
  at D&B or a workshop.** Matches the area-chair's upgraded estimate.
- **Recommendation: RESHAPE (not go, not drop).** The core -- physically-computed geometric
  ground truth + periodic/toroidal manifold recovery + within-manifold dose-response, protein-
  first -- is real, defensible, and currently unoccupied. But the plan as scoped is over-broad,
  scoop-exposed, and partly pre-empted. Reshape to: ESM3-primary + one genomic generality panel,
  BSF-only, benchmark + toroidal-causal framing, ICLR workshop stake (~Feb, UNVERIFIED),
  NeurIPS 2027 main / D&B as the target. Before submission, have a human open goodfire.ai/research
  to check the one live scoop unknown.
