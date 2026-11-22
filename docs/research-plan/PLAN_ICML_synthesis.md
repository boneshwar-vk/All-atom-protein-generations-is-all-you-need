# ICML plan synthesis — three ideas, four expert passes
Consolidated 2026-11-19 from: plan_task1_geometry.md, plan_task2_crossmodel.md, plan_task3_steering.md,
plan_icml_novelty.md (and the earlier FINAL_FINDINGS.md evidence base). Deadlines UNVERIFIED — confirm on venue sites.

================================================================================
## THE VERDICT (all four agents agree)
================================================================================
- Task 1 (geometry-matched dictionaries) = the ICML bet, but ONLY if reframed from an application
  ("BSF with a different norm on a protein model") into a THEORY paper with a separation theorem.
- Task 2 (which model is better) = NOT standalone. Both sharp versions are already done on bio models
  (Steenwyk ESM-2 vs ESM3 SAE feature-sharing; Whitfield concept-recovery across 8 protein LMs). FOLD IN
  to Task 1 as a width/tokenisation-invariant cross-model corollary section (generality, not a leaderboard).
- Task 3 (localise + intermediate-timestep steering) = real but NOT ICML and switches model family
  (diffusion/co-folding, not ESM3/AlphaGenome). Sharpen into a (layer × denoising-timestep) decodable-vs-
  steerable study → NeurIPS 2027 / workshop, strong FOLLOW-UP to Task 1. "ipSAE" is an interface
  confidence score (Dunbrack), not an SAE — a target property, and a Goodhart trap if used as the primary one.
- Portfolio: Task 1 >> Task 3 > Task 2. Do ONE paper (Task 1). Do not combine all three.

================================================================================
## THE RECOMMENDED PAPER (Task 1, as theory)
================================================================================
### One-sentence contribution
Single-direction SAEs must pay a covering-number penalty in number of atoms to faithfully represent a
compact/periodic concept manifold, whereas a curvature-matched block dictionary represents it with constant
capacity and recovers it up to the manifold's isometry group; we give the separation theorem, a per-atom
curvature model-selection rule, and validate on synthetic manifolds and the Ramachandran (phi,psi) torus
of a protein language model.

### Why this clears the ICML bar (not an application paper)
(a) A separation THEOREM, not a method transfer — it answers exactly what Zah26 "The Geometric Wall"
    measured empirically (curvature drives SAE width-scaling) but did not prove, and builds the curvature-
    matched block that Jer26 "Deeper Analysis of BSF" named but did not construct.
(b) Generalises beyond biology — applies to any SAE pipeline with periodic/compact concepts (the LLM
    "days of week / months" circles of Engels et al. are the same phenomenon).
(c) Resolves the "more parameters vs the right geometry" confound (the comparison is at matched bits/MDL).
(d) Converts BSF's within-block rotation non-identifiability from a caveat into an optimality statement
    (recovery up to isometry), and ships an algorithm with a curvature-selection rule.

### The three theorems to prove (plausibly provable with standard tools; T3 hardest)
- T1 (lower bound): to reconstruct a planted compact/periodic manifold M embedded in activation space to
  L2 tolerance epsilon, a scalar/single-direction sparse dictionary needs Omega(covering-number(M, epsilon))
  atoms. CRITICAL: must be about faithful reconstruction of the CURVED embedding, not spanning its linear
  subspace (2 atoms trivially span a circle's plane — see Kill #1/#3).
- T2 (upper bound): a curvature-matched block (sphere/torus/Poincare of the right dimension) reconstructs M
  to the same epsilon with O(1) atoms — controls the capacity confound.
- T3 (identifiability): under a planted generative model, the block recovers M up to its isometry group
  (the rotation caveat becomes the recovery guarantee). State under explicit assumptions.
- Plus: a per-atom curvature model-selection rule (choose Euclidean / spherical / hyperbolic per block from
  data) — genuinely new vs mixed-curvature VAEs and Euclidean BSF.

### Experimental core (compresses because the contribution is a proof, not a model sweep)
1. Synthetic manifold zoo (reuse Jer26's prebuilt zoo): plant circle/torus/sphere/tree at realistic
   activation norm; show scalar-SAE atom count grows with covering number, curvature-matched block stays O(1),
   at matched BITS/MDL. This is the theorem's empirical face.
2. ONE biological vignette: the Ramachandran (phi,psi) torus in ESM3-open — a toroidal block code recovers it
   where scalar SAEs fragment and flat Euclidean BSF splits; SEQUENCE-ONLY input; wrap-around dose-response on
   the model's own output head as the causal test a 1-D latent structurally cannot pass.
3. Corollary (absorbs Task 2): the recovery metric is width/tokenisation-invariant → a small ESM-2 vs ESM3
   generality panel (is the torus shared or model-unique, and more causal in the multimodal model).
4. (Optional genomic panel) codon-phase S^1 in Evo 2 / AlphaGenome — only if time allows.

### Controls (mandatory, from the earlier review + Task 1 agent)
Matched bits/MDL (NOT matched feature count); random-init model; >=5 seeds + Tournament Top-K; the
local-sequence-window positional-confound control run FIRST (helix/codon phase may be trivially positional);
pre-registered Betti numbers / persistence (anti "Spokes, not a Ring", Gupta et al.); geometry x variable
DOUBLE DISSOCIATION (each geometry wins on its matched variable, loses on a mismatched one such as SASA).

================================================================================
## KILL CRITERIA (any one unmet → the strong claim collapses)
================================================================================
- KILL #1: T1 must be genuinely Omega(covering-number), not Omega(1). Check a bias + 2 antipodal scalar atoms
  (Dor26's scalar-SAE result) cannot already tile the planted circle cheaply, or the separation vanishes.
- KILL #2: the gap must be GEOMETRY, not CAPACITY. At matched bits/MDL the curvature-matched block must beat
  BOTH a scalar SAE AND a Euclidean block of equal dimension b. If plain Euclidean BSF already captures the
  torus, only the weaker SAE-vs-block claim survives (and that is Goodfire/SASA's turf).
- KILL #3: the ESM3 torus must be recovered from SEQUENCE-ONLY input and survive the local-window control,
  else "a protein model encodes a torus" is a positional artifact and the biology drops out.
- STRONGEST UNANSWERED REBUTTAL: "real SAEs represent a circle with ~2 split features spanning its plane, so
  'single-direction' is a strawman; your blow-up is just a labelling/identifiability problem (Bha26/Jer26),
  not a theorem." Must make T1 about faithful reconstruction of the curved embedding to L2 epsilon (not its
  linear span, which 2 atoms span), with T3/isometry doing the monosemanticity work — airtight BEFORE committing.

================================================================================
## SCOOP EXPOSURE (human must re-scan arXiv + goodfire.ai/research before committing)
================================================================================
- Goodfire: owns BSF and the Evo 2 SAE — maximal exposure.
- Penn State "SASA" (Dalili & Mahdavi, arXiv 2606.06333, NeurIPS 2026): independent Euclidean subspace-SAE —
  so "blocks beat scalar" is already taken; the theorem + curvature is the only safe headline.
- Zah26 "The Geometric Wall" (measured the phenomenon) and Jer26 "Deeper Analysis of BSF" (named curvature-
  matched blocks) — the two papers you are completing; cite as motivation, differentiate on the theorem.
- HyperSAE (hyperbolic SAE on LLMs) — UNVERIFIED (PyPI self-report); a human must check before any hyperbolic work.
- Engels et al. (circular features in LLMs) and Setlur (geometry-annotated ESM-2 SAE features) — cite, differentiate.
- Tan25f (sparse-dictionary non-identifiability, 1-D negative result) — the springboard for T3.

================================================================================
## SCOPE DECISIONS carried over
================================================================================
- DROP hyperbolic to a conditional appendix: highest implementation cost (Riemannian opt, boundary blow-ups),
  highest scoop risk, weakest bio match (fold/GO hierarchies are trained-on annotations → circular). Circle/
  torus is the publishable core. Only pursue hyperbolic if the hierarchy is both decodable AND tree-like (low delta).
- DROP Bonbon (sequence-only, closed-weight, unverified) and AlphaGenome 2 (API-only). Keep ESM3-open primary;
  Evo 2 / AlphaGenome v1 only as an optional codon-phase panel.
- Jacobian/J++ half: NOT in this paper (ill-defined on bidirectional ESM3; collides with the categorical Jacobian).

================================================================================
## TIMELINE (deadlines UNVERIFIED)
================================================================================
- ICML 2027 (full paper reported ~late Jan 2027, ~9 weeks): realistic ONLY for a reduced cut — T1+T2 proved
  cleanly, T3 as a conjecture + synthetic evidence, Jer26's prebuilt zoo, ONE ESM3 phi/psi vignette. The core
  is a proof, so it compresses better than a 4-model sweep.
- DECISION GATE ~mid-December: if T1 and T2 do not close cleanly by then, ABANDON ICML 2027. An almost-proved
  theorem is an ICML reject but a strong NeurIPS 2027 submission.
- Honest primary target: NeurIPS 2027 main track (NeurIPS Datasets & Benchmarks as the floor). ICLR 2027
  workshop (~Feb) as an interim stake if a workshop fits.
- Task 3 follow-up: NeurIPS 2027 / workshop, after Task 1.

================================================================================
## IMMEDIATE NEXT ACTIONS (for a human / next session)
================================================================================
1. Decide ICML 2027 vs NeurIPS 2027 as primary (drives how hard to push the proofs now).
2. Theory spike (highest priority, low compute): attempt T1 + T2 on the planted circle/torus; settle the
   antipodal-pair and "spanning vs reconstructing" questions (Kill #1, rebuttal) — this is the gate for everything.
3. Pilot 0 (tens of GPU-h, 24 GB): circular/toroidal block vs scalar SAE vs Euclidean BSF on ESM3-open
   helix-phase/omega (matched) + SASA (mismatched), sequence-only — look for a double dissociation at matched bits.
4. Human scoop re-scan: arXiv + goodfire.ai/research for a same-quarter curvature-dictionary theorem; verify
   HyperSAE, SASA (2606.06333), Zah26, Jer26, the ICML 2027 dates.
5. Dual-use: still owed a human/compliant-model pass (kept getting cut by the safeguard); benign structural
   variables only; this theory paper is low dual-use risk, Task 3 is the sensitive one.
