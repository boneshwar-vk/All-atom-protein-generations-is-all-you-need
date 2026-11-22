# MASTER PLAN — Identity, Cause, and Truth-vs-Confidence in biological foundation models
Owner: this session (full ownership taken 2026-11-22). Merges plan_exec_engineering.md, plan_derisk_gates.md,
plan_narrative_novelty.md over PROJECT_BRIEF.md, after four rounds of independent vetting. Decisions are MADE below,
not deferred. "UNVERIFIED" = confirm against primary source before relying on it.

================================================================================
## 0. DECISIONS (owned, not open questions)
================================================================================
D1. ONE paper. SPINE = Track A (ESM3-open, exact DSSP/physics ground truth, cheapest, lowest dual-use).
    Track B (Boltz, experimental Kd vs ipSAE) is an EXTENSION that generalises the hook and supplies the
    "confidently wrong" example; it is never load-bearing for acceptance.
D2. HEADLINE = TRUTH vs CONFIDENCE: a biological model holds a physically-true quantity in a direction distinct
    from its own calibrated confidence, and we show this with identity + causality that the current LLM
    truth/confidence wave lacks. "A fancier dictionary geometry (sphere/hyperbolic)" is NOT the contribution —
    the vetting (SASA/Dal26) killed "curved beats flat"; curved/block dictionaries appear only as CONTROLS.
D3. ORDER = cheapest-decisive-test first; the most-likely death (edits wash out under iterative decoding) is
    gated BEFORE any expensive steering. Probe-level gates all clear before one costly decode/steer job launches.
D4. Every headline compares at MATCHED BITS/MDL and must beat a single shared NUISANCE-ONLY BASELINE
    (position/RoPE phase + train-set-membership/cluster-id + model self-confidence). No result counts unless it
    beats that baseline by a pre-registered margin.
D5. Minimum Publishable Result is defined (Sec 9) so a null on the ambitious claims still yields a paper.
D6. Durability: this dossier is committed to the repo (docs/research-plan/) because the scratchpad is temporary.

================================================================================
## 1. THE PAPER
================================================================================
One-sentence contribution: On quantities fixed by physics — backbone (phi,psi) and helix phase in ESM3 (from
DSSP), and laboratory Kd in a co-folding model — a biological foundation model represents the quantity as an
identifiable, gauge-free coordinate that (i) matches the physical truth up to isometry on a held-out split,
(ii) causally drives the output with a wrap-around (returns to baseline after a full 2*pi turn) that a single-
direction feature cannot fake, and (iii) lives in a direction distinct from the model's own calibrated confidence
(pLDDT; ipSAE) — shown convergently for a periodic STRUCTURAL and a scalar THERMODYNAMIC variable.

Why it clears a top bar:
- It is causal + identifiable on EXACT external ground truth, which SAE/probe interpretability on LLMs cannot have.
- The truth-vs-confidence separation is a 2026 LLM frontier; biology is the one setting where BOTH truth (physics)
  and a deployed confidence head are externally pinned, and we add causality + identifiability the LLM work lacks.
- It converts the finding into a defensive AUDITING primitive (a "trust axis" that flags confidently-wrong outputs).
- It ships honest negative/calibration results (persistence/commitment) that the field needs.

The six claims (each -> experiment -> figure -> control -> prior art it out-positions):
- C1 IDENTITY up to isometry: recovered coordinate = physical truth on held-out data (circular/toroidal corr,
    geodesic R^2 after train-fitted isometric alignment). Out-positions SASA (reconstruction only, no identity).
- C2 WRAP-AROUND causal dose-response: steer the coordinate; output follows and closes at 2*pi; a scalar latent
    cannot. Out-positions Steenwyk (single-layer scalar steering, no circular closure).
- C3 TRUTH ⟂ CONFIDENCE (the hook): the truth coordinate and the model's confidence (pLDDT / ipSAE) occupy
    distinct directions (principal angle > 0, causal double dissociation). Out-positions the LLM truth/confidence
    wave (no external truth, no causality, no identifiability).
- C4 RoPE-PROOF phase-slip: feature tracks structure-derived phase slips (kinks, 3-10 helices) that positional
    encoding cannot manufacture. Out-positions "it's just position".
- C5 PERSISTENCE / COMMITMENT: whether mid-computation edits survive iterative decoding; reported honestly
    (likely a partial negative). Extends Zha26g / Hin26 / Jed26b from one layer / ESM-2 / Boltz to a layer map.
- C6 CROSS-TRACK INVARIANCE: the same identity+cause+truth⟂confidence pattern holds for a periodic structural
    variable (A) AND a scalar thermodynamic one (B) — the answer to "cherry-picking" and B's reason to exist.

================================================================================
## 2. PHASE-BY-PHASE PLAN (cheapest-decisive first)
================================================================================
Each phase: GOAL -> STEPS -> DATA/CACHE -> TRAINED -> METRICS -> CONTROLS -> GATE -> DELIVERABLE -> COMPUTE.

### PHASE 0 — Synthetic calibration + the geometry kill-test (CPU, hours)
GOAL: settle, with no model weights, (a) does a curved/circular code beat a flat Euclidean block of equal dim at
matched bits (predicted TIE -> confirms curved is a control, not the headline), and (b) is a planted angular
coordinate recoverable up to isometry, while single-direction (scalar) codes pay a covering-number penalty.
STEPS: plant circle (and torus) in R^64 with noise; fit scalar-k=1 dictionary (vary atom count), flat 2D block
(PCA), circular/normalized 2D block; measure reconstruction FVU vs atoms at matched bits, and angle-recovery
circular correlation up to isometry.
METRICS: FVU vs #atoms; atoms-to-FVU<=eps; circular corr(recovered angle, true) after best rotation/reflection.
GATE G0: GO to real models if (i) curved ~ flat at matched bits (headline correctly demoted) AND (ii) scalar pays
the covering penalty AND (iii) angle recovers up-to-isometry (corr>0.95) on clean data. If curved STRICTLY beats
flat at matched bits, revisit (unexpected). DELIVERABLE: Fig S1 + the metric harness reused in all later phases.
COMPUTE: CPU, minutes-hours. [Implemented now: code/phase0_synthetic/phase0_synthetic.py]

### PHASE 1 — Persistence / commitment gate (the most-likely death) 
GOAL: do mid-computation edits survive to the final output? Test BEFORE any steering claim.
STEPS: Track A (ESM3-open): at a mid layer, add a known displacement to the residual at target positions, run the
rest of the masked-decoding forward, measure whether the decoded structure token / backbone reflects the edit vs a
matched-random displacement. Track B (Boltz): inject at a mid diffusion step, measure final-structure change.
DATA/CACHE: ~2-5k single-chain domains (A); ~200 benign complexes (B).
METRICS: realized final-output delta vs matched-random; monotonicity; a persistence threshold.
CONTROLS: matched-norm random direction; validity filter (pLDDT/pTM/clash/ProteinMPNN-refold) so a change is not
off-manifold garbage (cite Zha26g off-manifold collapse).
GATE G2: GO if edits persist above matched-random and stay on-manifold. NO-GO -> pivot to identity + truth⟂confidence
+ the persistence NEGATIVE as a headline (still publishable; extends Hin26/Zha26g). COMPUTE: A 24GB feasible; B 1x80GB.

### PHASE 2 — Real-model IDENTITY + TRUTH⟂CONFIDENCE (the core)
GOAL: C1 and C3. Recover the physical coordinate and show it is distinct from the model's confidence.
STEPS (A): cache ESM3 residual stream (sequence-only input) at ~6-8 layers; compute DSSP (phi,psi)/helix phase/SASA
from experimental PDB; fit circular/torus block + scalar-SAE + linear/MLP probes; align recovered coordinate to
truth on a held-out split; measure principal angle between the truth direction and the pLDDT-confidence direction;
run the confidence-matched-pair control (hold confidence fixed, vary true physics) to kill the entanglement rebuttal.
STEPS (B): leakage-aware (time + scaffold) splits of PDBBind/PLINDER; cache Boltz single-rep + interface-pair subset;
fit a Kd coordinate; compute the three-way angle among {experimental Kd, ipSAE self-confidence, Boltz's own predicted
affinity}; run the Kd leakage probe FIRST.
METRICS: held-out circular/toroidal corr & geodesic R^2 (C1); principal angle + causal double dissociation + trust-
residual AUROC for flagging confidently-wrong cases (C3).
CONTROLS: random-init model; >=5 seeds; positional/local-window control FIRST (RoPE); nuisance-only baseline;
confidence-matched pairs; leakage probe (B).
GATE: identity corr beats nuisance baseline by margin; principal angle significantly > 0 and survives the
confidence-matched control. NO-GO on identity -> MPR benchmark paper. COMPUTE: 1-2x80GB.

### PHASE 3 — CAUSE (wrap-around) + RoPE-proof validator
GOAL: C2 and C4. STEPS (A): steer the recovered coordinate across >=8 targets incl. 2*pi wrap; measure decoded
backbone change + closure; curved vs scalar vs matched-random at matched norm; phase-slip panel on kinks/3-10 helices.
STEPS (B): monotone dose-response on buried surface area (model-independent) as the primary causal target; SKEMPI
mutation agreement as a causal oracle; must-beat baseline = Feynman-Kac / guidance steering; ipSAE used only to expose
Goodhart, never as the steer target. GATE G4: wrap-around closes and beats matched-random; geometry gives measurable
lift over scalar at matched norm. COMPUTE: A 1-2x80GB; B up to 1.5-3k GPU-h (heaviest).

### PHASE 4 — CROSS-TRACK INVARIANCE + full sweep + write-up
GOAL: C6 + robustness. Full >=5-seed / matched-bits-MDL / random-init sweep on both tracks; stitch the truth⟂confidence
result across the periodic structural and scalar thermodynamic variable; freeze pre-registered predictions. DELIVERABLE:
the paper. COMPUTE: small cluster 4-8k GPU-h for the full ensemble (the core result needs far less).

================================================================================
## 3. KILL-GATE LADDER (run in this order; probes before any steer/decode job)
================================================================================
G0 synthetic matched-bits + identifiability (CPU) ||  G-B1 Kd-leakage probe (CPU/1 GPU)
-> G1 RoPE/positional control, probes only
-> G2 persistence/commitment, ONE decode cell
-> G-B2 ipSAE-collinearity (is Kd just ipSAE?)
-> G3 decodable!=steerable + geometry-lift
-> G4 wrap-around dose-response
-> G-B3 ipSAE-Goodhart + SKEMPI agreement
Each gate: claim, cheapest falsifier, null+controls, quantitative GO/NO-GO, pivot-or-kill. The three most likely
kills, earliest testable: (1) G2 persistence (kills the causal half) — test right after G1; (2) G-B1 Kd-leakage —
test in parallel with G0; (3) G1 RoPE periodicity — probes on cached activations, before G2.

================================================================================
## 4. CONTROLS & BASELINES (mandatory)
================================================================================
- NUISANCE-ONLY BASELINE (the single highest-value control): a decoder given only {position/RoPE phase} U
  {train-set-membership + cluster id} U {model self-confidence}. Every GO threshold = "beat this by a margin".
- Confidence-matched negative pairs (vary true physics, hold confidence fixed) — run FIRST inside Phase 2 to kill
  the "truth and confidence are one entangled signal" rebuttal.
- Random-init same-architecture model through the full pipeline; >=5 seeds + stability statistic + Tournament-TopK.
- Positional/local-window control FIRST (RoPE can fake periodicity); sequence-only input for A.
- Matched BITS/MDL comparisons; matched-norm random-direction for every steering claim.
- Leakage-aware time + scaffold splits + an explicit leakage probe (B).
- Validity filter (pLDDT/pTM/clash/ProteinMPNN-refold) on every steered output.

================================================================================
## 5. DATA, MODELS, TOOLS (verified this session unless marked UNVERIFIED)
================================================================================
- ESM3-open weights: gated, non-commercial (UNVERIFIED exact terms + d_model ~1536).
- Boltz: open, MIT (jwohlwend/boltz) [verified by planner]. AF3/Chai = alternates.
- ipSAE: DunbrackLab/IPSAE + PyPI ipsae 1.0.1 [verified]. Used as CONTRAST only.
- PLINDER (leakage-aware splits) [verified exists]; PDBBind/BindingDB/SKEMPI/CASF counts+licences UNVERIFIED.
- geoopt (sphere / product-torus / Poincare) [verified]; DSSP/mdtraj; ProteinMPNN for refold validity.
- Phase 0 needs none of these (pure numpy).

================================================================================
## 6. PRIOR-ART DIFFERENTIATION (read full texts before writing)
================================================================================
SASA/Dal26 (capacity+block remedy — CONCEDE, don't compete); Skopek/Sko19 (learnable curvature ill-conditioned —
we don't rest on it); Steenwyk/Ste26 (scalar steers ESM3 L33 — we must beat on matched-norm geometry lift AND add
identity+truth⟂confidence); Zha26g + Hin26 (wash-out antecedents — we extend to a layer map + honest negative);
PairSAE/Mig26 (used the model's OWN affinity — we use EXPERIMENTAL Kd); Jed26b (decodable!=steerable Boltz — we add
geometry + wrap-around); NeurIPS-2026 codon-frame papers (CITE, don't compete); Setlur/2608.26419 (geometry-annotated
ESM-2 SAE — closest protein prior art); LLM truth/confidence wave (2603.25052, 2602.08159, 2602.20273 — UNVERIFIED,
the nearest conceptual scoop; verify before committing the hook).

================================================================================
## 7. RISKS / SCOOP
================================================================================
- Strongest unanswered rebuttal: truth and confidence may be two readouts of one "well-structured" signal (pLDDT high
  where geometry is regular). Close it with the confidence-matched-pair control + a confidently-wrong slice.
- Scoop: truth-vs-confidence is a hot 2026 LLM topic; differentiate on external ground truth + deployed confidence head
  + causality + identifiability. Human must read the three LLM papers + check goodfire.ai + BioEmu/HyperSAE pre-submission.
- Heaviest-effort phases: 3B (Boltz mid-diffusion causal + SKEMPI + FK baseline), 2B (Kd identity on leakage splits),
  1 (per-timestep persistence caching). Keep B minimal until A is end-to-end.

================================================================================
## 8. ETHICS / DUAL-USE (owes a human/compliance pass)
================================================================================
Track A (angles) low risk. Track B (binding) capability-adjacent. Framing = auditing/calibration (find and TRUST the
representation; flag confidently-wrong outputs via the trust axis). Benign complexes only (exclude toxins / viral-fusion
/ immune-evasion / select agents). ipSAE a CONTRAST never a steer target. No released affinity-maximising pipeline.
The dual-use statement must get a human/compliant-model pass (a safeguard has repeatedly cut it).

================================================================================
## 9. MINIMUM PUBLISHABLE RESULT (survives if the ambitious causal claims fail)
================================================================================
A leakage-aware exact-geometry benchmark + calibration/negative paper = (1) the physical-ground-truth suite +
RoPE-proof protocol, (2) identity-up-to-isometry recovery, (3) the ESM3 persistence/commitment negative, (4) the
truth!=confidence (Kd-vs-ipSAE; (phi,psi)-vs-pLDDT) angle. Concedes SASA/Steenwyk/Skopek; contributes what they lack.

================================================================================
## 10. IMMEDIATE NEXT ACTION
================================================================================
Run PHASE 0 (code/phase0_synthetic/phase0_synthetic.py) — pure numpy, no weights/network. It either confirms
"curved ties flat at matched bits" (commit to the truth+identity+cause paper, drop curved-dictionary machinery) or
flags the unexpected. Then: secure ESM3-open weight access and begin Phase 1 persistence caching on Track A.
Open items a human must close: ESM3 weight access + terms; dataset counts/licences; read the 3 LLM truth/confidence
papers; dual-use pass.
