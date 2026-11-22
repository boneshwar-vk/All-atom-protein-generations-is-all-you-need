# PROJECT BRIEF — the consolidated project to plan (2026-11-22)
This is the single definition all planners work from. Full evidence base + vetting in the same scratchpad:
PITCH_VETTING_consolidated.md, FINAL_FINDINGS.md, PLAN_ICML_synthesis.md, plan_task1_geometry.md,
plan_task3_steering.md, vet_pitch1_curvature.md..vet_pitch4_steering.md, primer_bsf_jacobian_landscape.md.

## Working thesis (the throughline)
Biological foundation models hold quantities whose TRUE value is known from physics/chemistry (backbone
dihedral angles, helix phase, interface binding strength). We ask, for such a quantity, three things the
existing literature has NOT answered together: (1) IDENTITY — is the internal feature we recover actually
THAT physical quantity (up to an unimportant rotation of axes = "up to isometry"), not merely some directions
that reconstruct the activation; (2) CAUSE — if we change that feature inside the model, does the output move
in the matching way (including returning to baseline after a full circle for an angular quantity); (3) TRUTH
vs CONFIDENCE — does the model keep the real physical quantity in a DIFFERENT internal direction from its own
self-confidence about that quantity. The unifying contribution is causal + identifiability + truth-vs-confidence
on EXACT physical ground truth. "A fancier dictionary geometry" is NOT the contribution (see landmines).

## Two demonstration tracks (one paper, one method, two quantities)
TRACK A — geometric/structural, single-protein model.
  Model: ESM3-open (esm3-sm-open-v1, 1.4B, 48 layers, non-commercial). Masked generative, uses RoPE.
  Quantity: backbone (phi,psi) torus; helix phase; (secondary) relative SASA as a NON-periodic control variable.
  Ground truth: computed from experimental PDB structures by deterministic code (DSSP / mdtraj), NOT database labels.
  Confidence contrast for track A: the model's own pLDDT/structure-token confidence.
TRACK B — binding/interface, co-folding model.
  Model: Boltz (open; Boltz-2 predicts affinity and has PairSAE prior work to differentiate from). AF3/Chai as alt.
  Quantity: experimental binding strength Kd (real, lab-measured).
  Confidence contrast for track B: ipSAE / ipTM (the model's OWN interface self-confidence; Dunbrack lab score,
  code github.com/dunbracklab/IPSAE). ipSAE is a CONTRAST, never the steering target (steering toward it is circular).
  Also keep a model-independent geometric interface descriptor (buried surface area / contacts) as a third, distinct target.
  Data with Kd: PDBBind (protein-ligand Kd/Ki/IC50), PLINDER (leakage-aware splits), BindingDB (scale), SKEMPI
  (protein-protein delta-delta-G on mutation = a built-in causal test), CASF (benchmark). Exact counts/licences UNVERIFIED — verify.

## The four surviving technical pillars (apply to both tracks)
1. IDENTIFIABILITY UP TO ISOMETRY: show the recovered coordinate IS the physical one (monotone/rotation-invariant
   alignment to ground truth on a held-out split), not just a reconstructing subspace. Winding/phase is rotation-invariant.
2. WRAP-AROUND CAUSAL DOSE-RESPONSE: steer the recovered coordinate; output follows and returns to baseline at 2*pi
   (a test a 1-D latent structurally cannot pass). For Kd: monotone dose-response + SKEMPI mutation agreement.
3. RoPE-PROOF / PHYSICS VALIDATOR: feature tracks STRUCTURE-derived irregularities (kinks, 3-10 helices; phase slips)
   that positional encoding alone cannot manufacture. For Kd: tracks experimental Kd, not the model's predicted affinity.
4. PERSISTENCE / COMMITMENT: test whether a mid-computation edit SURVIVES the model's iterative decoding/generation
   to the final output (most likely failure point; test cheapest-first). Report honestly as a calibration finding if it fails.

## Decision rules (from the user)
- Run the CHEAPEST DECISIVE test first; prefer a small test over a proof unless a paper already settles it.
- Compare at MATCHED BITS / MDL, never matched feature count (a flat block of equal dim already captures low-dim manifolds).
- Controls are mandatory, not optional: random-init same-architecture model; >=5 seeds + a stability statistic;
  POSITIONAL/local-window control run FIRST (RoPE can fake periodicity); sequence-only input for track A; leakage-aware
  time/scaffold splits for track B; matched-norm random-direction control for every steering claim; validity filter
  (pLDDT/pTM/clash/ProteinMPNN-refold) so a "steered" output is not just off-manifold garbage.
- "geometry-matched vs scalar-SAE vs matched-random" is a CONTROL inside the paper, not the headline.

## Landmines / prior art the plan MUST differentiate from (read full texts before writing)
SASA (Dal26, arXiv 2606.06333, NeurIPS 2026) owns capacity theorem + block remedy. Skopek (Sko19) owns learnable
signed curvature + documents its ill-conditioning. Steenwyk (Ste26) already steers ESM3 at layer 33 with a SCALAR SAE
direction. Off-Manifold Collapse (Zha26g, arXiv 2608.18597) + masked-pLM edit persistence (Hin26) = the wash-out
antecedents. PairSAE (Mig26, arXiv 2606.27440) predicted Boltz-2's OWN affinity (circular) — track B must use EXPERIMENTAL Kd.
Jedryszek (Jed26b, arXiv 2608.11475) = decodable!=steerable on Boltz-1. Two NeurIPS 2026 codon-frame papers own the codon
leg. BioEmu+ESM3 SAE-steering camera-ready + HyperSAE = UNVERIFIED, check before writing. Setlur (2608.26419) = geometry-
annotated ESM-2 SAE features (closest protein prior art).

## Known failure modes the plan must gate on
curved-block ties flat-block at matched bits; learnable-curvature ill-conditioned; RoPE positional artifact; wash-out/
commitment erases edits; dataset leakage inflates Kd prediction; Goodhart if ipSAE is used as a target.

## Ethics / dual-use
Track A (angles) is low risk. Track B (binding) is capability-adjacent (binder design). Keep defensive/auditing framing
("can we find and trust the binding representation; does the model separate truth from confidence"); benign complexes
only (exclude toxins/viral-fusion/select-agents); no released affinity-maximizing steering pipeline. The dual-use section
needs a human/compliant pass (a safeguard has repeatedly cut it).

## Target
ICML-standard result (strong, causal, identifiable, with exact ground truth). Keep honest go/no-go gates so a null
result becomes a calibration/benchmark paper rather than a dead end.
