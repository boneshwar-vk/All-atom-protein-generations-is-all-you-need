# Four proposed research directions (to be critiqued + planned)
Author: orchestrating session, 2026-11-20. These are NEW proposals layered on the earlier plans
(see PLAN_ICML_synthesis.md, plan_task1_geometry.md, FINAL_FINDINGS.md, primer_bsf_jacobian_landscape.md).
Unifying thesis: biological models encode physical variables whose TRUE geometry is known a priori
(dihedral = circle; (phi,psi) = torus; helix/codon phase = circle; fold hierarchy = tree). That exact
geometric ground truth is the asset LLM interpretability lacks. Match the dictionary geometry to the
known biology and use physics as the validator.

## Pitch 1 — Learnable-curvature sparse dictionary ("let the model tell you the geometry")
Instead of fixing sphere vs hyperbolic per block, LEARN a curvature scalar kappa per feature-block,
differentiable, kappa>0 spherical / kappa=0 Euclidean / kappa<0 hyperbolic (stereographic / gyrovector
parameterization). The dictionary discovers each concept's intrinsic geometry. Headline figure: learned
kappa is positive for dihedral/phase features, ~0 for SASA/charge, negative for fold-hierarchy features —
geometry recovered unsupervised. Closest prior art found: Riemannian dictionary learning on DATA manifolds
(SPD/shapes; Xie12, Che15, Cha18), mixed-curvature product spaces (Gu18), hyperbolic NNs (Gan18b),
"Text Has Curvature" (Gro26, measures LLM-rep curvature), HyperSAE (hyperbolic SAE on LLMs, UNVERIFIED).
Claimed-open: a learnable PER-FEATURE curvature inside a SPARSE DICTIONARY on foundation-model activations.

## Pitch 2 — Phase-velocity / winding-number validation ("prove the ring is a ring with physics")
A genuine cyclic feature must advance its phase at the physically known rate: ~100 deg/residue in an
alpha-helix (3.6 residues/turn); exactly period-3 along a coding sequence; and the phase must RESET at a
frameshift. Headline figure: feature phase vs residue index tracking the helix turn rate; a sharp phase
jump exactly at an introduced frameshift (a built-in causal intervention with a known-correct answer).
Direct, physics-anchored rebuttal to "Spokes, not a Ring" (Gupta et al.) — stronger than topology alone.
Closest prior art: Engels et al. (circular day/month features in LLMs via SAE), neural phase precession
(Byr09, neuroscience). Claimed-open: phase-advance-matches-physics as an interpretability validator on bio models.

## Pitch 3 — Measure the capacity wall instead of proving it
Plant manifolds of known covering number into real activations; count scalar-SAE atoms vs one curved block
needed to reconstruct to a fixed tolerance epsilon. Headline figure: atoms-vs-covering-number — scalar SAE
climbs, curved block flat. This IS the separation, shown empirically (de-risks the theory; theorem optional).
Closest prior art: "The Geometric Wall" (Zah26, measured curvature->SAE width-scaling, no remedy), Jerpelea &
Ananthram "Deeper Analysis of BSF" (manifold zoo through Euclidean blocks, no curvature remedy), Bhalla
(do SAEs capture concept manifolds). Claimed-open: the curved-block remedy + the controlled atoms-vs-covering curve.

## Pitch 4 — Causal-efficacy map on ESM3 (steering without diffusion wash-out)
ESM3 is masked-generative, so steer and re-run the forward pass — no denoising trajectory, no wash-out.
Define causal efficacy = (output change from a matched-norm nudge) / (probe decodability); map over layers;
show geometry-matched block directions steer structure tokens where scalar SAE directions do not, and that
the most-decodable layer is not the most-steerable. Headline: steering success along the (phi,psi) coordinate
with wrap-around, curved vs scalar vs matched-random. Cite "Off-Manifold Collapse in Guided Protein Language
Models" (Zha26g) as the caution; add a validity filter. Natural follow-up / second section; reuses Pitch 1.

## Flagship = Pitches 1+2+3 as one paper; Pitch 4 is the steering follow-up.
Decision rule the user set: prefer running a SMALL TEST over demanding a proof, UNLESS a concrete paper
shows the idea is already done or concretely shown not to work.
