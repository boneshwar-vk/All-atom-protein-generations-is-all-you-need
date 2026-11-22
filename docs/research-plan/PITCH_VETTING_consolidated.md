# Consolidated vetting of the four pitches (2026-11-20)
From vet_pitch1_curvature.md, vet_pitch2_phase.md, vet_pitch3_capacity.md, vet_pitch4_steering.md.
Decision rule applied: run a small test unless a concrete paper shows it already done or disproven.

## Scoreboard
| Pitch | Verdict | Why |
|---|---|---|
| 1. Learnable-curvature dictionary | OPEN but narrow; headline at risk | Signed-curvature mechanism = Skopek 2019 mixed-curvature VAE (Sko19) + Gu18; and Sko19 reports learning curvature is BADLY CONDITIONED — on trees kappa drifted to ~0, NO sign recovery. The exact money-shot (unsupervised geometry recovery) is what the closest prior work could not deliver. Bio side (curved dictionary on a bio model) genuinely open. |
| 2. Phase-velocity / winding validator | PARTIAL; mostly pre-empted | Codon period-3 + frameshift leg ALREADY DONE (two NeurIPS 2026 papers: emergent codon-frame Z3 representations + causal frame steering). Category error: phase-velocity does NOT rebut "Spokes, not a Ring" (Gup26b disputes between-state arcs, a different question). ESM3 uses RoPE → constant phase velocity can be an index artifact. Salvage = feature phase tracking STRUCTURE-derived phase slips (kinks/3-10 helices) that RoPE cannot fake. |
| 3. Measure the capacity wall | ALREADY DONE (hard kill) | SASA (Dal26, arXiv 2606.06333, NeurIPS 2026) already has: the k-sparse covering-number definition, the scalar lower bound theorem, the empirical atoms-vs-covering curve (Fig 2c), AND the O(1) block remedy (Fig 1, Thm 11). Structural killer: a FLAT Euclidean block of equal dim already pays O(1) for every named manifold → "curved" does not beat "Euclidean block of equal dim"; the key control is pre-registered to TIE. |
| 4. ESM3 causal-efficacy steering | OPEN but narrow + two sub-claims already contradicted | Premise "ESM3 escapes wash-out" is half-false: iterative unmasking has its own commitment/re-projection problem (Hin26 masked-pLM edits don't survive decode; Zha26g off-manifold collapse). "Scalar SAE doesn't steer structure tokens" is likely FALSE: Steenwyk (Ste26) already steers ESM3 at layer 33 with a scalar SAE direction (dose r=0.87). Open atoms remain (below). |

## What this collectively kills
- The "new GEOMETRY for the dictionary" headline (sphere/hyperbolic/curved-block beats Euclidean): dead or tying.
  SASA owns capacity+remedy; Skopek owns learnable signed curvature and shows it's ill-conditioned; a flat block
  of equal dim already captures these low-dim manifolds. Do NOT spend the headline on "curved beats flat".
- The "capacity curve" as a contribution: published (SASA).
- "Phase velocity rebuts Spokes" and the codon leg: wrong framing / scooped.
- "Scalar SAE can't steer ESM3": false (Steenwyk).

## What genuinely survives (the un-scooped kernel, flagged independently by >=2 agents)
These are CAUSAL + IDENTIFIABILITY claims that SASA (theory/recon, no bio, no causal), Steenwyk (single-layer
scalar steering, no circular geometry, no identifiability), and Skopek (VAE, no sparsity/bio/causal) did NOT make:
1. IDENTIFIABILITY up to the isometry group: recover the biological coordinate itself (the angle on the circle /
   the point on the torus) as a monosemantic, gauge-free quantity — not merely "a subspace exists". Winding/phase
   is rotation-invariant, so it sidesteps BSF's within-block gauge ambiguity. SASA proves a subspace is enough for
   RECONSTRUCTION; nobody has shown the recovered coordinate IS the physical coordinate up to isometry.
2. The WRAP-AROUND causal dose-response test: steer along the recovered circular coordinate and show the output
   returns to baseline at 2*pi — a test a 1-D latent structurally cannot pass, and which SASA/Steenwyk did not run.
3. STRUCTURE-DERIVED PHASE SLIPS: feature phase tracks kinks / 3-10 helices beyond index-mod-3.6 — the one
   periodicity signal RoPE/positional encoding cannot manufacture. The RoPE-proof causal-validator panel.
4. LAYER-RESOLVED decodable != steerable efficacy map on ESM3 + a COMMITMENT/PERSISTENCE finding: naive activation
   edits are erased by iterative unmasking/decoding (a real negative/calibration result, extending Jed26b/Hin26/Zha26g
   from "one layer / ESM-2 / Boltz" to "layer map on ESM3 structure tokens").

## The consolidated surviving paper (pivoted, honest)
NOT "a new geometry for sparse dictionaries". INSTEAD:
"Do biological foundation models causally USE the known geometry of physical variables, and can we recover that
geometry as an identifiable, gauge-free, steerable coordinate?" Contribution = CAUSAL + IDENTIFIABILITY on exact
physical ground truth, with (a) recovery-up-to-isometry of the (phi,psi) torus / helix-phase circle, (b) the
wrap-around dose-response causal test, (c) the RoPE-proof phase-slip panel, (d) the commitment/persistence negative
result. Geometry-matched vs scalar vs matched-random is a CONTROL inside this, not the headline.
Primary model ESM3-open; synthetic planted manifolds for calibration. This is differentiated from SASA, Steenwyk,
Skopek, and the codon papers.

## The ONE test to run first (predicted to FAIL — which is the point)
Pure synthetic, ~1-2 CPU-hours, no model weights: plant circle/torus/sphere/swiss-roll in R^64; fit scalar-SAE(k=1,2)
vs Euclidean-block vs curved-block at MATCHED BITS; measure atoms-to-FVU<=epsilon.
- If curved TIES Euclidean-block-of-equal-dim (predicted): the "geometry" headline is confirmed dead → commit to the
  identifiability+causal paper above, do NOT build the curved-dictionary machinery. Saves weeks.
- If curved strictly wins at matched bits (only on a manifold whose linear embedding dim >> intrinsic+1): revisit.
Then, on a real model: ESM3-open (phi,psi) persistence gate G0 (do edits survive iterative decode?) BEFORE any
steering claim — this is the most likely death of the causal half, so test it cheapest-first.

## Landmine / scoop list (a human must read these full texts before writing)
- Dal26 SASA (arXiv 2606.06333, NeurIPS 2026) — owns capacity theorem + remedy. CRITICAL.
- Sko19 mixed-curvature VAE — owns learnable signed curvature; documents the ill-conditioning. HIGH.
- Ste26 Steenwyk — ESM3 layer-33 scalar SAE steering works. HIGH (your must-beat).
- Zha26g Off-Manifold Collapse (arXiv 2608.18597) + Hin26 (masked-pLM edit persistence) — the wash-out/commitment antecedents. HIGH.
- Two NeurIPS 2026 codon-frame papers (Z3 representations + causal frame steering) — own the codon leg. HIGH for genomics.
- Jed26b (Boltz-1 decodable!=steerable), Xun26 (LLM readable-before-causal) — the decodable!=steerable antecedents.
- BioEmu+ESM3 SAE steering camera-ready (NeurIPS 2026) — check for a (phi,psi)/steering scoop. UNVERIFIED.
- HyperSAE (hyperbolic SAE on LLMs, PyPI) — UNVERIFIED; owns "non-Euclidean SAE on LLMs".
Network note: neurips.cc blocked this session → codon papers + BioEmu camera-ready read from abstracts only; verify.
