# VET — PITCH 2: Phase-velocity / winding-number validator for cyclic features

Reviewer stance: senior mech-interp + structural-biology, blunt. Benign framing (structural/geometric
variables only). Research/planning only — no repo changes. Date 2026-11-20.
Decision rule applied: prefer a small decisive experiment UNLESS a concrete paper shows it already done /
shown not to work — then say so with the reference.

## VERDICT IN ONE LINE
**PARTIAL / SPLIT.** The **codon period-3 + frameshift leg is ALREADY DONE** (two concurrent NeurIPS 2026
papers that fit a 2D codon-phase subspace, rotate it 120°/nt to causally shift the reading frame, and show
the shifts obey the cyclic group Z3 and compose with indels). The **alpha-helix phase-velocity leg is OPEN**
— nobody has measured ~100°/residue helix phase advance in model activations — **but it is now a derivative
application of a template someone else just published**, not a new method. Do not build the paper on Pitch 2
as a standalone method; at most it is one physics-anchored validator panel inside the BSF/manifold benchmark.

---

## TASK 1 — PRIOR-ART KILL CHECK

Question: has anyone validated a circular/periodic FEATURE in a model's activations by its PHASE-ADVANCE
RATE / winding number against a physical period?

### 1a. The codon / reading-frame leg — ALREADY DONE (HIGH confidence, concurrent)
Two NeurIPS 2026 entries do essentially the exact experiment Pitch 2 proposes for genomics. Seen only as
conference abstracts (neurips.cc full text is network-blocked in this session — the two pages must be
whitelisted to read them in full; treat quoted detail as from the abstract/landing text, UNVERIFIED on
method depth):

- **"Emergent Codon Frame Representations and Causal Translation Steering in Genomic Foundation Models"**
  (NeurIPS 2026, neurips.cc/virtual/2026/160431). Reported: across **four** genomic models, coding
  representations show **stronger period-3 structure than non-coding**; **codon phase is linearly decodable**
  from hidden states; they **fit a 2D codon-phase subspace and rotate it** — a **120° rotation moves the
  protein-decoding reading frame by exactly one nucleotide**, the inverse rotation moves it the other way,
  **repeated rotations follow modulo-3 arithmetic**, and **matched controls produce no signed frame shift**.
  In the multimodal model (MIMIC) the leading coding-RNA representation **reorganizes from nucleotide identity
  toward codon phase with depth**.
- **"Emergent, Context-Dependent Group Representations in Genomic Foundation Models"** (Kovalic, Zhang,
  Golkar, Ho — NeurIPS 2026, neurips.cc/virtual/2026/157321). Reported: across **four** genomic FMs, decoded
  **frame shifts follow the cyclic group of order three** and **compose predictably with insertions and
  deletions** (= the frameshift intervention).

Map to Pitch 2's own words: "exactly period-3 along a coding sequence" ✔, "phase must RESET at a frameshift"
✔ (indel composition / 120°-rotation steering), "built-in causal intervention with a known-correct answer"
✔ (rotate by the physical per-step angle; Z3 modular check is the winding/known-correct answer), "the NULL a
spoke fails" ✔ (matched-control direction gives no signed shift). This is not merely "period-3 detected"; it
is the **full phase-as-cyclic-group-coordinate + physical-rate + causal-frameshift package**. The genomic leg
is scooped end to end.

### 1b. The alpha-helix phase-velocity leg — OPEN (MEDIUM-HIGH confidence) but derivative
No paper measures helix phase advance (~100°/residue, 3.6 res/turn) in a protein-LM's activations and checks
the rate against the physical helical repeat. Closest items, and why each misses:
- **Che12** "Detecting Periodicity Associated with the Alpha-Helix Structure Using Fourier Transform"
  (DOI 10.4236/CMB.2012.24011). DFT of a **sequence property (hydrophobicity-like)** recovers the ~3.6-residue
  helix periodicity — but on **raw sequence/structure, not model activations**. This is your physics anchor
  and prior art for "helix periodicity via Fourier," not a scoop of the interpretability claim.
- **Set26** "Interpreting Latent Protein Language Model Features with Geometric Annotations" (closest protein
  prior art). Annotates ESM-2 SAE features with local Cα geometry incl. **signed Cα torsion**, curvature,
  planarity, FDR-controlled, with ablation→contact shifts. Read confirms: **no periodicity, no phase advance
  per residue, no Fourier, no winding** — torsion is used as a static local descriptor, never a repeating
  phase. OPEN confirmed, but Set26 owns the "geometric SAE annotation" turf you'd be adjacent to.
- **Bes25** "Towards Understanding the Shape of Representations in PLMs." Global shape-space (SRV) + kNN graph
  filtrations across ESM-2 layers. Read confirms: **no periodicity, no helix phase, no persistent homology /
  Betti, no Fourier.** OPEN confirmed.
- **Liu26f** "Universal physical principles of protein structural genesis emerge in LM representation space"
  (ProtGenesis, bioRxiv 2026-08, DOI 10.64898/2026.02.20.706798). Maps assembly/elongation/mutation into PLM
  rep-space **trajectories** and claims "discrete topological transitions." Occupies the "physics emerges in
  rep-space" narrative you'd be pitching into — **must cite/differentiate** — but it is trajectory/topology,
  **not phase-velocity of a cyclic feature**. Not a scoop, but a framing collision.
- **Kes26** "Induction Meets Biology: Repeat Detection in PLMs" — mechanism of detecting sequence repeats;
  adjacent to "periodicity" but about induction-head repeat copying, not helical phase.

### 1c. The LLM lineage you are rebutting — characterized precisely
- **Eng24** (Engels et al., "Not All LM Features Are One-Dimensionally Linear," ICLR 2024, arXiv 2405.14860).
  Read confirms: they validate circular features by (i) label-ordered PCA pictures, (ii) **irreducibility**
  metrics (separability index, ε-mixture index), (iii) **causal modular-arithmetic interventions**, (iv)
  off-distribution angle sweeps. They **never measure an empirical phase-advance RATE against a known
  physical period** — targets are hand-set to 2πα/7 and 2πα/12. So "phase-advance-matches-physics" is
  genuinely absent from Engels. Good news for novelty of the *rate* idea; bad news is the discrete labels
  have no continuous physical period to match in the first place.
- **Gup26b** "Spokes, not a Ring: Testing Continuity of Representation Manifolds" (Gupta, Fruengel,
  Heimersheim; **NeurIPS 2026**; arXiv 2610.37796). Read confirms what it refutes and — critically — what it
  does **not** claim:
  - It attacks the **continuous-manifold interpretation**: that points *between* the 7 weekday states carry
    meaningful intermediate values. Evidence: a polar-grid Jacobian (L28 resid wrt L2 act) has **seven minima
    at the weekdays with low-sensitivity "spokes" radiating out**; steering is **step-like**; time-of-day,
    uncertainty, and natural text **do not populate the arcs**.
  - It **explicitly does NOT** refute the circular arrangement, and it **does not** test phase advance,
    winding number, phase velocity, or uniform spacing (measured adjacent spacing is a non-uniform
    44.33°–59.81°).
  - **Their stated bar for "continuous":** intermediate activations must (a) occur naturally/in-distribution
    AND (b) causally move downstream behavior like intermediate concept values. They state outright that a
    **phase-progression / physical-period experiment "would address a different dynamical question, but would
    not by itself establish that the arcs encode meaningful intermediate values."**
  → **This is the sharpest finding of the whole review:** Pitch 2's premise that phase-velocity is "a direct,
  causal rebuttal to Spokes-not-a-Ring" is a **category error**. Matching the helix rate proves the *states
  advance at the right rate*; it does **not** prove the *between-state arc is meaningful*, which is the only
  thing Gupta et al. disputed. A helix could be seven-residues-worth of discrete states that happen to sit at
  the right angles and still be "spokes." To actually rebut them you need the **intermediate-phase causal
  test** (off-grid / sub-residue phase has graded, in-distribution downstream effect), which is a different
  and harder experiment than phase-velocity.
- **Feu26** "Arithmetic in the Wild: Llama uses Base-10 Addition to Reason About Cyclic Concepts" (arXiv
  2605.01148). Read of abstract: circular geometry is present but the model computes month-addition with a
  **generic base-10 mechanism + Fourier features of period 2/5/10**, NOT the concept period (12). I.e.
  **circular geometry ≠ computation in that period** — the dominant live skepticism. Combined with the
  field's "decodable ≠ causal" (Jedryszek, your FINAL_FINDINGS §3), this is the headwind: you must show the
  helix phase is *used*, not merely present.

### 1d. The method (phase/winding extraction) is itself mature elsewhere
- Circular coordinates via **persistent cohomology** (Sil09, "Persistent Cohomology and Circular
  Coordinates"; Gra26d "Selecting Interpretable Circular Coordinates from Data"; Blu24 "circular coordinates
  in recurrent time series"). Extracting a winding/phase coordinate from a point cloud is standard TDA. Novel
  part would be *tying the extracted winding rate to a physical constant on a bio-model* — but the codon
  papers already did exactly that for Z3.

**Kill-check bottom line:** phase-velocity / winding-against-a-physical-period as an interpretability
validator is **no longer conceptually open as of NeurIPS 2026** — it has been demonstrated (codon/Z3, with
causal frameshift). The specific *protein-helix* instance is unoccupied but derivative, and its advertised
job (rebutting "Spokes") is mis-specified.

---

## TASK 2 — MAKE THE TEST RIGOROUS

### Phase extraction for a 2D cyclic feature
Take a candidate 2D subspace per residue i: a BSF block of size b=2 (code z_i = (u_i, v_i) ∈ R²), or a 2D
PCA/probe plane. Phase φ_i = atan2(v_i, u_i).
- **Gauge caveat (load-bearing).** BSF pins only the *subspace*, not its axes (rotation-invariant only — see
  primer §1.2). So **absolute φ is gauge-dependent and meaningless**; the **phase VELOCITY Δφ_i =
  wrap(φ_{i+1} − φ_i) and the net winding W = (1/2π)Σ Δφ_i are rotation-invariant.** This is the one real
  methodological nicety of the pitch: the observable you care about (velocity, winding) is exactly the part
  immune to BSF's basis ambiguity. State this explicitly; it is the pitch's best technical selling point.
- Unwrap φ along the backbone within a contiguous secondary-structure segment; estimate slope by circular–
  linear regression of φ on index.

### Expected phase velocities (physical anchors, deg/residue or deg/nt)
- **α-helix:** 3.6 res/turn → **100°/res** (1.745 rad).
- **3_10-helix:** 3.0 res/turn → **120°/res**.
- **π-helix:** ~4.3–4.4 res/turn → **~82–84°/res**.
- **β-strand:** ~2.0–2.3 res/turn → **~160–180°/res** (the classic i,i+2 hydrophobic-face period-2).
- **Codon:** period-3 → **120°/nt** (matches the NeurIPS "120° rotation = 1 nt frame shift" exactly).
- **Frameshift reset:** a +1/−1 indel rotates the downstream codon-phase coordinate by ∓120° and the net
  winding over the ORF changes by ±1 turn per full-codon slip.

### The NULL it must beat
1. **Spoke/cluster null:** a discrete-state feature has **non-constant Δφ** (phase sits at a few values with
   jumps); circular–linear R² low, slope undefined/zero net winding. This is the Gupta picture — and note a
   *high* circular–linear R² with the *right* slope is what distinguishes a true screw from spokes **on the
   velocity axis only** (not on the "intermediate-value" axis; see Task 1c).
2. **i.i.d. / residue-shuffle null:** permute residues within a protein → destroys sequential coherence;
   slope→0, R²→chance. Gives the null distribution of the statistic.
3. **Phase-randomized surrogate:** preserve the power spectrum, randomize phase → kills any genuine phase
   locking while matching amplitude; separates "periodic power exists" from "phase advances coherently."
4. **Circular-shift / block-rotation null:** random matched-norm 2D directions (the BSF-appropriate control)
   → no signed slope.

### The POSITIONAL-CONFOUND TRAP (the one that matters)
ESM3-open uses **RoPE** (rotary) in place of absolute PE (confirmed from the ESM3 supplement: "rotary
embeddings in place of absolute positional embeddings," Pre-LN, SwiGLU; geometric attention only in block 1).
**RoPE literally rotates 2D channel-pairs by an angle linear in the position index at fixed per-channel
frequencies.** Therefore a 2D subspace can show a perfectly constant phase velocity that is **just the index**,
with zero learned biology. With hundreds of RoPE frequencies, some pair sits near any target rate (incl. near
100°/res) by chance → **false-positive machine**. Rule it in/out with, in increasing strength:
1. **Helix vs coil at matched indices.** RoPE's phase advance is **identical** for helix and coil at the same
   positions; only a biology-driven velocity differs. A ~100°/res slope that appears in helix residues but
   **not in coil residues at the same sequence indices** is biology, not RoPE. (Primary go/no-go.)
2. **Trained − untrained with the SAME RoPE.** Run an **untrained / random-weight ESM3** with identical RoPE
   to isolate the pure positional contribution; the **trained-minus-untrained gap at the helix frequency** is
   the learned part. (This is also your Heap-&-Aitchison random-baseline table-stakes control.)
3. **Sequence-only input** (no structure tokens), so helix phase can't be trivially copied from the structure
   channel.
4. **Phase must track the STRUCTURE-derived helical phase, not index-mod-3.6.** Real helices have kinks,
   bulges, 3_10 caps, variable rise → **real phase slips off the ideal 100°/res**. Compute each residue's true
   helical phase from the coordinates (screw-axis projection) and test **partial** circular correlation of
   feature-phase with structure-phase **after regressing out index-mod-period**. A feature that follows the
   *deviations* (kink → local velocity change) is doing biology RoPE/periodic-PE structurally cannot fake.
   **This is the single strongest discriminator** and should be the key figure if the pitch survives.

---

## TASK 3 — SMALL DECISIVE TEST (1 GPU, hours–days)

**Do the protein leg only.** The genomic leg is scooped (Task 1a); if you touch codon at all, frame it as a
*replication* of the NeurIPS codon-frame result with citation, not a contribution.

### Protein: ESM3-open (esm3-sm-open-v1, 1.4B), α-helix phase advance
- **Data:** ~2–5k single-domain chains with experimental structures; DSSP labels for H (α), G (3_10), E
  (strand), coil. Compute per-residue **structure-derived helical phase** (screw-axis projection) for every
  helix. Hold out from any annotation matching.
- **Model pass:** sequence-only input, cache activations at ~3 mid/late layers (sweep, e.g., L18/L30/L40 of
  48). Also cache an **untrained-weights** pass (same RoPE) for the positional baseline.
- **Candidate subspace:** (a) a BSF block of size 2 trained on cached activations (group_size=2), or cheaper
  (b) 2D PCA of residual within helical windows / a 2-output probe to [cos,sin] of structure-phase. Discover,
  then test velocity — do **not** fit the slope you then "confirm."
- **Primary metric:** circular–linear correlation of unwrapped feature-phase vs residue index within helices,
  and **slope (deg/res) with bootstrap CI**; compare to 100 (α) and 120 (3_10).
- **Causal (optional, cheap):** masked-generative re-run — rotate the 2D coordinate by +100°/res along a
  helix and check the structure/sequence readout tracks; matched-norm random-direction control. (Mirrors the
  codon-steering move; shows *use*, not just presence.)

### GO / NO-GO GATES
- **G1 (rate):** α-helix slope = 100 ± ~15°/res, 3_10 ≈ 120°/res, with CI excluding 0 and excluding the coil
  slope. 
- **G2 (not RoPE):** helix slope ≫ coil slope **at matched indices**, AND trained−untrained gap at the helix
  frequency is significant. **If G2 fails, the pitch is dead** (positional artifact).
- **G3 (biology not index-mod-period):** partial circular correlation of feature-phase with *structure*-phase
  > 0 after removing index-mod-3.6; helices with kinks/3_10 show matching velocity deviations.
- **G4 (causal, if run):** rotation by the physical rate moves the readout monotonically and returns at 2π;
  random direction does not.
- **STOP conditions:** signal equal in coil or in untrained model (→ RoPE artifact); slope not matching
  physics (→ not a helix-phase feature); feature-phase tracks only index-mod-3.6 and not structure deviations
  (→ trivial positional count, not learned geometry).

Compute: activation caching on 2–5k chains + a size-2 BSF/probe on one GPU is ~tens of GPU-hours; untrained
baseline doubles the cache. Well within "hours–days, 1 GPU."

---

## TASK 4 — NOVELTY, KILLERS, SCOOP

### Is the phase-velocity signature a KEY ICML figure, or a sanity check?
**Closer to a sanity check / one validator panel than a headline.** As of NeurIPS 2026 the *method* (2D
cyclic subspace → winding coordinate → rotate by the physical per-step angle → causal shift of a biological
readout → cyclic-group/modular consistency + matched-control null) is **published** (codon/Z3). The
protein-helix version is a clean, physics-anchored *instance*, and the gauge-invariance point (velocity/
winding is immune to BSF's rotation ambiguity) is a nice technical hook — but a reviewer who knows the codon
papers will read it as "the codon result redone for helices." **Recommended role:** fold it in as the
**causal validator panel** of the BSF/manifold-recovery benchmark (the flagship Pitch 1+3 paper), where
"feature advances at the physical helical rate, tracks structure-derived phase slips, and a matched-norm
rotation steers the readout" is a strong *supporting* figure — not the paper's thesis.

### Two most likely killers
1. **Positional artifact (RoPE).** The phase velocity is inherited from rotary PE, not learned biology.
   Mitigated by G2 (helix-vs-coil at matched index; trained−untrained) and G3 (structure-phase slips), but if
   those controls come back null, there is no result. This is the primary existential risk and must be run
   **first**.
2. **Category error vs "Spokes, not a Ring."** Phase-velocity does not answer Gupta et al.'s actual
   objection (are the *between-state arcs* meaningful?). Advertising it as "the causal rebuttal to Spokes" is
   wrong and a reviewer who read Gup26b (NeurIPS 2026, same cycle) will say so. Reframe honestly: phase-rate
   is evidence the states form a *screw at the physical pitch*, a **different and complementary** claim; the
   Spokes rebuttal needs the intermediate-phase in-distribution causal test. Secondary: **Feu26 / decodable≠
   causal** — helix phase may be present but unused, so you need the G4 causal leg to claim it matters.

### Scoop exposure
- **Codon/genomic: already scooped (HIGH).** Two NeurIPS 2026 papers; Kovalic/Zhang/**Golkar**/Ho and a
  second group. Do not compete here.
- **Method/winding framing: scooped (MEDIUM-HIGH).** The "emergent cyclic-group representation + causal
  rotation" template is now public.
- **Protein-helix instance: currently OPEN (MEDIUM).** But it is the obvious next target for (a) the same
  Golkar/Ho-style groups, (b) EvolutionaryScale/ESM3 insiders, (c) the Setlur "feature-manifolds" line, (d)
  Oxford/Crook. On a ~ICML-2027 timeline (full paper ~late Jan — your own FINAL_FINDINGS calls this
  unrealistic in ~9 weeks) this is live scoop risk. A human should check neurips.cc/virtual/2026 (blocked
  here) and goodfire.ai before relying on "open."

### Recommendation
- **Keep** the α-helix phase-velocity experiment as a **supporting causal-validator panel** in the BSF/
  manifold benchmark paper; lead with the gauge-invariance argument (velocity/winding beats BSF's rotation
  ambiguity) and the structure-phase-slip test (G3), which are the genuinely novel bits.
- **Drop** the codon/frameshift leg as a contribution; cite the two NeurIPS 2026 codon-frame papers as the
  thing that already did it (and as *positive evidence the approach works*, which actually helps your prior).
- **Rewrite** the "direct rebuttal to Spokes-not-a-Ring" claim — it is a category error; position phase-rate
  as a complementary screw-pitch check, and if you want to engage Gupta et al., budget the intermediate-phase
  in-distribution causal test instead.

### References (cite keys → locators)
Eng24 (arXiv 2405.14860) · Gup26b (arXiv 2610.37796, NeurIPS 2026) · Feu26 (arXiv 2605.01148) · Che12
(DOI 10.4236/CMB.2012.24011) · Set26 (Interpreting Latent PLM Features with Geometric Annotations) · Bes25
(Towards Understanding the Shape of Representations in PLMs) · Liu26f (ProtGenesis, DOI
10.64898/2026.02.20.706798) · Kes26 (Induction Meets Biology) · Sil09 (Persistent Cohomology & Circular
Coordinates) · Gra26d (Selecting Interpretable Circular Coordinates) · NeurIPS 2026 codon-frame papers
(neurips.cc/virtual/2026/160431 and /157321 — full text BLOCKED this session, verify when whitelisted) ·
ESM3 arch: Hayes et al. ESM3 supplement (RoPE, geometric attention block-1 only).
UNVERIFIED flags: NeurIPS codon-paper method depth (abstract only); ESM3 exact RoPE channel frequencies;
whether any codon paper also tested proteins.
