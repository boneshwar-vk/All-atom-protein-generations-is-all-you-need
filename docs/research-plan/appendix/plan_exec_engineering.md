# EXECUTION / ENGINEERING PLAN — causal + identifiable + truth-vs-confidence on exact physical ground truth

Role: engineering / execution lead. Planning only — **no git repository changes**. Date 2026-11-22.
Authoritative definition: `PROJECT_BRIEF.md`. This plan operationalizes the pivoted paper from
`PITCH_VETTING_consolidated.md` §"consolidated surviving paper" and the staged gates in
`plan_task1_geometry.md`, `plan_task3_steering.md`, `FINAL_FINDINGS.md`.

Framing guardrails carried verbatim from the brief:
- The contribution is **IDENTITY + CAUSE + TRUTH-vs-CONFIDENCE on exact physical ground truth**, NOT "a fancier
  dictionary geometry". Geometry-matched vs scalar-SAE vs matched-random is a **control inside the paper**.
- Track A (angles) = low risk, benign. Track B (binding) = **defensive/auditing framing, benign complexes only,
  no affinity-maximizing pipeline released, no misuse guidance.** ipSAE is a **contrast, never a steering target**.
- Compare at **matched bits / MDL**, never matched feature count. Controls are mandatory, not optional.

Verification tags: **[CONFIRMED]** = checked a primary/official source this session; **[UNVERIFIED]** = could not
confirm from a primary source and must be verified by a human before relying on it.

---

## 0. TOOLING / SOFTWARE LEDGER (verify before Phase 0)

| Component | Source | Status |
|---|---|---|
| ESM3-open `esm3-sm-open-v1` (1.4B, 48 layers) | `pip install esm`; HF `EvolutionaryScale/esm3-sm-open-v1` (gated, **non-commercial** licence; accept terms) | weights gated+non-commercial **[UNVERIFIED access]**; d_model≈1536 **[UNVERIFIED exact]** (primer states 1536) |
| Boltz-2 (co-folding + affinity) | `github.com/jwohlwend/boltz`, MIT licence; affinity head outputs log(IC50) | **[CONFIRMED]** open/MIT; Boltz-1 fallback has Jedryszek (`Jed26b`) SAE/intervention plumbing |
| ipSAE (interface confidence) | `github.com/DunbrackLab/IPSAE`; PyPI `ipsae` (1.0.1); consumes AF2/AF3/Boltz PAE+structure | **[CONFIRMED]** repo+PyPI exist; official-vs-fork path verify |
| geoopt (Riemannian opt) | `github.com/geoopt/geoopt` — Sphere `S^{b-1}`, ProductManifold (torus), PoincareBall | **[CONFIRMED]**; install from git (PyPI lags) |
| DSSP (mkdssp) | `libcifpp`/`dssp` v4; secondary structure, phi/psi, SASA | standard **[CONFIRMED tool exists]** |
| mdtraj | `pip install mdtraj` — dihedrals, SASA (Shrake-Rupley), contacts | standard |
| ProteinMPNN | `github.com/dauparas/ProteinMPNN` — refold/designability validity filter | standard |
| FreeSASA | `pip install freesasa` — ΔSASA/BSA interface descriptor (Track B primary target) | standard |
| PDBBind / CASF | `pdbbind.org.cn` (registration) — protein-ligand Kd/Ki/IC50; CASF-2016 benchmark | counts/licence **[UNVERIFIED]** — verify registration + redistribution terms |
| PLINDER | `github.com/plinder-org/plinder`, PyPI `plinder`; leakage-aware splits | **[CONFIRMED]** split machinery exists; >400k PLI systems claim **[UNVERIFIED]** |
| BindingDB | `bindingdb.org` — scale Kd/Ki/IC50 | licence/counts **[UNVERIFIED]** |
| SKEMPI 2.0 | `life.bsc.es/pid/skempi2` — protein-protein ΔΔG on mutation (built-in causal test) | counts **[UNVERIFIED]**; ~7k mutations claimed |
| TDA (Betti) | `ripser` / `gudhi` — persistent homology for topology gate | standard |

**Known scoop/landmine checks a human must close before writing** (from brief §landmines): Dal26 SASA (2606.06333),
Sko19 mixed-curvature VAE, Ste26 Steenwyk ESM3 L33 scalar steering, Zha26g off-manifold collapse (2608.18597),
Hin26 masked-pLM edit persistence, PairSAE Mig26 (2606.27440), Jed26b (2608.11475), Setlur 2608.26419, the two
NeurIPS-2026 codon papers, BioEmu+ESM3 SAE camera-ready **[UNVERIFIED]**, HyperSAE **[UNVERIFIED]**.

---

## 1. REPO / SCAFFOLD LAYOUT (built later in a real repo; shown for planning only)

```
bioman/                              # "biological manifolds, identifiable + causal"
  pyproject.toml                     # pinned deps; python 3.11; torch 2.x; geoopt@git
  configs/
    base.yaml                        # seeds, paths, dtype=fp16, device
    phase0_synth.yaml                # planted-manifold calibration
    trackA_esm3.yaml                 # layers, corpus, dict sweeps
    trackB_boltz.yaml                # (layer x timestep) grid, targets, splits
  bioman/
    data/
      pdb_fetch.py                   # RCSB download -> own empty dir per archive (untrusted)
      dssp_gt.py                     # mkdssp/mdtraj -> phi,psi,omega,helix-phase,relSASA
      interface_gt.py                # FreeSASA/mdtraj -> BSA, contacts (Track B geometric target)
      kd_tables.py                   # PDBBind/BindingDB/SKEMPI parse -> (complex, Kd, split)
      splits.py                      # PLINDER leakage-aware + time-split loaders
    cache/
      esm3_hooks.py                  # residual-stream capture, seq-only input, fp16 shards
      boltz_hooks.py                 # (layer x denoising-step) capture, single+iface-pair rep
    dict/
      scalar_sae.py                  # TopK SAE (b=1 baseline)
      block_bsf.py                   # Euclidean block (b=2,3) Vanilla/Grassmann/Group-Lasso
      circular_block.py              # S^1 (b=2) magnitude/direction split (geoopt)
      torus_block.py                 # T^2=(S^1)^2 (b=4) per-pair normalise (geoopt)
      curved_block.py                # S^2 / Poincare (stretch/appendix only)
      mdl.py                         # quantise codes -> bits; rate-distortion frontier
    probe/
      linear_mlp.py                  # ridge + 1-hidden-layer MLP probes, per site
      circular_regression.py         # circular/geodesic regression + circ-corr
      confidence_probe.py            # pLDDT / ipSAE direction; truth-vs-confidence angle
    topo/ betti.py                   # ripser/gudhi persistent homology + pre-registered Betti
    steer/
      esm3_steer.py                  # geodesic traversal, wrap-around, output-head readout
      boltz_steer.py                 # mid-diffusion injection, persistence to final structure
      controls.py                    # matched-norm random dir, output-only, local-window
      validity.py                    # pLDDT/pTM/clash + ProteinMPNN refold
      fk_baseline.py                 # Feynman-Kac particle steering (Track B must-beat)
    eval/ metrics.py registry.py
  experiments/ (phase0.., phaseA.., phaseB..)  # thin drivers calling bioman/
  artifacts/   (figures, tables, cached-manifest.json, MANIFEST of hashes)
  tests/       (unit: retractions, wrap-around identity, MDL monotonicity)
```

**Reproducibility / seed plan (minimal, non-negotiable):**
- One `base.yaml` seed list `[0,1,2,3,4]` (≥5). Every dict/probe trained across all 5; report **median + a
  stability statistic** (seed variance of recovered coordinate; Tournament-Top-K `Jer26` for block selection).
- Deterministic cudnn where feasible; log `torch`, `esm`, `boltz`, `geoopt`, CUDA versions into each run's
  `run_meta.json`. Content-hash every activation shard and ground-truth table into `MANIFEST`.
- **Random-init same-architecture control model** instantiated once per track (Heap-Aitchison trap): every headline
  number reported as `trained − random-init` gap, not absolute.
- Fixed held-out splits frozen before any fitting: Track A = sequence/cluster split (MMseqs2 30% identity);
  Track B = PLINDER leakage-aware + a time-split (deposition date) — frozen and hashed.
- Config-driven; no magic numbers in code. Each phase emits a single `artifacts/<phase>/REPORT.md` + figures.

---

## PHASE 0 — SYNTHETIC MATCHED-BITS CALIBRATION ("cheapest test that can kill the project")

**Purpose.** The one test to run first (brief §"ONE test to run first", predicted to TIE). Pure synthetic, **no
model weights, ~1–2 CPU-hours.** If curved/circular blocks only TIE a flat Euclidean block of equal dim at matched
bits, the "geometry is the headline" idea is confirmed dead → we **commit to the identity+causal+truth-vs-confidence
paper and do NOT build curved-dictionary machinery** (saves weeks). This phase also hardens the dictionary + MDL +
Betti code paths that every later phase reuses.

**Steps.**
1. Plant manifolds in `R^64` with controlled ambient embedding dim and additive Gaussian noise (SNR sweep):
   circle `S^1`, torus `T^2`, sphere `S^2`, swiss-roll (non-periodic curved), and a monotone scalar ramp (the
   "mismatched" control variable). Each as `N≈50k` points. Also a **matched-bits linear-embedding-dim sweep**:
   embed each manifold at linear dim `m ∈ {intrinsic+1, intrinsic+4, ... , 64}` — the brief's one caveat is that
   curved could win only when `m >> intrinsic+1`.
2. Fit, at **matched description length (bits)**: scalar TopK SAE (k=1,2) · Euclidean BSF block (b=2,3,
   Vanilla+Tournament-TopK) · circular `S^1` block (b=2) · torus `T^2` block (b=4) · (optional) `S^2`, Poincaré.
3. Metric: atoms-to-`FVU ≤ ε` **and** rate–distortion frontier (quantise codes → bits; plot FVU and
   circular/toroidal-R² vs bits). Pre-registered Betti check per recovered point cloud (circle (1,1,0),
   torus (1,2,1), sphere (1,0,1)).

**What gets trained.** All dictionaries above; probes are trivial here (ground truth is the planted coordinate).
Config sweep: `b∈{1,2,3,4}`, `l0∈{1,2,4}`, `λ_sparse` 5-pt log grid, 5 seeds, SNR∈{∞,20,10,5}dB, `m` sweep.

**Intervention machinery.** None (no model). But implement + unit-test the **wrap-around identity**: traverse the
recovered `S^1` coordinate 0→2π and assert reconstruction returns to baseline — the same routine Track A reuses.

**Metrics.** FVU@matched-bits; circular-corr (Jammalamadaka–Sarma) / toroidal-corr; MDL frontier dominance;
Betti match; seed variance.

**Compute.** 24GB tier only / CPU: **1–3 CPU-hours.** No GPU, no weights.

**Deliverable.** `artifacts/phase0/REPORT.md` + the rate-distortion frontier figure + a **frozen prediction**
("curved ties flat at matched bits except for `m≫intrinsic+1`") logged with a timestamp (pre-registration).

**GATE P0.**
- **TIE (predicted):** curved/circular ≈ Euclidean-block-of-equal-dim at matched bits → **geometry headline dead;
  the curved dictionary is demoted to a within-paper control; proceed to the identity+causal+truth paper.** The
  circular/torus block is still used in Track A as *one of several* candidate parameterizations of the recovered
  coordinate, judged by identifiability, not by "beats flat".
- **STRICT WIN** (only on manifolds with `m≫intrinsic+1`): note it as a bounded secondary result; still do not make
  it the headline.
- Either way **GO** to Phase 1 — Phase 0 only decides whether "geometry" is a headline or a control.

---

## PHASE 1 — PERSISTENCE / COMMITMENT GATE (G0) ("most likely death; test cheapest-first")

**Purpose.** Brief pillar 4 + both sub-plans flag edit **wash-out** as the single most likely killer of the causal
half (Zha26g off-manifold collapse, Hin26 masked-pLM persistence, Jed26b decodable≠steerable). Before ANY
identity or dose-response work, answer: **does a mid-computation activation edit survive to the final output?** Run
both tracks here because each has a different decode loop and either can die independently.

### 1A — ESM3-open (iterative unmasking persistence)
**Steps.** (a) Cache residual stream at **one mid-depth layer** (start L≈24 of 48; also probe the Steenwyk L33
collision site) for ~1–2k sequence-only proteins. (b) Fit a **cheap scalar probe direction** for a crude periodic
coordinate (helix phase) just to have *a* direction to push — identity comes later. (c) During ESM3's
iterative unmasked generation, add `α·d` at the chosen layer at an intermediate decode step; let decoding finish;
recompute the realized structural quantity from the generated structure tokens (decode to coords; DSSP). (d)
Subtract matched-norm random-direction band.

**Metrics.** Realized final-output shift vs `α` (monotone?); z-score vs matched-random band; fraction of injected
signal surviving decode (|final Δ| / |immediate Δ|).

**GATE G0-A.** Realized final shift clears matched-random band (|z|>2) with monotone `α` at ≥1 layer/step →
**GO**. Null everywhere → **persistence fails**; the causal/steering half of Track A becomes a *negative/calibration
result* (honestly reported, extends Hin26/Zha26g to ESM3 structure tokens) and the paper leans on identity +
truth-vs-confidence. **This is a publishable outcome, not a dead end.**

### 1B — Boltz (mid-diffusion persistence)
**Steps.** (a) For N≈100 benign complexes (filtered per §Ethics), run the sampler caching the **single
representation** at a coarse `L×T` grid (start 3 layers × 4 denoising steps). Prefer single-rep + an
interface-residue subset of the pair rep to bound storage (see caching plan). (b) Fit a cheap ridge direction for
**BSA** (model-independent geometric target). (c) Inject `α·d` inside one reverse diffusion step; continue
denoising; recompute **final-structure** BSA by FreeSASA. (d) Matched-norm random band + ProteinMPNN/clash validity
filter (so a "shift" is not off-manifold garbage).

**GATE G0-B.** Final-structure BSA clears the random band with monotone `α` at ≥1 cell → **GO**. Null everywhere →
diffusion washes out intermediate edits; Track B pivots to output-only steering + the decodable≠steerable map as a
negative result (extends Jed26b from "which layer" to "which (layer,step)").

**Compute (Phase 1).** 24GB tier feasible for 1A (ESM3 1.4B fits); 1B needs **1×80GB** comfortably.
Order-of-magnitude **[UNVERIFIED]**: 1A ≈ 30–80 GPU-h; 1B ≈ 80–200 GPU-h (per-timestep sampling dominates).

**Deliverable.** `artifacts/phase1/` persistence heatmaps (layer×step, both tracks) + the GO/NO-GO decision that
routes the rest of the project. **This phase decides whether the paper is "causal + identity + truth" or
"identity + truth + an honest persistence negative".**

---

## PHASE 2 — REAL-MODEL IDENTITY (up to isometry) + TRUTH-vs-CONFIDENCE

**Purpose.** Brief pillars 1 & 3 (identity up to isometry; physics validator) + the truth-vs-confidence axis. Show
the recovered coordinate **IS** the physical quantity (monotone/rotation-invariant alignment on a held-out split),
not merely a reconstructing subspace — and that the model keeps the **true physical quantity in a different internal
direction from its own confidence**. No causal claims yet.

### 2A — ESM3-open: (phi,psi) torus, helix phase; pLDDT as confidence contrast
**Activation-caching plan.**
- **Input: sequence-only** (structure track OFF — else dihedrals are trivially present and the claim is void).
- Corpus: ~5k experimental-structure PDB chains, MMseqs2-clustered to ≤30% identity, time-split; ~200 res each
  ≈ **1M residue tokens**.
- Sites: residual stream **post-block** at **8 layers** spanning depth (e.g. 6,12,18,24,30,36,42,48) for the
  layer-resolved identity map; fp16.
- **Storage:** 1M tok × 1536 × 2 B = **3.07 GB/layer** → ~**24.6 GB** for 8 layers (disk; stream shards to GPU).
- Ground truth from **DSSP/mdtraj on experimental coords** (NOT ESM3): phi, psi, omega, helix-phase (res index
  mod 3.6 within DSSP H), **relative SASA (non-periodic control)**. Confidence = ESM3 pLDDT / structure-token
  entropy, recorded per residue.

**What gets trained.** Per layer: scalar TopK SAE (b=1) · Euclidean BSF (b=2,3; Vanilla/Grassmann/Group-Lasso,
Tournament-TopK) · circular `S^1` (b=2) · **torus `T^2` (b=4)** · linear ridge probe · 1-hidden-MLP probe ·
confidence probe (direction that predicts pLDDT). Sweep `b,l0,λ`, 5 seeds.

**Identity metrics (the headline evidence).**
- **Rotation-invariant alignment:** circular-corr(recovered angle, true phi/psi) and toroidal-corr on **held-out**
  proteins; geodesic-R² from geodesic regression on the matched manifold metric.
- **Betti pre-registration** (anti-`Gup26b` spokes-not-ring): persistent homology of recovered cloud must match
  torus (1,2,1)/circle (1,1,0) **AND** show a trained−random-init gap (pre-registered circular-R² gap > 0.1).
- **Physics validator (RoPE-proof):** feature tracks **structure-derived phase slips** (kinks, 3-10 helices,
  omega≈180° cis/trans) that positional encoding cannot manufacture. **Positional/local-window control runs FIRST:**
  if a 1-hot sequence-index window predicts the phase as well as the block code, the geometric claim is dead.
- **Truth-vs-confidence (new axis):** angle between the recovered phi/psi direction and the pLDDT/confidence
  direction; show they are **separated** (physical coordinate ≠ model self-confidence about it). Report where in
  depth they diverge.

**GATE G1-A.** On held-out proteins: (i) block code recovers the torus/phase at circular-R² above scalar-SAE and
above the local-window baseline, with trained−random gap > 0.1 and correct Betti; (ii) phase-slip panel shows the
feature tracks structure irregularities beyond index-mod-3.6; (iii) truth and confidence directions are measurably
distinct (angle bounded away from 0). **All three → GO.** (Note: scalar-SAE may also decode the torus by
*fragmenting* it across many latents — that is expected and is itself a reported finding; the identity claim is
about recovering it as ONE gauge-free coordinate, judged by circular-corr + Betti, not by "beats scalar".)

### 2B — Boltz: experimental Kd identity; ipSAE/ipTM as confidence contrast
**Activation-caching plan.** Reuse Phase-1B single+interface-pair cache; extend to N≈300–500 benign complexes with
measured Kd (PDBBind/BindingDB) on the **frozen PLINDER-leakage-aware + time split**; cache the trunk/pairformer
layers at the final (or a few) denoising steps. Also compute, per complex: **BSA/contacts (geometric target,
model-independent)**, ipSAE + ipTM (DunbrackLab/IPSAE on Boltz PAE), and Boltz's own predicted affinity.

**What gets trained.** Ridge + MLP probes `activation → {experimental Kd, BSA, ipSAE, Boltz-predicted-affinity}`;
scalar SAE + block dictionaries on pair reps; a **confidence probe** for ipSAE/ipTM.

**Identity metrics.**
- **Kd identity:** held-out R²/Spearman of probe → **experimental** Kd on the leakage-aware split; CASF ranking
  power as external benchmark. Primary **geometric** target BSA for clean readout independence.
- **Truth-vs-confidence (crown of Track B):** show the internal direction that predicts **experimental Kd** is
  **distinct from** the direction that predicts **ipSAE/ipTM** and distinct from the direction predicting
  **Boltz's own predicted affinity** (the PairSAE circularity trap). Report the three-way angle matrix. This is the
  "truth in a different direction from confidence" pillar made concrete and is the differentiator from PairSAE/Mig26.

**GATE G1-B.** On the leakage-aware held-out split: (i) experimental-Kd probe beats a sequence-length/size-only
null and does NOT collapse to the model's predicted-affinity direction (angle bounded away from 0); (ii) BSA probe
is clean under readout independence; (iii) ipSAE direction is separable from the experimental-Kd direction.
**→ GO.** If the only recoverable direction is the model's own predicted affinity (Kd indistinguishable from
confidence), report the **circularity/Goodhart negative** (strengthens the honest-benchmark framing).

**Compute (Phase 2).** Caching + fitting. 24GB: Track-A identity fits comfortably (ESM3 cache streamed; dict/probe
training is cheap). Track-B needs 1–2×80GB for the extra sampling. **[UNVERIFIED]** 2A ≈ 100–250 GPU-h;
2B ≈ 150–400 GPU-h.

**Deliverable.** `artifacts/phase2/` — the **layer-resolved identity map** + Betti/phase-slip panels (A) and the
**three-way truth-vs-confidence angle matrix** (B). These are the paper's identifiability + truth-vs-confidence
figures.

---

## PHASE 3 — CAUSAL (wrap-around dose-response; SKEMPI mutation agreement)

**Purpose.** Brief pillar 2. Only reachable if Phase-1 persistence passed for the track. Steer the recovered
coordinate and show the output follows — and **returns to baseline at 2π** (a test a 1-D latent structurally
cannot pass). Readout must be the **model's OWN output head**, not a same-variable probe.

### 3A — ESM3: wrap-around dose-response on the (phi,psi)/phase coordinate
**Steps.** (a) Pick the layer with best identity+persistence from Phases 1A/2A. (b) Geodesic traversal of the
recovered circular coordinate 0→2π (geoopt exp-map on `S^1`); at each step inject into the residual stream during
generation. (c) Readout on ESM3's **own** masked-AA / structure-token logits (not the training probe). (d)
Controls: matched-norm random direction; output-only steering; local-window. (e) Validity filter: pLDDT/pTM,
steric clash, **ProteinMPNN refold designability** so a "steered" structure is on-manifold.
**Metrics.** Monotone dose-response over 0→π; **wrap-around closure** (output distance back to baseline at 2π,
vs the flat-code control that cannot close the loop); z vs random band; % passing validity.
**GATE G2-A.** Output follows the toroidal coordinate, exceeds the random band, **and closes at 2π** where a
flat/scalar code does not → **GO** (this is the crown-jewel causal figure).

### 3B — Boltz: monotone Kd/BSA dose-response + SKEMPI mutation agreement
**Steps.** (a) Mid-diffusion injection at the best cell from 1B/2B; α-sweep; recompute final-structure BSA
(primary) and record ipSAE/predicted-affinity (secondary, to expose Goodhart). (b) **SKEMPI built-in causal test:**
for complexes with measured ΔΔG on single mutation, check that the model's internal Kd coordinate moves in the
**sign/direction** SKEMPI reports when the mutation is applied — a label-independent causal check the model did not
train on as a steering target. (c) Must-beat baseline: **Feynman-Kac particle steering (`Har25c`)** at matched
reward-eval budget; plus matched-norm random + output-only.
**Metrics.** Monotone realized ΔBSA(α) clearing random band + surviving validity; **SKEMPI directional agreement
rate**; Goodhart gap (does steering move only BSA, or also the held-out ipSAE?); interpretability-site vs probe-site
vs output vs FK at matched compute.
**GATE G2-B.** Monotone persistent ΔBSA beating random + SKEMPI directional agreement above chance → **GO**. If FK
dominates at every budget, report honestly: "internals explain *where* binding strength lives; resampling is the
practical tool" — still publishable.

**Ethics guard (Track B, enforced in code):** primary target is the **geometric** descriptor (BSA/contacts), not
affinity; benign complexes only (exclude toxins / viral fusion-entry / immune-evasion / select-agent organisms via
a hard allow/deny list in `data/splits.py`); **no released affinity-maximizing steering pipeline**; ipSAE is a
contrast readout only. Dual-use section requires a human/compliant pass (brief §ethics — a safeguard has repeatedly
cut it).

**Compute (Phase 3).** The intervention sweeps dominate. **[UNVERIFIED]** 3A ≈ 100–250 GPU-h (1–2×80GB);
3B ≈ 300–700 GPU-h pilot, up to 1.5–3k GPU-h full (per-cell = N full samplings × α × seeds + FK baseline).

**Deliverable.** `artifacts/phase3/` — wrap-around dose-response figure (A); monotone-dose + SKEMPI-agreement +
Goodhart-gap + FK-comparison figures (B).

---

## PHASE 4 — CROSS-TRACK SYNTHESIS, FULL SWEEP, ROBUSTNESS

**Purpose.** Tie the two tracks into **one method, two quantities**: "do bio foundation models causally use the
known structure of a physical variable, recoverable as an identifiable gauge-free coordinate, kept distinct from the
model's confidence?" Full controls, seed ensembles, generality panel.

**Steps.** (a) Full dictionary × probe × 5-seed × Tournament-TopK sweep on both tracks; MDL/rate-distortion
frontier reported for every headline claim (matched **bits**). (b) Random-init same-architecture control for every
figure. (c) One **generality panel** for Track A (e.g. a second periodic variable or a second ESM3 layer region) to
show the identity result is not layer-cherry-picked. (d) Unify the truth-vs-confidence axis across tracks: pLDDT
(A) and ipSAE (B) both shown orthogonal-ish to the physical coordinate. (e) Negative-result bookkeeping: any failed
gate becomes a calibration finding, not a hidden result.

**Metrics.** Seed-stability statistics; frontier dominance; trained−random gaps; a single cross-track summary table.

**GATE G3.** Results survive 5 seeds + matched-bits + random-init controls on at least one track end-to-end
(identity→causal) OR identity+truth-vs-confidence on both tracks with an honest persistence negative. **→ write-up.**

**Compute.** Small-cluster tier: full seed ensembles + all variants + both tracks **[UNVERIFIED]** ≈ 4–8k GPU-h.

**Deliverable.** Camera-ready figure set + `cached-manifest.json` + a reproducibility appendix (seeds, hashes,
versions) + the dual-use statement (human-passed).

---

## 2. ACTIVATION-CACHING PLAN (consolidated)

| | Track A (ESM3-open) | Track B (Boltz) |
|---|---|---|
| Input | **sequence-only** (structure track OFF) | sequence + ligand/second chain |
| Sites | residual stream post-block, 8 layers spanning depth; add L33 (Steenwyk) | pairformer/trunk layers + diffusion module; **single rep + interface-residue pair subset** |
| Axis | layer | **layer × denoising-timestep** (the expensive axis) |
| Corpus | ~5k chains ≈ 1M residue tokens (identity); 1–2k for persistence pilot | N≈100 (persistence) → 300–500 (identity) benign complexes |
| dtype | fp16 | fp16 |
| Storage | 3.07 GB/layer; ~24.6 GB for 8 layers | single-rep ≈ 0.2 MB/complex/cell → ~2 GB for 6×8×200; **avoid full pair tensor** (≈23 MB/complex/cell → ~220 GB) — cache interface subset only |
| Ground truth | DSSP/mdtraj on experimental coords: phi,psi,omega,helix-phase,relSASA | FreeSASA BSA/contacts (primary); PDBBind/BindingDB Kd; SKEMPI ΔΔG; ipSAE/ipTM + Boltz-pred-affinity (contrasts) |
| Confidence | pLDDT / structure-token entropy | ipSAE / ipTM |

Caching is a **one-time** cost per corpus; all dictionaries/probes train on the cache (no grad through the big
model). Interventions re-run the model and are the real compute sink — hence persistence (Phase 1) is gated first.

---

## 3. ENGINEERING SIMPLIFICATIONS + ADDED CONTROLS (the "evolve" asks)

**Simplifications that cut cost/risk without weakening the claim:**
1. **Demote the curved dictionary to a control from day one.** Phase 0 is predicted to tie, so build the circular/
   torus block only as a *parameterization option* inside the identity analysis, not as a trained-from-scratch
   competing method. Saves the geoopt/Poincaré boundary-instability engineering the geometry plan warns about.
2. **Projected-gradient (renormalise) retractions instead of full geoopt exp/log** for `S^1`/`T^2` — simpler,
   usually sufficient; only reach for geoopt if projected-grad underfits (per `plan_task1_geometry` §3.1).
3. **Track B: cache the single representation + interface-residue pair subset, not the full pair tensor** — ~100×
   storage reduction (2 GB vs 220 GB) with no loss for an interface-property study.
4. **Reuse one dictionary/MDL/Betti/wrap-around code path across Phase 0 and both real tracks** — Phase 0 is also
   the integration test for that shared code, so later phases inherit tested components.
5. **BSA (geometric, model-independent) as the primary Track-B target, Kd second, ipSAE never a target** — gives
   clean readout independence and sidesteps the leakage/Goodhart minefield for the first decisive result.

**Added controls that strengthen the result (beyond the brief's mandatory set):**
1. **Confidence-ablation cross-check:** after showing truth ≠ confidence directions, *ablate* the confidence
   direction and show the physical coordinate (and its causal effect) **survives** — proving the physical feature is
   not a re-encoding of pLDDT/ipSAE.
2. **SKEMPI as an in-dataset causal oracle for Track B** (brief flags it; make it a first-class gate): mutation
   sign-agreement is a label the model never steered toward — the cleanest non-circular causal check available.
3. **Phase-slip panel as the RoPE killer** (brief pillar 3): pre-register that the feature tracks cis-omega/3-10/
   kinks; a positional artifact cannot, so this doubles as existence + anti-`Gup26b` evidence.
4. **Length/size confounds explicit:** regress out chain length and total SASA before any BSA/Kd claim (both plans
   flag entanglement with size).
5. **Pre-registered frozen predictions** (Phase 0 tie; Betti numbers; gap thresholds) time-stamped before fitting —
   turns reviewer skepticism about post-hoc geometry into a strength.
6. **Matched-bits MDL frontier for every headline**, not just the geometry claim — the one test a geometry-literate
   reviewer will demand, and it protects the identity claim too.

---

## 4. CRITICAL PATH + EFFORT HOTSPOTS

**Critical path (longest dependency chain to a complete claim):**
`Phase 0 (tie → commit)` → `Phase 1 persistence GO/NO-GO` → `Phase 2 identity + truth-vs-confidence` →
`Phase 3 causal (only if persistence passed)` → `Phase 4 full sweep + write-up`.
Track A and Track B run **in parallel** from Phase 1 onward; the paper needs **≥1 track end-to-end
(identity→causal)** plus **truth-vs-confidence on both**. Phase 1 is the true fork in the road: a persistence
NO-GO reshapes the paper into identity + truth + honest negative (still publishable per `FINAL_FINDINGS` §0).

**The 3 phases most likely to consume the most effort:**
1. **Phase 3B — Boltz mid-diffusion causal + SKEMPI + FK baseline.** Per-cell = N full samplings × α × seeds, plus
   the Feynman-Kac must-beat baseline at matched budget, plus the validity (ProteinMPNN refold) and SKEMPI
   plumbing. Highest GPU-h (up to 1.5–3k) and the most moving parts; also the dual-use-sensitive one.
2. **Phase 2B — experimental-Kd identity on leakage-aware splits.** The data engineering (PDBBind/BindingDB/
   SKEMPI parsing, PLINDER + time splits, de-duplication, the benign allow/deny list) and the three-way
   truth-vs-confidence disentanglement (experimental Kd vs ipSAE vs Boltz-predicted-affinity) are fiddly and
   leakage-fragile — the place a subtle split error silently inflates everything.
3. **Phase 1 (both tracks) persistence caching.** Per-timestep Boltz caching (storage + sampler instrumentation)
   and ESM3 iterative-unmasking injection are the first time the intervention machinery meets a real decode loop;
   expect debugging of hooks, decode-time injection points, and validity filters here before anything downstream
   works.

**Secondary effort risks:** ESM3-open gated-weight access + non-commercial compliance **[UNVERIFIED]**; geoopt
Poincaré boundary blow-ups if the curved stretch-goal is ever pursued; dataset licence/redistribution terms for
PDBBind/BindingDB **[UNVERIFIED]**; and the human/compliant pass required for the dual-use section.

---

## 5. OPEN ITEMS A HUMAN MUST CLOSE BEFORE EXECUTION
- Confirm ESM3-open weight access + non-commercial licence terms for the intended use **[UNVERIFIED]**.
- Confirm PDBBind / BindingDB / SKEMPI / CASF counts, licences, redistribution **[UNVERIFIED]**.
- Read full texts of the scoop list (Dal26 SASA, Ste26 Steenwyk, Jed26b, Zha26g, Hin26, Mig26, Setlur) + verify
  BioEmu+ESM3 SAE camera-ready and HyperSAE **[UNVERIFIED]** before committing headline framing.
- Human/compliant pass on the Track-B dual-use statement (benign-only scope, no affinity-max release).
- Verify ESM3 d_model (1536?) and exact hookable residual-stream sites; verify Boltz single/pair rep dims for the
  storage math.
```
