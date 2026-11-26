# Plan 1 (complete) — Layer-resolved dissection of proteins: real features, model scores, and the depth law

**The full, detailed first-phase plan.** Reconciled from four independent expert passes (code architecture;
models/inference; study design; final verification) over the Phase-1 setup (`implementation.md`), the datasets
(`datasets.md`), and the core scientific plan (`ICML2027_CORE.md`). This document is the engineering+science
contract for Plan 1 as extended beyond backbone angles.

**Convention.** `[U]` = unverified (confirm at the model boundary before it gates a headline). `[V]` = verified
this session. Benign/defensive framing: co-folding work is probe-only / localization-first, no steering; the
specificity quantity stays a separate no-steering appendix.

**Verdict (final verification): FIX-FIRST (split).** Build the weight-free scaffold through PR6.5 now; the
must-fix list in §8 must be closed before any headline cell is computed or the pre-registration is frozen. A pass
of the un-fixed architecture would be a *false green* — it validates a raw-layer atlas, not the cross-model
normalized-depth law the paper stands on.

---

## 1. What Plan 1 is now

A **layer-resolved dissection** of a protein/complex dataset that asks, for every (model, layer), how well that
layer's activations encode each of two kinds of target, broken down by protein category:

- **Real features** (external ground truth): backbone dihedrals φ/ψ, secondary structure, SASA, Cα–Cα contacts,
  **experimental binding affinity Kd / ΔG**, fold stability ΔG, interface buried-surface-area / hot-spots.
- **Model-learned scores** (the model's own outputs): confidence — **pLDDT, pTM, ipTM, PAE, ipSAE** — and
  **affinity/binding scores** (Boltz-2's affinity module). Obtained by running sequences/complexes through
  structure-prediction models.

The raw object is a 4-tensor `A[category, target, model, layer]`. But the **deliverable is not the atlas** — the
atlas is the *instrument* behind one sharpened claim (§2).

---

## 2. The scientific framing (why this is more than a table)

**The claim (new, C7): truth is encoded at a different — earlier / more distributed — network *depth* than the
model's own confidence and affinity, and this holds with a consistent sign across model architectures.**

- The atlas → claim converter is a single derived statistic: **Δℓ = depth\*(truth) − depth\*(own-score)**, in
  **normalized fractional depth** (not raw layer index), where depth\*(·) is the decodability-peak / saturation
  depth. Headline = **Δℓ < 0 with consistent cross-model sign**, shown as a **forest plot** (the money figure);
  the full `category × target × layer` heatmap goes to an FDR-controlled appendix.
- This **adds a depth axis** to the plan's existing truth-vs-confidence *direction* axis (C3) and upgrades the
  cross-model invariance (C6) into a quantitative signed law.
- **Strongest reviewer objection:** "trivial artifact of a confidence head bolted on the end, and you fished
  thousands of cells." **Answer:** the own-score curve is only a *localizer*, never a reported finding; the
  reported numbers are the non-circular legs (truth-vs-score gap backed by the C3 causal double dissociation, and
  one model's internals vs **another** model's score); the sign is consistent across architectures (not a
  per-architecture head artifact); multiplicity is controlled by pre-registering the Δℓ family + a hierarchical
  model + greying under-powered cells. **NO-GO rule:** if Δℓ is not cross-model consistent, the law reverts to an
  appendix atlas and the paper stands on C1 + C3.

---

## 3. Targets

### 3.1 Real features (family = `real`; external ground truth only)
| Target | Level | Ground truth source | Model(s) it is probed in |
|---|---|---|---|
| φ/ψ dihedrals | residue | DSSP/mdtraj on experimental structure | ESM3 (seq-only), Boltz-2 |
| Secondary structure | residue | DSSP | ESM3, Boltz-2 |
| Relative SASA | residue | FreeSASA (non-periodic control) | ESM3, Boltz-2 |
| Cα–Cα contacts / distances | pair | experimental structure | Boltz-2 (pair rep) |
| Fold stability ΔG | domain | Tsuboyama mega-scale (experimental) | ESM3 |
| Binding affinity Kd / ΔG | complex | PDBBind/PLINDER (**experimental only**) | Boltz-2 |
| Interface BSA / hot-spots | pair/residue | structure / SKEMPI ΔΔG | Boltz-2 |

### 3.2 Model-learned scores (family = `model_score`; provenance-tagged columns, never labels)
| Score | Level | Where it comes from | Producing model(s) |
|---|---|---|---|
| pLDDT | residue/token | confidence head | Boltz-2, AF3, Protenix (ESM3 seq-only pLDDT `[U]`) |
| pTM / ipTM | global / chain-pair | confidence head | Boltz-2, AF3, Protenix |
| PAE / PDE | token-pair | confidence head | Boltz-2, AF3, Protenix |
| ipSAE | directed chain-pair + residue | **derived** (Dunbrack, from PAE+pLDDT+CIF) — one shared impl | all co-folders |
| affinity (log10 IC50) + binary prob | complex | Boltz-2 affinity module (`affinity_*.json`) | Boltz-2 only `[V]` |

**Hard rule:** experimental Kd/ΔG is a `real` target; the model's *predicted* affinity is a `model_score` column —
never used as a truth label (the PairSAE circularity mistake). When relating the two, partial out predicted affinity.

---

## 4. Models

### 4.1 Which models are activation-probed vs score-only (scope — fixes contradiction M2)
- **Activation-probed (we hook internal layers): ESM3-open and Boltz-2.** These carry the Δℓ law on the internals side.
- **Score-only columns: AlphaFold3 and Protenix** (and optionally Chai-1 / OpenFold-3). We ingest their output
  scores for the non-circular cross-model leg, but do **not** hook their internals.
  - AF3 is JAX/Haiku (no `forward_hook`) and its **weights are gated** (Google request form, non-commercial) — not
    openly runnable with hooks. It enters as scores only; where an "AF3-class" *runnable* model is needed, **alias
    to Protenix or OpenFold-3 (Apache)** with honest naming. `[V]`

### 4.2 Weights / licenses `[V unless noted]`
| Model | Weights | License | Role |
|---|---|---|---|
| ESM3-open (`esm3-sm-open-v1`, 1.4B, 48L) | gated HF | non-commercial Community | activation-probed |
| Boltz-2 | free download (code+weights) | MIT | activation-probed + scores + affinity |
| Chai-1 | free auto-download | Apache-2.0 | score-only (optional) |
| Protenix ("v2") | download `[C]` license conflict (PyPI Apache vs ByteDance commercial) — academic OK | score-only |
| AlphaFold3 | **gated** (request form, ~2–3 days), non-commercial | score-only |
| OpenFold-3 | open | Apache | clean AF3-class substitute (score-only, or activation if needed) |

### 4.3 Inference, scores, activations (from the models agent)
- **Scores:** Boltz-2 emits `confidence_*.json` (ptm/iptm/…), `plddt/pae/pde` npz, and `affinity_*.json`
  (`affinity_pred_value = log10(IC50 µM)`, lower = stronger; + binary prob) `[V]`. AF3 emits
  `summary_confidences.json` + per-token `confidences.json` (pLDDT, PAE); default `num_recycles=10`,
  `diffusion_samples=5` `[V]`. ESM3-open = single-chain, **no native PAE/ipTM**.
- **ipSAE:** `DunbrackLab/IPSAE`, computed post-hoc from PAE+pLDDT+CIF; per directed chain-pair and per-residue;
  accepts AF2/AF3/Boltz outputs → use it as the **uniform, model-agnostic interface score** (pin the d0 variant `[U]`).
- **Activation hooks:** PyTorch `register_forward_hook` on Pairformer + diffusion blocks for Boltz-2 (and ESM3);
  no source edits. Reference: Jedryszek's Boltz-1 released SAEs+activations+code and PairSAE (Boltz-2 Z at layer
  33/64, recycle 3). Dims: `d_pair=128`, `d_single=384`, Boltz trunk ≈48–64 layers `[U]`.
- **Determinism:** trunk reps are deterministic given inputs + recycles; **diffusion reps are seed/noise-dependent**
  → cache at the **final recycle**, pin recycle count + seed + identical noise across conditions; prefer trunk.
- **Which layers:** early blocks 0–15 = sequence/biochemistry; mid-late ~25–35 = pairwise geometry; avoid block 0
  (positional embeddings). The seq→pair pathway lives in the upstream MSA module, not the Pairformer.

---

## 5. The dissection table (study design + statistics)

### 5.1 Rows — protein categories (fixes M5: facets vs covariates)
- **Load-bearing facets (external annotation only, never a model output):** monomer vs complex; the 4 CATH classes.
- **Covariates — regressed out, NOT faceted:** sequence length / size; **MSA depth / N_eff**. (Faceting on these
  manufactures spurious effects.) A size-facet **negative control** is run to prove no artifact.
- **Appendix:** enzyme class. **Memorization control:** organism. **Integrity flag:** intrinsic disorder (no single
  φ/ψ truth; low pLDDT by construction — excluded from identity, used to sanity-check).

### 5.2 Columns & cells
Columns = {real features} ∪ {model scores}. Each cell = **best-depth decodability in normalized fractional depth**
+ trained−random-init gap + CI + an **underpowered flag**. Self-decode cells (a model's own score from its own
activations) are labelled `self_score_trivial` (localizer only).

### 5.3 The statistic and multiplicity (fixes M1, M-S7, M-H)
- **Normalized fractional depth** within comparable sub-stacks makes depth commensurable across encoder ESM3 vs
  diffusion co-folders; depth\*(score) is defined as commitment/saturation depth, validated on synthetic MPU-0.
- Unit of inference = **Δℓ** (~24 tests), not the ~2,300-cell heatmap. **Holm** on the Δℓ family + **BH-FDR q=0.05**
  on the exploratory heatmap. Fit a **hierarchical partial-pooling model** (not thousands of independent probes) —
  the single best multiplicity defense. Best-layer argmax is optimistic → **nested bootstrap over layer selection**
  + peak-region fallback for seed stability.
- **Minimum N:** ≥~100–150 non-redundant clusters per cell or grey it out; ~45–55 cases for the confidently-wrong
  AUROC (DeLong), counted per category **at split-freeze**.

### 5.4 The circularity doctrine (the crux; resolved + code-enforced)
A model's confidence/affinity score is computed *from* its own late-trunk activations, so "layer predicts own
score" is near-tautological. Legitimate comparisons only:
(a) **external truth vs own score** — the within-model Δℓ gap, backed by the C3 causal double dissociation;
(b) **one model's internals vs ANOTHER model's score** — the non-circular jewel (is confidence a shared
input-difficulty signal no single head explains?);
(c) **experimental Kd/ΔG vs model-predicted affinity**, predicted-affinity partialled out, never predicted-as-truth.
Enforced in code: the target registry carries `producing_model`; the sweep demotes self-decode to the labelled
trivial curve.

---

## 6. Code architecture

- **`ModelAdapter` ABC** (one file per model: `esm3`, `boltz2`, `protenix`, `af3`, `synthetic`) with
  `run / extract_scores / token_map / hook_layers / steer`; `RepSpec` (rep_kind ∈ {single, pair}, layers,
  forward protocol) and `TargetSpec` (family, level, rep_kind, isometry, producing_model, reduction) dataclasses.
  All unverified model internals are isolated inside the adapter, so nothing downstream changes when AF3/Protenix land.
- **Unified cache** keyed `(model, rep_kind, layer/step, unit∈{residue,pair,global}, category, split)`:
  single-rep = the Phase-1 fp16 memmap; **pair-rep = flat INTERFACE-SUBSET store** (not dense L×L) + `index.parquet`;
  `meta.parquet` is the join hub carrying the category + **provenance-stamped** score columns. Reader gains
  `iter_pairs` and `align(model_a, model_b)` on the residue intersection.
- **Normalized-depth reducer** + Δℓ computation + forest-plot/heatmap renderers.
- **Leakage lints:** (i) category source may never be a model output; (ii) **global invariant (M-H1): a cached
  score may never be used as a split, covariate, or nuisance variable**; (iii) all unsupervised fits train-only.

---

## 8. The consolidated plumbing contract (must-fix, severity-ranked)

Old 8-item contract (from `implementation.md`) **plus** the new multi-model hazards. Build the weight-free
scaffold first; close S1–S2 before freezing the pre-registration; close all before any headline cell.

**S1 — blockers (decidable now; the law is ill-posed without them):**
- **M1 normalized/fractional depth + Δℓ + saturation-depth.** The architecture must reduce on normalized depth,
  not raw layer index; implement depth\*(·) and Δℓ; validate on synthetic. *Blocks PR5.5/PR6.5.*
- **M2 scope = 2 activation-probed (ESM3, Boltz-2), AF3+Protenix score-only.** Re-scope PR7.x adapters. *Blocks PR7.*
- **M3 MSAs pre-built, hashed, frozen; add MSA-depth/N_eff to the score-nuisance set.** Live MSA servers break
  determinism and leak post-cutoff homologs. *Blocks any co-folder cache.*

**S2 — before any headline cell:**
- **M4 per-model post-cutoff held-out for score cells** (each model's training cutoff differs; memorization inflates
  own-score decodability).
- **M5 size / MSA-depth are covariates (regressed out), not facets** + a size-facet negative control.
- **M6 cache BOTH ESM3 confidence signals** (seq-only pLDDT **and** structure-token predictive entropy) with provenance.

**S3 — correctness:**
- **M-S7 hierarchical/partial-pooling estimator** for the table (not thousands of independent probes).
- **M-S8 cross-model `align` must report coverage**; the residue intersection drops ligand/modified/cropped
  residues — exactly the interface-relevant ones — so report and handle this selection bias in score Δℓ.
- **M-S9 diffusion-determinism carve-out** (pin seed/noise; cache final recycle; trunk preferred).

**S4 — hygiene:**
- **M-H1 global no-score-as-split/covariate/nuisance invariant** (lint beyond categories).
- **M-H2 pin the ipSAE d0 variant**, one shared implementation.
- **M-H3 pre-register Δℓ family + thresholds before any caching.**

**Carried old contract (confirmed generalized):** residue join key (CIF token-map, M-S8 extends it); pinned forward
+ shared hook point; frozen hashed splits before any cache; fp16 storage / **fp32 metric math**; train-only fits;
confidently-wrong power at split-freeze; determinism as a checked gate (with the M-S9 carve-out).

---

## 9. Data & splits
Per `datasets.md`: structure features from non-redundant PDB chains (MMseqs2 ≤30% ∩ deposition-date), Tsuboyama
(CATH WT-domain-cluster split) for stability, PLINDER/PINDER/Bernett for complexes/interfaces/specificity. **MSAs
frozen** (M3). **Per-model post-cutoff held-out** (M4) because each model's training cutoff differs. Experimental
binding only. Splits frozen+hashed before any cache; the confidently-wrong slice counted per category at freeze.

## 10. Compute
Single-rep (L×384) caching for all layers is cheap (1×80 GB over a day or two). **Pair-tensor caching is the
bottleneck** (dense z at L=500 ≈64 MB/layer/complex fp16 → multi-TB naive) → **interface-subset only, few layers,
fp16/int8, token cap ≤512–768**, per-complex sharding; derived scores (ipSAE) recomputed, not stored. AF3/Protenix
are score-only (run inference for outputs; no activation caching). Hook activations in the SAME inference pass.

## 11. Build order (extends the 8-PR plan; CPU/weight-free through PR6.5)
PR1 runner/CI · PR2 splits+freeze · PR3 cache+reader+synthetic · PR3.5 **ModelAdapter ABC + synthetic adapters +
target registry** (refactor ESM3 runner into an adapter) · PR4 controls · PR5 probes/atlas · PR5.5 **multi-model
cache + interface-pair store + reader.align + normalized-depth reducer** · PR6 dicts+truthconf · PR6.5 **sweep
engine + Δℓ/table + extended smoke (closes the extended weight-free pipeline)** · PR7.1 Boltz-2 adapter (scores +
activations) · PR7.2 ESM3 real cache (first GPU) · PR7.3 AF3/Protenix **score-only ingestors** · PR8+ interventions
(persistence G2, then wrap-around; gated).

## 12. Minimal provably-correct first milestone (CPU, no weights)
Extended MPU-0 smoke on **≥2 synthetic "models" with different `n_layers`**, each with a planted truth signal AND a
planted "score" channel in a DIFFERENT layer, asserting: (a) the circularity guard fires (self-decode labelled
trivial); (b) `align` reports residue coverage; (c) **Δℓ is recovered in normalized depth with the planted sign**
(forces M1); (d) size/MSA enter as a regressed covariate with a size-facet negative control (forces M5);
(e) underpowered strata are flagged; (f) the Δℓ forest-plot reduction renders. This proves the *mechanic*; it proves
no boundary fact.

## 13. Boundary facts to confirm before any headline (all `[U]`)
ESM3 `d_model`≈1536 / `n_layers`≈48 / RoPE / seq-only pLDDT existence; Boltz-2 layer count (~48–64) + affinity-head
source + hook point; Protenix exact version/license/dims/hook feasibility; AF3 runnability (→ alias) + `--save_embeddings`
filenames/shapes; ipSAE d0 variant; per-model ligand tokenization; MSA source + N_eff; per-model training cutoffs;
dataset counts/licenses. All isolated to adapters; none block building/testing the pipeline now.

## 14. What needs the user
Secure **ESM3-open** access and submit the **AF3 weights request form early** (2–3 days); confirm the §13 boundary
facts as each adapter is written; and a human/compliance pass on the dual-use statement (binding/specificity).

## 15. Go/no-go
Build PR1–PR6.5 now (CPU, weight-free) with S1–S2 fixes baked in; freeze the pre-registration only after the
extended smoke passes with normalized-depth Δℓ. Headline cells require the full S1–S4 fixes + the boundary facts.
If Δℓ is not cross-model consistent once real activations are in, the dissection reverts to an appendix atlas and
the paper stands on C1 (identity) + C3 (truth-vs-confidence).
