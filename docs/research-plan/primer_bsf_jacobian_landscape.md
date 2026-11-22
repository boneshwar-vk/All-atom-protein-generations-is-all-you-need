# A primer: BSF, the Jacobian lens, J++, and the current landscape of interpretability for biological ML models

Written for someone new to mechanistic interpretability. Plain, literal language; no analogies.
Dates and claims reflect a literature review done 2026-11-19. Items that could not be confirmed
against a primary source are marked (unverified).

---

## 0. Setup and notation you need once

A deep network processes an input (a protein sequence, a DNA window, a cell's gene-expression
vector) and, at every layer, holds an internal vector of numbers. That vector is called an
**activation** (also "representation", "hidden state", "embedding").

- Write one activation vector as `x`, a list of `d` numbers, i.e. `x ∈ R^d`. `d` is the model's
  hidden width (for example 1536 for the open ESM3 model, a few thousand for large models).
- A transformer keeps a running vector at each position that is updated layer by layer. The vector
  at layer `l` is written `h_l`. The vector at the last layer is `h_final`.
- The model ends with a linear "unembedding" matrix `W_U` that turns `h_final` into scores over the
  output vocabulary (amino acids, nucleotides), followed by a softmax to make probabilities.

Interpretability asks: what information is inside these activations, and which parts of it does the
model actually use to produce its output. The three objects below are three specific tools for that.

---

## 1. BSF — the Block-Sparse Featurizer

**What problem it addresses.** The standard tool for reading activations is a **sparse autoencoder
(SAE)**. You first need to understand the SAE to understand what BSF changes.

### 1.1 The SAE baseline

An SAE is a small network trained to copy an activation `x` through a narrow, sparse bottleneck.

- **Encode:** produce a long list of feature strengths `a = f(W_enc · x + b_enc)`, where `f` keeps
  only a few entries nonzero (for example "TopK": keep the `k` largest, set the rest to 0).
- **Decode:** rebuild the activation as a sum of fixed directions:
  `x̂ = Σ_i a_i · d_i`, where each `d_i ∈ R^d` is one learned **direction** (one column of the
  decoder), and `a_i` is its scalar strength.
- **Train** by minimising the reconstruction error `‖x − x̂‖²` plus a penalty that forces most
  `a_i` to be 0.

The result: each learned feature `i` is a single direction `d_i`. When feature `i` is active, it
contributes `a_i · d_i` — a fixed direction scaled by one number. So an SAE feature can represent
**how strongly** something is present, but a single feature is one direction only.

**The limitation BSF targets.** Many real quantities are not "present / absent" or "weak / strong";
they are **continuous and cyclic**. A protein backbone angle runs from 0° to 360° and wraps around.
To represent a smoothly varying angle, you need at least a 2-dimensional code (for example a pair
(cos θ, sin θ)), not one scalar. A standard SAE tends to cover such a quantity by spending many
separate single-direction features, one for each small arc of the range. This is called
**feature splitting** or **fragmentation**.

### 1.2 What BSF changes

BSF (Block-Sparse Featurizer, Goodfire, arXiv 2606.25234) keeps the "sparse dictionary" idea but
makes each feature a small **block** of directions instead of a single direction.

- The dictionary is split into `G` blocks. Each block `g` is a small matrix `D_g ∈ R^{d × b}`,
  holding `b` directions. `b` is the **block size** (`group_size`, default 3).
- The code for block `g` is now a short **vector** `z_g ∈ R^b`, not a scalar.
- **Decode:** `x̂ = Σ_g D_g · z_g`.
- **Sparsity is across blocks, density is within a block.** You keep only a few blocks active (for
  example the `l0` blocks with the largest norm `‖z_g‖`), but inside an active block all `b`
  coordinates of `z_g` may be nonzero.

Consequence: an active BSF feature is a **point in a `b`-dimensional subspace** (the span of the `b`
directions in `D_g`), not just a scaled single direction. That subspace can hold a small smooth or
cyclic shape — exactly what a single-direction SAE feature cannot. A block of size 2 can carry a
value on a circle; a block of size 3 can carry a value on a sphere or a small 2-surface.

### 1.3 The simple math

Training objective, in words and symbols:

```
minimise over the dictionary:   Σ over activations  ‖ x − Σ_g D_g z_g ‖²        (reconstruct well)
subject to:                      only l0 of the blocks are active per input       (be sparse in blocks)
```

The three BSF variants differ only in how the block codes `z_g` are produced and how sparsity is
enforced:

- **Vanilla:** a free linear encoder produces the `z_g`; keep the `l0` blocks with the largest norm
  (block-TopK); decoder directions are re-normalised each step. If you set `b = 1` you recover an
  ordinary TopK SAE — this is the natural control to compare against.
- **Grassmannian:** each block's `b` directions are forced to be orthonormal (a clean little frame),
  and the code is tied to the decoder: `z_g = γ_g · D_gᵀ x` with one learned scalar `γ_g` per block.
- **Group-Lasso:** a gate turns a block on only when its norm passes a learned threshold `θ_g`
  (`gate = 1 if ‖a_g‖ > θ_g else 0`), and the overall sparsity is pushed to a target level.

**One important caveat built into the math.** Inside a block, only the **subspace** is pinned down,
not the particular axes. Any rotation of the `b` directions describes the same subspace. So you may
not read meaning into a single raw axis of a block; you may only read **rotation-invariant**
properties of the block (its subspace, and quantities aligned to it after a fitted alignment). This
is why evaluation of BSF features has to use invariant measures.

**Cost and requirements.** BSF trains on **cached activations only** — you run the big model once to
save activations, then train BSF on them. It never needs gradients through the big model and makes
no assumption about the model's architecture beyond "there is an activation vector per position".
Training cost is the same as an SAE of width `G × b`.

**Known weakness.** A follow-up critique (Jerpelea & Ananthram, arXiv 2608.27515) shows BSF can
split one true manifold across several blocks, and the result depends on the random seed. Their
proposed fix is a different block-selection rule ("Tournament Top-K"). BSF has so far been shown on
vision models; as of this review it had not been published on a biological model.

---

## 2. The Jacobian lens

**What problem it addresses.** The two methods above decompose a single activation. The Jacobian
lens instead asks a directional question: for a state sitting at some middle layer, **what is it
pushing the final output toward**, given the rest of the network.

### 2.1 The object

A Jacobian is the matrix of partial derivatives of one vector with respect to another. For the
mapping from a middle-layer state `h_l` to the final state `h_final`, the Jacobian is

```
J = ∂ h_final / ∂ h_l        (a d × d matrix)
```

Entry `(i, j)` says: if you nudge the `j`-th number of `h_l` a little, how much does the `i`-th
number of `h_final` move. It is the best **linear** description of how the second half of the
network transforms this state.

### 2.2 The lens

A single input gives a noisy, input-specific Jacobian. The **Jacobian lens** (Gurnee et al., arXiv
2607.15495) averages it over a corpus of inputs and positions to get one stable matrix per layer:

```
J_l = average over many prompts and positions of ( ∂ h_final / ∂ h_l )
```

Then it **reads out** through the model's own output head. For a state `h`:

```
lens_l(h) = softmax( W_U · norm( J_l · h ) )
```

- `J_l · h` carries the state forward to output space using the averaged linear map.
- `norm` is the model's final normalisation; `W_U` is the unembedding.
- The output is a probability distribution over the vocabulary — the tokens this layer's state is
  disposed to produce.

If you replace `J_l` by the identity matrix, you get the older, cruder **logit lens**. The point of
the Jacobian lens is that `J_l` accounts for the transformations still to come, so it reads the
middle layers more faithfully.

**Cost and requirements.** You need the model weights and the ability to take gradients
(backpropagation). About 100 prompts are enough to estimate `J_l`. Applying the lens afterwards is a
single matrix multiply per layer.

**Built-in assumptions.** The method assumes (a) a single running activation vector per position
(a residual stream), and (b) a meaningful token output head to read into. Both hold for ordinary
language models. Both are awkward for biological models — a point returned to in Section 5.

**Known weakness.** Ostrow et al. ("A Mathematical Theory of the Jacobian Lens", arXiv 2610.31955)
show the **averaged** Jacobian need not resemble any single input's Jacobian, and the directions it
emphasises can explain very little of the actual variation in activations. The averaging can
mislead. (This critique is about the base Jacobian lens, not specifically about J++ below.)

---

## 3. J++ — a more faithful Jacobian lens

J++ (Ayonrinde & Lindsey, Anthropic, Oct 2026; code at github.com/safety-research/jpp_lens;
identity unverified through indexed databases because it is a blog post plus a repository) keeps the
same end product — one linear map per layer — but makes three changes to reduce the errors above.

1. **Jacobian filtering (handle the averaging problem).** Instead of one global average, group the
   activations into a handful of clusters (k-means), compute one average Jacobian per cluster (an
   "expert"), then combine the experts with learned weights. This avoids blending very different
   situations into a single blurred average. The weight-fitting step is reported to use a small set
   of labelled examples, i.e. it is partly supervised (unverified — the post could not be opened).

2. **A modified backward pass (handle the linearisation problem).** When computing the derivatives,
   certain nonlinear terms (some normalisation scales, some gate terms) are held fixed rather than
   differentiated through. This "layer-wise relevance propagation" style of backward pass gives a
   linear map that tracks the true computation more closely.

3. **Readout filtering.** Drop output tokens that carry no real content before ranking.

On language models, the authors report J++ surfaces intermediate, unspoken concepts earlier and
more accurately than the plain Jacobian lens. It inherits the general caveat from Ostrow et al.: it
is still an averaged, first-order (linear) summary of a nonlinear network.

---

## 4. The current landscape of interpretability for biological models

This section explains what people are actually doing, as of late 2026, on protein language models
(for example ESM-2, ESM C, ESM3), genomic sequence models (Nucleotide Transformer, Evo 2,
AlphaGenome, Enformer, Borzoi), protein-structure and co-folding models (ESMFold, Boltz, RoseTTAFold)
and single-cell models (Geneformer, scGPT, scFoundation). Most of this literature is 2025–2026 and
much of it is still bioRxiv/arXiv preprints.

### 4.1 Two methodological directions — and the difference you asked about

There are two distinct research moves, and most papers are one or the other (some combine them):

**Direction A — start from a known, labelled quantity and check whether the model represents it.**
You decide in advance on a property with a clear definition — a backbone angle, a secondary-structure
label, solvent exposure, a binding-site position, a cell type, a transcription-factor binding motif.
You compute or look up the true value of that property for many inputs, producing a small, controlled,
explainable dataset. Then you check whether that property can be recovered from the model's
activations, usually by fitting a simple (often linear) classifier or regressor from activation →
property. This is called **probing** or **linear decoding**.
- If a simple model recovers the property, the information is present and accessible in that layer.
- You can repeat this layer by layer to see **where** and **when** in the network the property
  appears. (Example: Whitfield et al., bioRxiv 2026-07, trace how physicochemical → secondary
  structure → domain concepts appear with depth across eight protein models.)
- The planned project discussed elsewhere in this workspace is of this type: compute true backbone
  angles from experimental structures, then test how the model encodes them.
- **Key limitation:** a property being decodable does **not** prove the model uses it. A model can
  carry information it never acts on. This was shown directly for a structure model (Jedryszek et
  al., arXiv 2608.11475): a β-strand-direction property was decodable with high accuracy, yet
  pushing on that decoded direction did not change the model's output.

**Direction B — start from the activations and discover what structure is inside them.**
You do not pick a property in advance. You decompose the activations into many parts with an
unsupervised method (an SAE, a transcoder, or BSF), and only **afterwards** ask what each part means
by seeing which inputs switch it on and matching those to known annotations. This is **dictionary
learning** followed by **auto-interpretation**.
- This is discovery-oriented: it can surface features nobody asked about.
- **Key limitations:** (1) the features are matched against annotation databases (Swiss-Prot, GO,
  InterPro) that the model was often trained on, so a match can be circular rather than a discovery;
  (2) different random seeds learn different features; (3) some control models with random,
  untrained weights can produce features that score as "interpretable" too, so a trained-vs-random
  comparison is needed to claim the structure was actually learned.

In short: **Direction A asks "is this specific known thing in there, and does the model use it?"**
**Direction B asks "what is in there at all?"** Mature papers increasingly do both: use Direction B
to find candidate features, then Direction A plus an intervention to test whether a candidate is
real and used.

### 4.2 A third activity that cuts across both: causal testing

Neither decoding nor dictionary learning, on its own, shows that the model **uses** a piece of
information. For that, people intervene and measure:
- **Ablation / activation patching:** remove or replace a feature or a component and see whether the
  output changes.
- **Steering:** add a feature's direction to the activation and see whether the output moves in the
  predicted way.
- **Circuit tracing:** with transcoders, follow how features in one layer drive features in the next.
This is the strongest form of evidence, and the current frontier. The repeated finding across
proteins, genomics and single-cell is that **information is often present (decodable) but the model
uses much less of it causally than the decoding suggests.**

### 4.3 What methods the recent papers actually use

**Papers built on SAEs / dictionary learning (Direction B):**
- Proteins: InterPLM (Simon & Zou, Nature Methods 2025) — the foundational protein-SAE paper;
  Adams et al. (ESM-2 TopK SAEs); Gujral et al. (PNAS 2025, SAEs plus transcoders); Reticular
  (Matryoshka SAEs on the ESMFold base model); Hu et al. (SAE on ESM C 6B for enzyme function).
- Structure/co-folding: PairSAE (SAE on Boltz-2's pairwise representation; note it predicts the
  model's own predicted affinity, not experimental affinity, and has no causal controls); FoldSAE
  (RFdiffusion); VFUSE (first SAE on an all-atom diffusion model).
- Genomics: the Evo 2 paper ships a BatchTopK SAE on layer 26 of the 7B model (Arc + Goodfire);
  Decode-gLM (Nucleotide Transformer, SAEs plus "meta-SAEs", which found a feature exposing
  contaminated training data); SPIRAL (an RNA model).
- Single-cell: Pedrocchi et al. (BatchTopK SAEs on scGPT/scFoundation/Geneformer).

**Papers that do NOT use SAEs (they use other tools):**
- **Derivative / input-output analyses on the model's predictions** (close relatives of a Jacobian,
  but input→output rather than middle-layer→output): the "categorical Jacobian" (Zhang et al., PNAS
  2024) extracts residue-residue contacts from ESM-2 by differentiating predicted probabilities;
  nucleotide-dependency maps (Tomaz da Silva et al., Nature Genetics 2025) do the analogous thing
  for DNA models and discovered real RNA structures. Note this "Jacobian on proteins/DNA" already
  exists and owns that name — different from the middle-layer Jacobian lens of Section 2.
- **Attention read-outs:** Thorstenson (arXiv 2606.21876) extracts contacts directly from ESM-2
  attention in a single forward pass.
- **Probing / representational-geometry (Direction A):** Whitfield et al. (concept emergence by
  depth); Steenwyk (similarity analysis of ESM3 across depth); Bigot et al. (a "nativeness" axis).
- **Transcoder circuits (not SAEs):** ProtoMech and ProGenMech (Tsui et al., Aghazadeh lab) trace
  cross-layer circuits in ESM-2 and a generative protein model.
- **Embedding-transfer into a causal model:** DeepSCENIC takes the internal embeddings of
  Enformer/Borzoi and builds a gene-regulatory network validated against CRISPR perturbation — the
  strongest perturbation-validated use of a sequence model's internals so far.
- **Weight-level analysis:** Vinod et al. (PLOS Comp Biol 2025) find structure knowledge factorised
  into trainable subnetworks.

**Papers whose main point is a caution or a negative result:**
- Ali (arXiv 2607.19618): naive validation of DNA-model "transcription-factor features" is confounded
  by GC content and repeats; with composition-matched controls most apparent features vanish and only
  some survive a causal test.
- A large single-cell line of work (dominated by one author, I. Kendiukhov, ~17 papers) argues these
  models capture co-expression, not causal gene regulation; independently corroborated by benchmark
  papers (e.g. Ahlmann-Eltze et al., Nature Methods), though single-group replication is a caveat.
- General machine-learning cautions: automated interpretability scores do not separate trained from
  randomly-initialised networks (Heap & Aitchison); SAE features can be seed-unstable; feature "death"
  is severe on some structure models (Simon et al., >70% on AlphaFold3).
- A field review (Orlov et al., bioRxiv 2026-03) and a genomics position paper (arXiv 2606.07607)
  both argue the field must stop validating interpretability claims anecdotally.

### 4.4 What they are ultimately trying to see

Across all of the above, the questions reduce to four:
1. **Does the model represent known biology** (structure, function, regulatory elements, cell state)?
   — Largely yes, and findable with SAEs and probes.
2. **Does it represent it causally, or only correlationally?** — Much weaker; this is the active
   frontier and the source of most negative results.
3. **Is it real biology or a dataset shortcut / memorisation?** — Several papers show shortcuts
   (GC-content confounds, training-data contamination, flanking-sequence artefacts).
4. **Can we steer the model through its features**, and does the steering do what the feature's
   meaning predicts? — Sometimes; and "decodable but not steerable" is a repeated, important finding.

---

## 5. Why these three tools are a non-trivial fit for biological models (brief)

- **BSF** transfers most easily: it needs only cached activations, so it runs on any of these models.
  The open question it suits is whether continuous/cyclic biological quantities (backbone angles,
  helix phase, codon phase) are stored as small manifolds that single-direction SAEs fragment.
- **The Jacobian lens / J++** transfer poorly as written, because they assume a single causal token
  stream read out through a token vocabulary. Genomic models have only four nucleotide tokens;
  protein masked models read in both directions and have several output heads; some genomic models
  have no token output head at all. What transfers is the idea of an averaged middle-layer→output
  linear map; the read-out has to be redesigned, and it must be distinguished from the already-named
  "categorical Jacobian" above.

---

### Pointers
- BSF: Fel et al., arXiv 2606.25234; critique Jerpelea & Ananthram, arXiv 2608.27515.
- Jacobian lens: Gurnee et al., arXiv 2607.15495; critique Ostrow et al., arXiv 2610.31955.
- J++: Ayonrinde & Lindsey, github.com/safety-research/jpp_lens (blog + code; unverified via databases).
- Field review: Orlov et al., bioRxiv 2026-03 ("What Do Biological Foundation Models Compute?").
- Full annotated reading list and the project plan: the other files in this scratchpad
  (FINAL_FINDINGS.md, review_proteins.md, review_genomics.md), and the Undermind workspace.
