# data/weights/

Model weights live here, one subdirectory per model. **Nothing in this tree is committed to
git** (it is git-ignored) — weights are large and some are license-restricted, so they are
*fetched/placed on setup*, not version-controlled.

Populate it with:

```bash
make weights            # == python scripts/download_weights.py  (places open weights, prints the rest)
```

| Subdir | Model | License | How it gets here |
|--------|-------|---------|------------------|
| `boltz2/`       | Boltz-2 | MIT (open) | Boltz auto-downloads to its cache on first `boltz predict` |
| `proteinmpnn/`  | ProteinMPNN | MIT (open) | ships in the vendored `packages/proteinmpnn` fork |
| `rfd3/`         | RFdiffusion3 | BSD-3 (open) | `foundry install rfd3` |
| `protenix_v1/`  | Protenix v1 | Apache-2.0 (open) | `protenix` CLI auto-download |
| `protenix_v2/`  | **Protenix v2** | **PROPRIETARY** | **licensed users only** — download yourself and place here |
| `esm2/`         | ESM-2 | MIT (open) | huggingface auto-download (optional) |

## Protenix v2 (licensed)

Protenix v2 weights are proprietary and **must not be committed or redistributed**. If you are
licensed to use them, download them and drop them in `data/weights/protenix_v2/`. The default
`structure_auto` scorer detects them there and switches from Boltz-2 to Protenix v2 automatically;
otherwise the pipeline stays on open Boltz-2.
