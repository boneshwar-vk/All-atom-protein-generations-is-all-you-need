# Vendoring policy

Two upstream projects are **modified**, so they are vendored as first-party workspace packages
rather than consumed as external dependencies:

| Package | Upstream | Import name | Modified by |
|---------|----------|-------------|-------------|
| `packages/rfdiffusion3` | RFdiffusion3 (RFD3) | `rfdiffusion3` | module 2 (ellipsoid conditioning) |
| `packages/proteinmpnn`  | ProteinMPNN         | `proteinmpnn`  | module 4 (iterative loop) |

External tools that we only **call** (Protenix v2, OpenDDE, Rosetta, ESM, TM-align, Foldseek) are
**not** vendored — they are installed in their own containers and wrapped by adapters in
`packages/toolkit`.

## Rules

1. **Keep upstream import names.** Do not rename `rfdiffusion3` / `proteinmpnn`; this keeps the diff
   against upstream small and re-vendoring mechanical.
2. **Preserve the upstream license.** Each fork keeps its own `LICENSE` (and `NOTICE`/attribution if
   present). The repository-root MIT license does **not** override upstream terms. Confirm the
   upstream license permits modification and redistribution before vendoring.
3. **Record provenance** in the table below on every (re-)vendor: upstream URL, the exact commit
   SHA copied, the date, and a one-line summary of local modifications.
4. **Isolate local edits.** Where practical, keep our changes in clearly marked, separately
   importable modules/functions so they are easy to diff and re-apply after an upstream bump.

## Procedure to (re-)vendor

```bash
# Example for RFD3 — adjust URL/paths as needed.
git clone --depth 1 <upstream-url> /tmp/rfd3-src
# copy the library source into the package's src/ (NOT the whole repo)
rsync -a --delete /tmp/rfd3-src/<lib-dir>/ packages/rfdiffusion3/src/rfdiffusion3/
cp /tmp/rfd3-src/LICENSE packages/rfdiffusion3/LICENSE
# then re-apply local modifications and update the provenance record below
```

## Provenance record

| Package | Upstream URL | Commit SHA | Date vendored | Local modifications |
|---------|--------------|-----------|---------------|---------------------|
| rfdiffusion3 | _TBD_ | _TBD_ | _not yet vendored_ | placeholder package only |
| proteinmpnn  | https://github.com/dauparas/ProteinMPNN | 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57 | 2026-10-11 | Source + vanilla/soluble weights copied unmodified into `src/proteinmpnn/vendor/` (excluded: examples/, inputs/, outputs/, training/, colab_notebooks/, ca_model_weights/). LICENSE (MIT, © 2022 Justas Dauparas) preserved at `packages/proteinmpnn/LICENSE`. Driven as a subprocess by `adapter.py`; upstream code unchanged. |
