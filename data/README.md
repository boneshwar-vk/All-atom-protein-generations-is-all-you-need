# data/

Datasets, model weights, and caches. **Gitignored** — nothing here is committed. Fetch weights
with `uv run python scripts/download_weights.py` (manifest at `scripts/weights_manifest.yaml`).

Expected layout:

```
data/
  weights/     model checkpoints (per scripts/weights_manifest.yaml)
  cache/       scorer / prediction caches
  <datasets>/  raw and processed datasets
```
