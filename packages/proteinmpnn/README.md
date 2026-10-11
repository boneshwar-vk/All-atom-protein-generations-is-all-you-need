# proteinmpnn (vendored fork)

Vendored fork of **ProteinMPNN** (dauparas/ProteinMPNN, MIT). The upstream source and its small
MIT-licensed weights are vendored under `src/proteinmpnn/vendor/`:

- `protein_mpnn_run.py`, `protein_mpnn_utils.py`, `helper_scripts/` — upstream, unmodified.
- `vanilla_model_weights/` and `soluble_model_weights/` — `v_48_{002,010,020,030}.pt` (6.7 MB each).
- (excluded: `examples/`, `inputs/`, `outputs/`, `training/`, `colab_notebooks/`, `ca_model_weights/`.)

`adapter.py` drives `protein_mpnn_run.py` as a **subprocess** (via the current interpreter), so
torch is only needed in that subprocess — importing `proteinmpnn` stays light. Binder chains are
redesigned; target chains are kept fixed as context.

- Provenance (upstream commit SHA + date) is in the repo-root [`VENDORING.md`](../../VENDORING.md).
- The upstream MIT `LICENSE` is preserved here (`packages/proteinmpnn/LICENSE`).
- The weights are committed (small, MIT) via a `.gitignore` exception; the wheel ships them too.
- Runtime: `pip install torch numpy` in the interpreter that runs designs (the `run` extra is a
  documentation stub kept out of the workspace lock). `get_designer(mock=True)` needs neither.

Module 4 (iterative ProteinMPNN) will edit the vendored source in place.

**Status:** vendored (`__vendored__ = True`); the design adapter is wired (real subprocess path +
deterministic CPU mock).
