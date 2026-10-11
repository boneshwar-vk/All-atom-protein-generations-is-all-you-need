# rfdiffusion3 (fork + CLI adapter)

Interface to **RFdiffusion3** (RosettaCommons/foundry) for binder-backbone generation, keeping
the upstream import name `rfdiffusion3`. Module 2 (ellipsoid conditioning) will vendor and edit
the upstream model source here.

`adapter.py` is **concretely wired**: `RFD3Sampler` drives the RFD3 CLI as a **subprocess** (no
torch imported in-process), collects the generated complex PDBs, and extracts the binder CA trace
(every chain not in `target.chains`) into `Backbone.metadata["ca_coords"]` for scRMSD. A
dependency-free `MockRFD3Sampler` emits deterministic backbones for CPU tests/dry runs.

**Version-dependent flags.** RFD3 (foundry) uses Hydra-style `key=value` overrides whose exact
names differ across releases. The adapter builds the run command in one overridable place:
configure `command`, `config_name`/`config_path`, `checkpoint`, `spec` (a dict of Hydra overrides
for the design task — hotspots, contig/binder length, input PDB), `extra_args`, and `output_glob`
(see `configs/sampler/default.yaml`). The adapter owns running it + parsing outputs; match `spec`
to your installed RFD3. A `FUTURE-RL HOOK` docstring marks where per-step log-probs are recorded
for the policy-gradient `rl` mode.

- Install/vendor via `scripts/vendor_rfd3.sh` / `foundry install rfd3`; provenance in
  [`VENDORING.md`](../../VENDORING.md); preserve the upstream `LICENSE` here when vendoring source.

**Status:** CLI adapter wired (`__vendored__ = False` — model source not copied in-tree yet; the
subprocess path runs against an installed RFD3, the mock runs anywhere).
