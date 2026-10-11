# aapg-toolkit

Adapters to **external** protein tools, each implementing an `aapg.core.scoring` interface and
registering itself by name. Tools run in their own containers (see `containers/`) and are invoked
via subprocess, so heavy/conflicting deps never pollute the main environment.

Structure backends (interface scoring):
- `structure_auto` (**default**) — Boltz-2 if open, auto-switches to Protenix v2 when licensed
  weights are present in `data/weights/protenix_v2/`.
- `boltz2` — Boltz-2 (MIT, open weights; the open default).
- `protenix_v2` — Protenix v2 (**proprietary weights**, licensed users only) or Protenix v1
  (Apache-2.0) via `model_name`.

Other adapters: `rosetta_interface` (PyRosetta; dG/SC/CMS/SAP/pI), `ipsae` (pure-numpy metric),
plus placeholders `opendde`, `sequence_models` (ESM-2, ...).

All heavy backends are imported lazily and have deterministic `*_mock` variants for CPU/CI. See
the repo-root `THIRD_PARTY_LICENSES.md` for each backend's license and weight terms.
