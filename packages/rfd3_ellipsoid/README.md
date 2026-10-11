# aapg-ellipsoid (module 2)

Ellipsoid-constrained RFD3: ProtComposer-style conditioning of RFDiffusion3 on 3D ellipsoid
layouts, for **on-target** and **off-target** specification.

- Deep architecture edits (new conditioning channels) live in the vendored `rfdiffusion3` fork.
- This package holds the ellipsoid data structures (`layout.py`), featurizers, conditioning
  config, and the sampling API (`sampling.py`).
- Ellipsoid math is shared from `aapg.core.geometry`.

Config group: [`configs/ellipsoid/`](../../configs/ellipsoid).

**Status:** placeholder stubs only.
