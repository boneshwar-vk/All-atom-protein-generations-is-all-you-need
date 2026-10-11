# aapg-validation (module 3)

A generic staged pipeline that takes backbones and validates the resulting complexes:

```
backbones -> ProteinMPNN -> Protenix v2 -> OpenDDE -> verdict
```

- `stages.py` — the typed `Stage` abstraction (declared inputs/outputs) so stages compose and
  reorder from config.
- `pipeline.py` — the runner that chains stages into a verdict/report.
- Stages wrap tools from `aapg.toolkit` (each tool in its own container).

Config group: [`configs/pipeline/`](../../configs/pipeline).

**Status:** placeholder stubs only.
