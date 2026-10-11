# Architecture

## Package dependency graph

```
                         +-------------------+
                         |     aapg.core     |  pure: io / repr / geometry /
                         |  (registry, ABCs) |  metrics / scoring registry
                         +---------+---------+
                                   ^
        +--------------------------+--------------------------+
        |                          |                          |
+-------+--------+        +--------+---------+       +--------+---------+
|  aapg.toolkit  |        |  rfdiffusion3    |       |   proteinmpnn    |
| external tools |        | (vendored fork)  |       | (vendored fork)  |
+---+--------+---+        +--------+---------+       +--------+---------+
    ^        ^                     ^                          ^
    |        |                     |                          |
    |        |            +--------+--------+        +--------+--------+
    |        |            | aapg.ellipsoid  |        |  aapg.mpnn_iter |
    |        |            |   (module 2)    |        |   (module 4)    |
    |        |            +--------+--------+        +--------+--------+
    |        |                     ^                          ^
    |        |            +--------+--------+                 |
    |        +------------+   aapg.grpo     |                 |
    |                     |   (module 1)    |                 |
    |                     +-----------------+                 |
    |                                                         |
    +----------------------+  aapg.validation (module 3)  +---+
                           +------------------------------+
                                        ^
                              +---------+---------+
                              |   aapg.pipelines  |  end-to-end orchestration
                              +-------------------+
```

- **`aapg.core`** depends on nothing in-repo (pure).
- **`aapg.toolkit`** depends on `core` (implements its scoring interfaces).
- The **vendored forks** (`rfdiffusion3`, `proteinmpnn`) are standalone; the forks are where deep
  upstream edits land.
- **Modules** depend on `core` + the fork(s)/toolkit they need:
  - module 1 `aapg.grpo` → core, rfdiffusion3 (reward via core/toolkit)
  - module 2 `aapg.ellipsoid` → core, rfdiffusion3
  - module 3 `aapg.validation` → core, toolkit, proteinmpnn
  - module 4 `aapg.mpnn_iter` → core, toolkit, proteinmpnn
- **`aapg.pipelines`** depends on all modules and composes them.

Keep this graph acyclic; `core` must never import a module or a fork.

## Why these boundaries

See the repo-root `README.md` ("Design principles") and `CONTRIBUTING.md` ("Layout rules").
The short version: `core` stays pure for fast CPU imports/CI; heavy, container-bound tools are
isolated in `toolkit`; upstream edits are quarantined in the forks; everything else is composed
from config.
