# aapg-pipelines

End-to-end orchestration that composes the four modules into runnable pipelines, each a Hydra
entrypoint reading the shared [`configs/`](../../configs) tree:

| Command | Entry module | Composes |
|---------|-------------|----------|
| `aapg-generate` | `aapg.pipelines.generate` | ellipsoid-constrained (2) + GRPO-steered (1) generation |
| `aapg-design`   | `aapg.pipelines.design`   | iterative ProteinMPNN (4) |
| `aapg-validate` | `aapg.pipelines.validate` | validation pipeline (3) |
| `aapg-end2end`  | `aapg.pipelines.end2end`  | generate → design → validate |

**Status:** placeholder entrypoints (print a notice; Hydra wiring lands with the implementation).
