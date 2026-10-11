"""Training the score head (placeholder).

The head is trained by **distillation** from the expensive pipeline: run RFD3 -> (MPNN) ->
Protenix/Rosetta on a dataset of complexes, record their scores as labels, and fit the head to
predict those labels from structure alone. Once trained it replaces that chain at inference time.

This stub fixes the entry point; the optimiser/loop and torch dependency arrive later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters for score-head training (placeholder)."""

    data_dir: str | None = None  # complexes + their pipeline-computed score labels
    targets: tuple[str, ...] = ()
    max_steps: int = 10000
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    checkpoint_dir: str | None = None


def train(config: TrainConfig) -> str:
    """Train the score head and return the path to the best checkpoint (placeholder)."""
    raise NotImplementedError(
        "aapg.score_head.train.train is a placeholder; the distillation training loop arrives "
        "with the real head."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for training (placeholder)."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    print("aapg score_head train: placeholder — not implemented yet.")
    print(f"(received args: {argv})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
