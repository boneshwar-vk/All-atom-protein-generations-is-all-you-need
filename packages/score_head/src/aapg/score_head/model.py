"""The score-head neural network (placeholder).

``StructureScoreHead`` maps :class:`~aapg.score_head.features.ComplexFeatures` to a dict of
predicted scores (e.g. ipSAE / iPTM / interface dG / ... as regression targets, and/or a single
learned quality score). The real architecture (e.g. an SE(3)/EGNN or pair-transformer over the
residue graph) and its torch dependency arrive with the implementation; this stub only fixes the
output contract so trainers, inference and the scorer adapter can be wired against it.

torch is imported lazily inside :meth:`StructureScoreHead.load` / ``forward`` so importing this
module never requires torch.
"""

from __future__ import annotations

from typing import Any

# The score names this head is intended to regress (kept aligned with the GRPO reward metrics).
DEFAULT_TARGETS: tuple[str, ...] = (
    "ipsae",
    "iptm",
    "ipae_min",
    "ipae_mean",
    "complex_plddt",
    "interface_dg",
    "shape_complementarity",
    "contact_molecular_surface",
)


class StructureScoreHead:
    """Learned structure -> scores model (placeholder; no torch required to import)."""

    def __init__(self, targets: tuple[str, ...] = DEFAULT_TARGETS, **cfg: Any) -> None:
        self.targets = tuple(targets)
        self.cfg = cfg

    @classmethod
    def load(cls, checkpoint: str, **cfg: Any) -> StructureScoreHead:
        """Load a trained head from a checkpoint (placeholder; lazy torch)."""
        raise NotImplementedError(
            "StructureScoreHead.load is a placeholder; checkpoint loading (and torch) arrive "
            "with the real head."
        )

    def predict(self, features: Any) -> dict[str, float]:
        """Run a forward pass and return ``{target_name: score}`` (placeholder)."""
        raise NotImplementedError(
            "StructureScoreHead.predict is a placeholder; the forward pass arrives with the "
            "real head."
        )
