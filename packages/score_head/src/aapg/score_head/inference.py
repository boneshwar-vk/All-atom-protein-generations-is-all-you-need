"""Inference: predict scores for a complex structure (placeholder).

Thin wrapper tying featurisation + a loaded :class:`StructureScoreHead` together, so callers
(and the scorer adapter) have one function: structure path in, score dict out.
"""

from __future__ import annotations

from typing import Any

from aapg.score_head.features import featurize_complex
from aapg.score_head.model import StructureScoreHead


def predict_scores(
    structure_path: str,
    model: StructureScoreHead,
    binder_chains: tuple[str, ...] = ("A",),
    target_chains: tuple[str, ...] = ("B",),
    **cfg: Any,
) -> dict[str, float]:
    """Featurise ``structure_path`` and return the head's predicted scores (placeholder)."""
    features = featurize_complex(
        structure_path, binder_chains=binder_chains, target_chains=target_chains, **cfg
    )
    return model.predict(features)
