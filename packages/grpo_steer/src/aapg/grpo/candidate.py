"""Build a scoring :class:`Candidate` by threading a designed sequence onto a backbone.

The GRPO rollout samples a backbone, designs sequences onto its binder chain, and poses each
against the fixed target. :func:`build_candidate` packages one such proposal into the shared
:class:`aapg.core.scoring.Candidate` contract consumed unchanged by every scorer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aapg.core.scoring import Candidate

if TYPE_CHECKING:
    from rfdiffusion3.adapter import Backbone, Target

__all__ = ["build_candidate"]


def build_candidate(
    backbone: Backbone,
    sequence: str,
    target: Target,
    name: str,
    binder_chains: tuple[str, ...] | None = None,
) -> Candidate:
    """Thread ``sequence`` onto ``backbone`` and pose it against ``target`` as a candidate.

    The candidate's ``structure_path`` points at the backbone structure, its binder chains come
    from ``binder_chains`` and target chains from ``target.chains``, and ``metadata`` carries a
    reference back to the backbone (path, length, and its stored ``ca_coords`` when present) and
    the target (structure path, hotspots) so downstream stages (scRMSD, structure prediction)
    can recover the geometry.

    Args:
        backbone: the sampled binder backbone.
        sequence: the designed binder sequence to thread onto it.
        target: the fixed target the binder is scored against.
        name: human-readable identifier for logging and provenance.
        binder_chains: chain IDs assigned to the designed binder. When omitted, the chains the
            sampler recorded in ``backbone.metadata["binder_chains"]`` are used as the single
            source of truth (falling back to ``("A",)`` only for legacy backbones that predate
            that metadata).

    Returns:
        A :class:`Candidate` ready to be passed to any registered scorer.

    Raises:
        ValueError: if the resolved binder chains overlap the target chains, are empty, or (v1
            limitation) span more than one chain.
    """
    if binder_chains is None:
        recorded = backbone.metadata.get("binder_chains")
        binder_chains = tuple(recorded) if recorded else ("A",)
    binder_chains = tuple(binder_chains)

    if not binder_chains:
        raise ValueError(f"candidate {name!r} has no binder chains")
    overlap = set(binder_chains) & set(target.chains)
    if overlap:
        raise ValueError(
            f"candidate {name!r} binder chains {binder_chains} overlap target chains "
            f"{tuple(target.chains)!r}: {sorted(overlap)}"
        )
    if len(binder_chains) > 1:
        raise NotImplementedError(
            f"candidate {name!r} has a multi-chain binder {binder_chains}; the v1 GRPO pipeline "
            "threads a single designed sequence onto one binder chain. Multi-chain binders need "
            "per-chain sequence assignment (not yet implemented)."
        )

    metadata: dict[str, Any] = {
        "backbone_path": backbone.path,
        "backbone_length": backbone.length,
        "target_structure_path": target.structure_path,
        "target_chains": tuple(target.chains),
        "hotspots": tuple(target.hotspots),
    }
    ca_coords = backbone.metadata.get("ca_coords")
    if ca_coords is not None:
        metadata["backbone_ca_coords"] = ca_coords

    return Candidate(
        structure_path=backbone.path,
        binder_chains=tuple(binder_chains),
        target_chains=tuple(target.chains),
        binder_sequence=sequence,
        name=name,
        metadata=metadata,
    )
