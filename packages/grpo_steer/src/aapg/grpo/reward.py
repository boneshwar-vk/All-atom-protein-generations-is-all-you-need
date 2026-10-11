"""Config-driven gated reward for binder selection, plus per-sequence aggregation.

:class:`IpSAEGatedReward` turns the bundle of scorer outputs for one candidate into a single
scalar used to rank candidates. It is a *gated* reward: a small set of HARD gates must all pass
(e.g. no clash, ipSAE above a floor, confident interface) or the candidate is rejected with a
large-negative reward; otherwise the reward is the primary metric (ipSAE) minus SOFT hinge
penalties for secondary developability / quality metrics that fall outside their desired
bounds. Everything is driven by config so the gates can be recalibrated per target, and missing
metric keys are skipped so the reward degrades gracefully when a scorer stage is absent.

:func:`aggregate_sequences` reduces the several per-sequence scores for one backbone (one per
MPNN design) into a single representative :class:`Score` before the reward is applied.
"""

from __future__ import annotations

import math
from typing import Any

from aapg.core.scoring import REWARDS, Reward, Score, combine_scores

__all__ = ["IpSAEGatedReward", "aggregate_sequences", "bound_satisfied", "bound_violation"]

# Reward returned when any hard gate fails. A tiny +primary term is added on top so that even
# among rejected candidates the ordering by primary metric is still defined (useful for GRPO
# advantages on hard targets where every candidate fails a gate).
_HARD_FAIL_FLOOR = -1e3

# Floor on a soft-penalty denominator. A bound whose characteristic magnitude is 0 (e.g.
# ``interface_dg: {max: 0}``) would otherwise divide by ~0 and produce an enormous penalty that
# inverts the ranking; the floor caps the per-unit penalty at "violation itself". Give such a
# metric an explicit ``scale`` in its bound (e.g. ``{max: 0, scale: 15}``) to set the sensitivity.
_SCALE_FLOOR = 1.0


def _finite(value: float) -> bool:
    """True iff ``value`` is a finite real number (NaN/inf are not)."""
    return math.isfinite(value)


def bound_violation(value: float, bound: dict[str, Any]) -> tuple[float, float]:
    """Return ``(violation, scale)`` for ``value`` against a single ``bound`` spec.

    ``bound`` is one of ``{"min": x}``, ``{"max": x}`` or ``{"range": [lo, hi]}`` and may carry an
    explicit ``{"scale": s}``. The violation is the non-negative amount by which ``value`` is
    outside the bound; ``scale`` normalizes penalties across metrics and is floored at
    :data:`_SCALE_FLOOR` so a zero-valued bound cannot blow up the penalty. A satisfied bound has
    violation ``0``.
    """
    violation = 0.0
    default_scale = 1.0
    if "range" in bound:
        lo, hi = float(bound["range"][0]), float(bound["range"][1])
        if value < lo:
            violation = lo - value
        elif value > hi:
            violation = value - hi
        default_scale = abs(hi - lo)
    elif "min" in bound:
        lo = float(bound["min"])
        violation = max(0.0, lo - value)
        default_scale = abs(lo)
    elif "max" in bound:
        hi = float(bound["max"])
        violation = max(0.0, value - hi)
        default_scale = abs(hi)
    scale = max(float(bound.get("scale", default_scale)), _SCALE_FLOOR)
    return violation, scale


def bound_satisfied(value: float, bound: dict[str, Any]) -> bool:
    """Whether ``value`` satisfies ``bound``. A non-finite value NEVER satisfies (fail-closed)."""
    if not _finite(value):
        return False
    violation, _scale = bound_violation(value, bound)
    return violation <= 0.0


@REWARDS.register("ipsae_gated")
class IpSAEGatedReward(Reward):
    """ipSAE-primary reward with hard gates and soft hinge penalties (config-driven).

    Args:
        primary: metric name whose value is the reward floor before penalties (default
            ``"ipsae"``).
        gates: deprecated alias merged into ``hard`` for backward compatibility.
        soft: mapping ``metric -> bound`` of soft constraints; each violated bound subtracts a
            scaled hinge penalty.
        hard: mapping ``metric -> bound`` of hard constraints; any violated bound rejects the
            candidate.
        penalty_weight: global multiplier on every soft penalty.

    A bound is ``{"min": x}``, ``{"max": x}`` or ``{"range": [lo, hi]}``. Metrics absent from
    the merged scores are skipped for both hard and soft checks (graceful degradation).
    """

    name = "ipsae_gated"

    def __init__(
        self,
        primary: str = "ipsae",
        gates: dict[str, Any] | None = None,
        soft: dict[str, Any] | None = None,
        hard: dict[str, Any] | None = None,
        penalty_weight: float = 1.0,
        **cfg: Any,
    ) -> None:
        self.primary = primary
        self.hard: dict[str, Any] = dict(hard or {})
        # `gates` is an accepted alias folded into the hard gates.
        if gates:
            self.hard.update(gates)
        self.soft: dict[str, Any] = dict(soft or {})
        self.penalty_weight = float(penalty_weight)
        self.cfg = cfg

    def __call__(self, scores: list[Score]) -> float:
        values = combine_scores(scores).values

        # A candidate with no finite primary metric was not properly scored (empty/partial
        # score, early cascade exit, or a failed prediction) -> reject, never rank it mid-pack.
        if self.primary not in values or not _finite(float(values[self.primary])):
            return _HARD_FAIL_FLOOR
        primary_value = float(values[self.primary])

        # --- HARD gates: a present metric that is violated OR non-finite rejects the candidate. ---
        for metric, bound in self.hard.items():
            if metric not in values:
                continue  # not yet computed (e.g. a later cascade stage); not a failure here
            if not bound_satisfied(float(values[metric]), bound):
                # Keep ordering defined among rejects via a tiny primary term.
                return _HARD_FAIL_FLOOR + primary_value

        # --- SOFT penalties: scaled hinge on each present metric (non-finite -> one unit). ---
        penalty = 0.0
        for metric, bound in self.soft.items():
            if metric not in values:
                continue
            value = float(values[metric])
            if not _finite(value):
                penalty += self.penalty_weight
                continue
            violation, scale = bound_violation(value, bound)
            if violation > 0.0:
                penalty += self.penalty_weight * violation / scale

        return primary_value - penalty


def _mean_over_present(scores: list[Score]) -> dict[str, float]:
    """Element-wise mean over the ``values`` of ``scores``, averaging where each key is present."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for score in scores:
        for key, value in score.values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def aggregate_sequences(
    scores_per_seq: list[Score],
    how: str = "max",
    primary: str = "ipsae",
    top_k: int = 1,
) -> Score:
    """Reduce the per-sequence scores for one backbone to a single representative :class:`Score`.

    Args:
        scores_per_seq: one :class:`Score` per designed sequence on the same backbone.
        how: reduction strategy over the primary metric:
            ``"max"`` carries the full score of the single best-primary sequence;
            ``"mean"`` returns the element-wise mean over all sequences;
            ``"topk"`` returns the element-wise mean over the ``top_k`` best-primary sequences.
        primary: the metric the ``max`` / ``topk`` selection ranks on.
        top_k: number of sequences averaged by the ``topk`` strategy.

    Returns:
        The aggregated :class:`Score`. An empty input returns an empty score.

    Raises:
        ValueError: on an unknown ``how``.
    """
    if not scores_per_seq:
        return Score(values={}, metadata={"aggregate": how, "n_seqs": 0})

    def primary_of(score: Score) -> float:
        return float(score.values.get(primary, float("-inf")))

    if how == "max":
        best = max(scores_per_seq, key=primary_of)
        meta = dict(best.metadata)
        meta.update({"aggregate": "max", "n_seqs": len(scores_per_seq)})
        return Score(values=dict(best.values), metadata=meta)

    if how == "mean":
        return Score(
            values=_mean_over_present(scores_per_seq),
            metadata={"aggregate": "mean", "n_seqs": len(scores_per_seq)},
        )

    if how == "topk":
        k = max(1, min(int(top_k), len(scores_per_seq)))
        ranked = sorted(scores_per_seq, key=primary_of, reverse=True)[:k]
        return Score(
            values=_mean_over_present(ranked),
            metadata={"aggregate": "topk", "top_k": k, "n_seqs": len(scores_per_seq)},
        )

    raise ValueError(f"unknown aggregation strategy {how!r}; expected 'max', 'mean' or 'topk'")
