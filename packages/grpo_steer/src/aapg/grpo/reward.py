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

from typing import Any

from aapg.core.scoring import REWARDS, Reward, Score, combine_scores

__all__ = ["IpSAEGatedReward", "aggregate_sequences"]

# Reward returned when any hard gate fails. A tiny +primary term is added on top so that even
# among rejected candidates the ordering by primary metric is still defined (useful for GRPO
# advantages on hard targets where every candidate fails a gate).
_HARD_FAIL_FLOOR = -1e3

# Lower bound on a bound's magnitude when it is used to normalize a soft hinge penalty.
_SCALE_EPSILON = 1e-8


def _bound_violation(value: float, bound: dict[str, Any]) -> tuple[float, float]:
    """Return ``(violation, scale)`` for ``value`` against a single ``bound`` spec.

    ``bound`` is one of ``{"min": x}``, ``{"max": x}`` or ``{"range": [lo, hi]}``. The violation
    is the (non-negative) amount by which ``value`` is outside the bound, and ``scale`` is a
    characteristic magnitude of the bound used to make penalties across different metrics
    comparable. A satisfied bound has violation ``0``.
    """
    violation = 0.0
    scale = 1.0
    if "range" in bound:
        lo, hi = float(bound["range"][0]), float(bound["range"][1])
        if value < lo:
            violation = lo - value
        elif value > hi:
            violation = value - hi
        scale = max(abs(hi - lo), _SCALE_EPSILON)
        return violation, scale
    if "min" in bound:
        lo = float(bound["min"])
        violation = max(0.0, lo - value)
        scale = max(abs(lo), _SCALE_EPSILON)
        return violation, scale
    if "max" in bound:
        hi = float(bound["max"])
        violation = max(0.0, value - hi)
        scale = max(abs(hi), _SCALE_EPSILON)
        return violation, scale
    return violation, scale


def _bound_satisfied(value: float, bound: dict[str, Any]) -> bool:
    """Whether ``value`` satisfies ``bound`` (a min/max/range spec)."""
    violation, _scale = _bound_violation(value, bound)
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
        primary_value = float(values.get(self.primary, 0.0))

        # --- HARD gates: any present-and-violated metric rejects the candidate. ---
        for metric, bound in self.hard.items():
            if metric not in values:
                continue  # missing metric is not a failure
            if not _bound_satisfied(float(values[metric]), bound):
                # Keep ordering defined among rejects via a tiny primary term.
                return _HARD_FAIL_FLOOR + primary_value

        # --- SOFT penalties: scaled hinge on each present-and-violated metric. ---
        penalty = 0.0
        for metric, bound in self.soft.items():
            if metric not in values:
                continue
            violation, scale = _bound_violation(float(values[metric]), bound)
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
