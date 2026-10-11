"""Tests for per-sequence score aggregation."""

from __future__ import annotations

from aapg.core.scoring import Score
from aapg.grpo.reward import aggregate_sequences


def _scores() -> list[Score]:
    return [
        Score(values={"ipsae": 0.3, "iptm": 0.5}, metadata={"seq": 0}),
        Score(values={"ipsae": 0.9, "iptm": 0.7}, metadata={"seq": 1}),
        Score(values={"ipsae": 0.6, "iptm": 0.6}, metadata={"seq": 2}),
    ]


def test_max_carries_best_primary_full_score():
    agg = aggregate_sequences(_scores(), how="max", primary="ipsae")
    assert agg.values["ipsae"] == 0.9
    assert agg.values["iptm"] == 0.7  # the winning seq's own iptm, not the max iptm
    assert agg.metadata["aggregate"] == "max"


def test_mean_is_elementwise():
    agg = aggregate_sequences(_scores(), how="mean", primary="ipsae")
    assert abs(agg.values["ipsae"] - (0.3 + 0.9 + 0.6) / 3) < 1e-9
    assert abs(agg.values["iptm"] - (0.5 + 0.7 + 0.6) / 3) < 1e-9


def test_topk_averages_top_k_by_primary():
    agg = aggregate_sequences(_scores(), how="topk", primary="ipsae", top_k=2)
    # Top-2 ipsae: 0.9 and 0.6 -> mean 0.75; their iptm: 0.7 and 0.6 -> mean 0.65.
    assert abs(agg.values["ipsae"] - 0.75) < 1e-9
    assert abs(agg.values["iptm"] - 0.65) < 1e-9
    assert agg.metadata["top_k"] == 2


def test_empty_returns_empty_score():
    agg = aggregate_sequences([], how="max")
    assert agg.values == {}


def test_mean_handles_missing_keys():
    scores = [Score(values={"a": 2.0}), Score(values={"a": 4.0, "b": 10.0})]
    agg = aggregate_sequences(scores, how="mean", primary="a")
    assert agg.values["a"] == 3.0
    assert agg.values["b"] == 10.0  # averaged only where present
