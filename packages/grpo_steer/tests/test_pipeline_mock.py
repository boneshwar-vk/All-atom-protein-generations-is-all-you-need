"""End-to-end SelectionPipeline test on the mock stack (CPU, no heavy imports)."""

from __future__ import annotations

import aapg.toolkit  # noqa: F401  (registers the mock scorers)
from aapg.core.scoring import SCORERS
from aapg.grpo.reward import IpSAEGatedReward
from aapg.grpo.selection import METRIC_COLUMNS, SelectionPipeline
from proteinmpnn.adapter import MockMPNNDesigner
from rfdiffusion3.adapter import MockRFD3Sampler, Target


def _pipeline() -> SelectionPipeline:
    sampler = MockRFD3Sampler()
    designer = MockMPNNDesigner()
    scorers = [
        SCORERS.create("protenix_v2_mock"),
        SCORERS.create("rosetta_interface_mock"),
    ]
    reward = IpSAEGatedReward(
        primary="ipsae",
        hard={"has_clash": {"max": 0}, "ipsae": {"min": 0.6}},
        soft={"iptm": {"min": 0.6}, "interface_dg": {"max": 0}},
    )
    cfg = {
        "n_backbones": 4,
        "seqs_per_backbone": 2,
        "n_rounds": 1,
        "top_k": 100,
        "group_size": 64,
        "binder_chains": ("A",),
        "aggregate": "max",
        "seed": 0,
    }
    return SelectionPipeline(sampler, designer, scorers, reward, cfg)


def _rows(table: object) -> list[dict]:
    if hasattr(table, "to_dicts"):  # polars DataFrame
        return table.to_dicts()  # type: ignore[attr-defined]
    assert isinstance(table, list)
    return table


def test_pipeline_returns_all_candidates():
    table = _pipeline().run(Target(binder_length=20))
    rows = _rows(table)
    assert len(rows) == 4 * 2  # n_backbones * seqs_per_backbone


def test_pipeline_has_all_metric_and_rank_columns():
    table = _pipeline().run(Target(binder_length=20))
    rows = _rows(table)
    expected = {"name", "binder_sequence", "round", "reward", "advantage", *METRIC_COLUMNS}
    for row in rows:
        assert expected.issubset(row.keys())
        # scRMSD was computed from the mock backbone coords.
        assert row["scrmsd"] is not None
        assert row["ipsae"] is not None


def test_pipeline_is_deterministic():
    rows_a = _rows(_pipeline().run(Target(binder_length=20)))
    rows_b = _rows(_pipeline().run(Target(binder_length=20)))
    seqs_a = sorted(r["binder_sequence"] for r in rows_a)
    seqs_b = sorted(r["binder_sequence"] for r in rows_b)
    assert seqs_a == seqs_b


def test_pipeline_ranked_by_advantage():
    rows = _rows(_pipeline().run(Target(binder_length=20)))
    advantages = [r["advantage"] for r in rows]
    assert advantages == sorted(advantages, reverse=True)


def test_top_k_limits_kept_backbones():
    sampler = MockRFD3Sampler()
    designer = MockMPNNDesigner()
    scorers = [SCORERS.create("protenix_v2_mock"), SCORERS.create("rosetta_interface_mock")]
    reward = IpSAEGatedReward(primary="ipsae", hard={"ipsae": {"min": 0.6}})
    cfg = {
        "n_backbones": 4,
        "seqs_per_backbone": 2,
        "n_rounds": 1,
        "top_k": 100,
        "group_size": 64,
        "seed": 1,
    }
    pipeline = SelectionPipeline(sampler, designer, scorers, reward, cfg)
    rows = _rows(pipeline.run(Target(binder_length=16)))
    assert len(rows) == 8
