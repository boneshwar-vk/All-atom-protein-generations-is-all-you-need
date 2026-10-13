"""End-to-end SelectionPipeline test on the mock stack (CPU, no heavy imports)."""

from __future__ import annotations

import json

import pytest

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
    expected = {
        "name",
        "binder_sequence",
        "round",
        "selected",
        "reward",
        "advantage",
        *METRIC_COLUMNS,
    }
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


def test_top_k_marks_only_winners_selected():
    sampler = MockRFD3Sampler()
    designer = MockMPNNDesigner()
    scorers = [SCORERS.create("protenix_v2_mock"), SCORERS.create("rosetta_interface_mock")]
    reward = IpSAEGatedReward(primary="ipsae", hard={"ipsae": {"min": 0.6}})
    cfg = {
        "n_backbones": 4,
        "seqs_per_backbone": 2,
        "n_rounds": 1,
        "top_k": 2,  # keep 2 of 4 backbones
        "group_size": 64,
        "seed": 1,
    }
    pipeline = SelectionPipeline(sampler, designer, scorers, reward, cfg)
    rows = _rows(pipeline.run(Target(binder_length=16)))
    assert len(rows) == 8  # all candidates recorded
    # Exactly top_k backbones (2) are selected; each contributes seqs_per_backbone (2) rows.
    selected_backbones = {r["name"].rsplit("_s", 1)[0] for r in rows if r["selected"]}
    assert len(selected_backbones) == 2
    assert sum(1 for r in rows if r["selected"]) == 4


def _write_pipeline(output_path: str) -> SelectionPipeline:
    scorers = [SCORERS.create("protenix_v2_mock"), SCORERS.create("rosetta_interface_mock")]
    cfg = {"n_backbones": 2, "seqs_per_backbone": 2, "group_size": 64, "output_path": output_path}
    return SelectionPipeline(
        MockRFD3Sampler(), MockMPNNDesigner(), scorers, IpSAEGatedReward(primary="ipsae"), cfg
    )


def test_jsonl_output_path_is_real_jsonl(tmp_path):
    """A .jsonl output_path must contain JSON Lines regardless of whether polars is installed."""
    out = tmp_path / "ranked.jsonl"
    _write_pipeline(str(out)).run(Target(binder_length=12))
    assert out.exists()
    lines = out.read_text().splitlines()
    assert lines, "no rows written"
    parsed = [json.loads(line) for line in lines]  # must not raise (would raise on Parquet bytes)
    assert len(parsed) == 4  # 2 backbones x 2 seqs
    assert all(m in parsed[0] for m in METRIC_COLUMNS)


def test_parquet_output_path_is_parquet_when_polars_present(tmp_path):
    pytest.importorskip("polars")
    out = tmp_path / "ranked.parquet"
    _write_pipeline(str(out)).run(Target(binder_length=12))
    assert out.read_bytes()[:4] == b"PAR1"  # Parquet magic bytes
