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
        "seq_reward",
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


def test_pipeline_ranked_by_seq_reward():
    # The final table is ranked by absolute per-design reward (best first), NOT by advantage.
    rows = _rows(_pipeline().run(Target(binder_length=20)))
    seq_rewards = [r["seq_reward"] for r in rows]
    assert seq_rewards == sorted(seq_rewards, reverse=True)


def test_selected_is_one_gate_passing_sequence_per_winning_backbone():
    sampler = MockRFD3Sampler()
    designer = MockMPNNDesigner()
    scorers = [SCORERS.create("protenix_v2_mock"), SCORERS.create("rosetta_interface_mock")]
    # Permissive gate so some designs pass; selection keeps 2 of 4 backbones.
    reward = IpSAEGatedReward(primary="ipsae", hard={"ipsae": {"min": 0.3}})
    cfg = {
        "n_backbones": 4,
        "seqs_per_backbone": 2,
        "n_rounds": 1,
        "top_k": 2,
        "group_size": 64,
        "seed": 1,
    }
    rows = _rows(
        SelectionPipeline(sampler, designer, scorers, reward, cfg).run(Target(binder_length=16))
    )
    assert len(rows) == 8  # all candidates recorded
    selected = [r for r in rows if r["selected"]]
    # At most one selected row per winning backbone (its best, gate-passing sequence).
    backbone_of = lambda r: r["name"].rsplit("_seq", 1)[0]  # noqa: E731
    assert len(selected) <= 2
    assert len({backbone_of(r) for r in selected}) == len(selected)  # one per backbone
    from aapg.grpo.reward import passed_hard_gates

    for r in selected:
        assert passed_hard_gates(float(r["seq_reward"]))  # never a hard-gate failure


def test_selected_excludes_hard_gate_failures():
    # An impossible hard gate => every design fails => nothing is selected (no "least-bad" winner).
    sampler = MockRFD3Sampler()
    designer = MockMPNNDesigner()
    scorers = [SCORERS.create("protenix_v2_mock"), SCORERS.create("rosetta_interface_mock")]
    reward = IpSAEGatedReward(primary="ipsae", hard={"ipsae": {"min": 1.01}})  # unsatisfiable
    cfg = {"n_backbones": 4, "seqs_per_backbone": 2, "top_k": 2, "group_size": 64, "seed": 1}
    rows = _rows(
        SelectionPipeline(sampler, designer, scorers, reward, cfg).run(Target(binder_length=16))
    )
    assert rows  # candidates are still recorded
    assert not any(r["selected"] for r in rows)  # but none is marked a winner


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
