"""Test the aapg-grpo CLI argparse fallback runs the mock pipeline."""

from __future__ import annotations

from aapg.grpo.cli import main


def test_cli_mock_runs_and_exits_zero():
    code = main(["steering.mock=true", "n_backbones=2", "seqs_per_backbone=2"])
    assert code == 0


def test_cli_default_overrides_small_sizes(capsys):
    code = main(["steering.mock=true", "n_backbones=3", "target.binder_length=12"])
    assert code == 0
    out = capsys.readouterr().out
    assert "selection complete" in out


def test_cli_writes_table(tmp_path, capsys):
    out_path = tmp_path / "ranked"
    code = main(
        [
            "steering.mock=true",
            "n_backbones=2",
            "seqs_per_backbone=2",
            f"output_path={out_path}",
        ]
    )
    assert code == 0
    # JSONL when polars absent, Parquet otherwise; either way a file exists.
    written = list(tmp_path.glob("ranked.*"))
    assert len(written) == 1


def test_cli_bad_override_errors():
    try:
        main(["not_a_kv_pair"])
    except SystemExit as exc:
        assert exc.code != 0
        return
    raise AssertionError("expected SystemExit on malformed override")
