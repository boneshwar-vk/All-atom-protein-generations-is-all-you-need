"""End-to-end stage-wise runner test on the CPU mock stack (no GPU/weights/heavy deps).

Drives all five stages (rfd3 -> mpnn -> fold -> rosetta -> reward) via ``stagerun.main`` in a
tmp work dir and asserts each manifest is written with the right shape, plus a resume check that
re-running a stage adds no duplicates. Kept fast with small N/K.
"""

from __future__ import annotations

import json

from aapg.grpo import stagerun

N = 3  # backbones
K = 2  # sequences per backbone
LENGTH = 12


def _common(run: str) -> list[str]:
    return ["--run", run, "--mock", "--binder-length", str(LENGTH)]


def _run_all(run: str) -> None:
    assert stagerun.main(["rfd3", *_common(run), "--n", str(N), "--seed", "0"]) == 0
    assert stagerun.main(["mpnn", *_common(run), "--seqs-per", str(K)]) == 0
    assert stagerun.main(["fold", *_common(run)]) == 0
    assert stagerun.main(["rosetta", *_common(run)]) == 0
    assert stagerun.main(["reward", *_common(run), "--output", f"{run}/ranked.jsonl"]) == 0


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_stagewise_pipeline_mock(tmp_path):
    run = tmp_path / "egfr_mac"
    _run_all(str(run))

    # Stage 1: backbones.json has N rows and each binder CA trace is on disk.
    backbones = json.loads((run / "backbones.json").read_text())
    assert len(backbones) == N
    for bb in backbones:
        assert (run / "backbones" / f"{bb['id']}.npy").exists()

    # Stage 2: N*K sequences, cand ids unique.
    seqs = _read_jsonl(run / "sequences.jsonl")
    assert len(seqs) == N * K
    assert len({row["cand"] for row in seqs}) == N * K

    # Stage 3: N*K structure rows, each carrying an ipsae metric.
    structures = _read_jsonl(run / "structure.jsonl")
    assert len(structures) == N * K
    assert all(row.get("ipsae") is not None for row in structures)
    # Mock structure scorer emits no CIF / predicted CA -> scRMSD is left uncomputed.
    assert all(row["cif"] is None for row in structures)

    # Stage 4: rosetta manifest present with one row per candidate.
    rosetta = _read_jsonl(run / "rosetta.jsonl")
    assert len(rosetta) == N * K
    assert all("interface_dg" in row for row in rosetta)

    # Stage 5: ranked.jsonl is valid JSON Lines sorted by reward descending.
    ranked = _read_jsonl(run / "ranked.jsonl")
    assert len(ranked) == N * K
    rewards = [row["reward"] for row in ranked]
    assert rewards == sorted(rewards, reverse=True)


def test_resume_skips_completed_ids(tmp_path):
    run = tmp_path / "egfr_mac"
    _run_all(str(run))

    seq_path = run / "sequences.jsonl"
    before = _read_jsonl(seq_path)

    # Re-running mpnn must skip every backbone already in the manifest (no duplicates appended).
    assert stagerun.main(["mpnn", *_common(str(run)), "--seqs-per", str(K)]) == 0
    after = _read_jsonl(seq_path)
    assert len(after) == len(before)
    cands = [row["cand"] for row in after]
    assert len(cands) == len(set(cands))  # still unique

    # Re-running fold must likewise add no duplicate candidates.
    assert stagerun.main(["fold", *_common(str(run))]) == 0
    structures = _read_jsonl(run / "structure.jsonl")
    struct_cands = [row["cand"] for row in structures]
    assert len(struct_cands) == len(set(struct_cands)) == N * K


def test_enough_memory_true_when_undeterminable(monkeypatch):
    """When memory cannot be measured, the guard must not spuriously bail out."""
    monkeypatch.setattr(stagerun, "available_memory_gb", lambda: None)
    assert stagerun.enough_memory(999.0) is True


def test_rfd3_partial_resume_generates_only_remainder(tmp_path):
    """Re-running rfd3 with a larger N tops up the missing backbones (no dup, no full re-run)."""
    run = tmp_path / "egfr_mac"
    assert stagerun.main(["rfd3", *_common(str(run)), "--n", "2", "--seed", "0"]) == 0
    first = json.loads((run / "backbones.json").read_text())
    assert [b["id"] for b in first] == ["bb0000", "bb0001"]

    assert stagerun.main(["rfd3", *_common(str(run)), "--n", "4", "--seed", "0"]) == 0
    second = json.loads((run / "backbones.json").read_text())
    ids = [b["id"] for b in second]
    assert ids == ["bb0000", "bb0001", "bb0002", "bb0003"]  # topped up, in order
    assert len(ids) == len(set(ids)) == 4
    for b in second:
        assert (run / "backbones" / f"{b['id']}.npy").exists()
    # The first two backbones are untouched by the resume.
    assert second[:2] == first


def test_mpnn_resume_per_candidate_backfills_missing_seq(tmp_path):
    """A backbone left with < K sequences (mid-backbone crash) is backfilled, not skipped."""
    run = tmp_path / "egfr_mac"
    assert stagerun.main(["rfd3", *_common(str(run)), "--n", str(N), "--seed", "0"]) == 0
    assert stagerun.main(["mpnn", *_common(str(run)), "--seqs-per", str(K)]) == 0

    seq_path = run / "sequences.jsonl"
    rows = _read_jsonl(seq_path)
    # Simulate a crash that dropped ONE candidate of bb0000 (its last appended seq).
    dropped = "bb0000_seq01"
    kept = [r for r in rows if r["cand"] != dropped]
    assert len(kept) == N * K - 1
    seq_path.write_text("".join(json.dumps(r) + "\n" for r in kept))

    assert stagerun.main(["mpnn", *_common(str(run)), "--seqs-per", str(K)]) == 0
    after = _read_jsonl(seq_path)
    cands = [r["cand"] for r in after]
    assert len(cands) == N * K  # invariant restored
    assert len(set(cands)) == N * K  # no duplicate
    assert dropped in set(cands)  # the missing one was backfilled


def test_reward_default_output_no_crash(tmp_path):
    """reward with no --output must write a ranked file even on a no-polars install (not crash)."""
    run = tmp_path / "egfr_mac"
    assert stagerun.main(["rfd3", *_common(str(run)), "--n", str(N), "--seed", "0"]) == 0
    assert stagerun.main(["mpnn", *_common(str(run)), "--seqs-per", str(K)]) == 0
    assert stagerun.main(["fold", *_common(str(run))]) == 0
    assert stagerun.main(["rosetta", *_common(str(run))]) == 0
    # No --output: must not raise. Writer picks parquet if polars present, else jsonl.
    assert stagerun.main(["reward", *_common(str(run))]) == 0
    produced = list(run.glob("ranked*"))
    assert produced, "reward wrote no ranked table"


def test_read_jsonl_tolerates_torn_trailing_line(tmp_path):
    """A truncated FINAL line (hard kill mid-append) is skipped; a bad non-final line raises."""
    good = tmp_path / "good.jsonl"
    good.write_text('{"cand": "a"}\n{"cand": "b"}\n{"cand": "c"')  # torn last line
    rows = stagerun._read_jsonl(good)
    assert [r["cand"] for r in rows] == ["a", "b"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"cand": "a"}\n{oops not json}\n{"cand": "c"}\n')  # corrupt MIDDLE line
    try:
        stagerun._read_jsonl(bad)
        raise AssertionError("expected a JSONDecodeError on a corrupt non-final line")
    except json.JSONDecodeError:
        pass
