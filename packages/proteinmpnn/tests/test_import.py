"""Tests for the vendored ProteinMPNN fork (no torch needed)."""

from __future__ import annotations

from pathlib import Path


def test_proteinmpnn_vendored():
    import proteinmpnn

    assert proteinmpnn.__vendored__ is True


def test_vendored_source_and_weights_present():
    import proteinmpnn

    vendor = Path(proteinmpnn.__file__).resolve().parent / "vendor"
    assert (vendor / "protein_mpnn_run.py").exists()
    assert (vendor / "protein_mpnn_utils.py").exists()
    # Standard binder-design weights are present.
    assert (vendor / "vanilla_model_weights" / "v_48_020.pt").exists()
    assert (vendor / "soluble_model_weights" / "v_48_020.pt").exists()


def test_mock_designer_is_deterministic():
    from proteinmpnn.adapter import MockMPNNDesigner

    class _BB:
        path = "mock://bb0"
        length = 24

    d = MockMPNNDesigner()
    a = d.design(_BB(), n_seqs=3, seed=0)
    b = d.design(_BB(), n_seqs=3, seed=0)
    assert a == b
    assert len(a) == 3
    assert all(len(seq) == 24 for seq in a)


def test_parse_mpnn_fasta(tmp_path):
    from proteinmpnn.adapter import _parse_mpnn_fasta

    fa = tmp_path / "x.fa"
    # First record = native/input (dropped); next two = designs (one multi-chain).
    fa.write_text(
        ">native, score=1.0\nMKV\n"
        ">T=0.1, sample=1, score=0.5\nAAAA\n"
        ">T=0.1, sample=2, score=0.4\nCCC/DDD\n"
    )
    designs = _parse_mpnn_fasta(fa)
    assert designs == ["AAAA", "CCCDDD"]  # native dropped, '/' chain-join collapsed
