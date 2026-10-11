"""Tests for the rfdiffusion3 adapter (mock sampler + stdlib PDB parser; no heavy deps)."""

from __future__ import annotations

import numpy as np

from rfdiffusion3.adapter import MockRFD3Sampler, Target, get_sampler, parse_pdb_ca


def test_rfdiffusion3_imports():
    import rfdiffusion3

    # Upstream RFD3 (foundry) is not vendored here; the adapter drives it as a subprocess.
    assert rfdiffusion3.__vendored__ is False


def test_mock_sampler_deterministic_with_ca_coords():
    s = get_sampler(mock=True)
    a = s.sample_backbones(Target(binder_length=12), n=3, seed=0)
    b = MockRFD3Sampler().sample_backbones(Target(binder_length=12), n=3, seed=0)
    assert len(a) == 3
    assert all(bb.length == 12 for bb in a)
    assert a[0].metadata["ca_coords"].shape == (12, 3)
    assert np.allclose(a[0].metadata["ca_coords"], b[0].metadata["ca_coords"])


def test_parse_pdb_ca_filters_chains():
    pdb = (
        "ATOM      1  N   ALA A   1      11.000  11.000  11.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      12.000  13.000  14.000  1.00  0.00           C\n"
        "ATOM      3  CA  GLY B   1      20.000  20.000  20.000  1.00  0.00           C\n"
        "ATOM      4  CA  SER A   2      15.000  16.000  17.000  1.00  0.00           C\n"
    )
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / "x.pdb"
    p.write_text(pdb)

    # Binder = everything NOT chain B -> chain A's two CA atoms.
    coords, chains = parse_pdb_ca(str(p), exclude_chains=("B",))
    assert chains == ["A", "A"]
    assert np.allclose(coords[0], [12.0, 13.0, 14.0])
    # include_chains picks exactly one chain.
    only_b, chains_b = parse_pdb_ca(str(p), include_chains=("B",))
    assert chains_b == ["B"]
    assert np.allclose(only_b[0], [20.0, 20.0, 20.0])
