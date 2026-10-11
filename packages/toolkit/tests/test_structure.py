"""Tests for the gemmi-backed interface-clash helper (skipped when gemmi is absent)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# A two-chain complex: chain A (binder) at the origin, chain B (target) placed near or far.
_PDB = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      2  CA  ALA B   1     {bx:7.3f}   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)


def _write(bx: float) -> str:
    path = Path(tempfile.mkdtemp()) / "complex.pdb"
    path.write_text(_PDB.format(bx=bx))
    return str(path)


def test_interface_clash_detects_cross_chain_overlap() -> None:
    pytest.importorskip("gemmi")
    from aapg.toolkit.structure import interface_clash_from_cif

    # Chains 1.0 A apart -> steric clash (< 2.0 A cutoff).
    clashing = _write(1.0)
    assert interface_clash_from_cif(clashing, binder_chains=("A",), target_chains=("B",)) == 1.0

    # Chains 10 A apart -> no clash.
    clear = _write(10.0)
    assert interface_clash_from_cif(clear, binder_chains=("A",), target_chains=("B",)) == 0.0


def test_interface_clash_zero_when_a_chain_is_missing() -> None:
    pytest.importorskip("gemmi")
    from aapg.toolkit.structure import interface_clash_from_cif

    # Target chain "C" is absent -> no interface -> 0.0, never a false clash.
    path = _write(1.0)
    assert interface_clash_from_cif(path, binder_chains=("A",), target_chains=("C",)) == 0.0
