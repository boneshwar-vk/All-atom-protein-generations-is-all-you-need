"""Tests for the config-switchable sequence-designer factory (aapg.grpo.designers)."""

from __future__ import annotations

import pytest

from aapg.grpo.designers import (
    MockProtonMPNNDesigner,
    ProtonMPNNDesigner,
    get_designer,
)


class _Backbone:
    """Minimal backbone stub exposing the attributes the mock designers read."""

    length = 24
    path = "/tmp/backbone_0.pdb"


def test_get_designer_dispatches_by_name():
    from proteinmpnn.adapter import MockMPNNDesigner, MPNNDesigner

    assert isinstance(get_designer("proteinmpnn", mock=True), MockMPNNDesigner)
    assert isinstance(get_designer("proteinmpnn", mock=False), MPNNDesigner)
    assert isinstance(get_designer("protonmpnn", mock=True), MockProtonMPNNDesigner)
    assert isinstance(get_designer("protonmpnn", mock=False), ProtonMPNNDesigner)


def test_default_name_is_proteinmpnn():
    from proteinmpnn.adapter import MockMPNNDesigner

    assert isinstance(get_designer(mock=True), MockMPNNDesigner)


def test_mock_protonmpnn_differs_from_proteinmpnn():
    backbone = _Backbone()
    proteinmpnn = get_designer("proteinmpnn", mock=True).design(backbone, n_seqs=4, seed=0)
    protonmpnn = get_designer("protonmpnn", mock=True).design(backbone, n_seqs=4, seed=0)
    assert len(proteinmpnn) == len(protonmpnn) == 4
    assert all(len(s) == _Backbone.length for s in proteinmpnn + protonmpnn)
    # Distinct hash salt => the switch is observable on the same backbone/seed.
    assert proteinmpnn != protonmpnn


def test_mock_protonmpnn_is_deterministic():
    backbone = _Backbone()
    a = get_designer("protonmpnn", mock=True).design(backbone, n_seqs=3, seed=7)
    b = get_designer("protonmpnn", mock=True).design(backbone, n_seqs=3, seed=7)
    assert a == b


def test_unknown_designer_raises():
    with pytest.raises(ValueError, match="unknown designer"):
        get_designer("nope")


def test_protonmpnn_real_path_is_placeholder():
    designer = get_designer("protonmpnn", mock=False)
    with pytest.raises(RuntimeError, match="placeholder"):
        designer.design(_Backbone(), n_seqs=2)


def test_mock_selection_run_with_protonmpnn():
    """A full mock selection run switched to the ProtonMPNN backend works end-to-end."""
    from aapg.grpo.cli import default_cfg
    from aapg.grpo.trainer import GRPOTrainer

    cfg = default_cfg()
    cfg["n_backbones"] = 2
    cfg["seqs_per_backbone"] = 2
    cfg["designer"] = {"name": "protonmpnn", "temperature": 0.1}

    table = GRPOTrainer(cfg).run()
    rows = table.to_dicts() if hasattr(table, "to_dicts") else table
    assert len(rows) == 2 * 2  # n_backbones * seqs_per_backbone
