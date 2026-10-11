"""Tests for build_candidate: binder-chain source of truth and its loud guards."""

from __future__ import annotations

import pytest

from aapg.grpo.candidate import build_candidate
from rfdiffusion3.adapter import MockRFD3Sampler, Target


def _backbone(target: Target, binder_chains: tuple[str, ...] | None = None):
    bb = MockRFD3Sampler().sample_backbones(target, n=1)[0]
    if binder_chains is not None:
        bb.metadata["binder_chains"] = binder_chains
    return bb


def test_build_candidate_uses_recorded_binder_chains_by_default():
    target = Target(chains=("A",), binder_length=8)
    backbone = _backbone(target)  # mock records binder chain "B" (first id not in target)
    cand = build_candidate(backbone, sequence="ACDEFGHI", target=target, name="c0")
    assert cand.binder_chains == ("B",)
    assert cand.target_chains == ("A",)


def test_build_candidate_explicit_chains_override_metadata():
    target = Target(chains=("A",), binder_length=8)
    backbone = _backbone(target)
    cand = build_candidate(
        backbone, sequence="ACDEFGHI", target=target, name="c0", binder_chains=("C",)
    )
    assert cand.binder_chains == ("C",)


def test_build_candidate_rejects_binder_target_overlap():
    target = Target(chains=("A",), binder_length=8)
    backbone = _backbone(target, binder_chains=("A",))  # overlaps the target chain
    with pytest.raises(ValueError, match="overlap target chains"):
        build_candidate(backbone, sequence="ACDEFGHI", target=target, name="c0")


def test_build_candidate_rejects_empty_binder_chains():
    target = Target(chains=("A",), binder_length=8)
    backbone = _backbone(target, binder_chains=())
    # Empty recorded chains fall back to ("A",), which then overlaps the target -> still rejected.
    with pytest.raises(ValueError):
        build_candidate(backbone, sequence="ACDEFGHI", target=target, name="c0")


def test_build_candidate_rejects_multichain_binder_in_v1():
    target = Target(chains=("A",), binder_length=8)
    backbone = _backbone(target, binder_chains=("B", "C"))
    with pytest.raises(NotImplementedError, match="multi-chain binder"):
        build_candidate(backbone, sequence="ACDEFGHI", target=target, name="c0")
