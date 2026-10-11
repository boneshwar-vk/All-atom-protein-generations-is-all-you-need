"""Unit tests for SelectionPipeline internals: max-reward aggregation and scRMSD fail-closed."""

from __future__ import annotations

import numpy as np

from aapg.core.scoring import Candidate, Score
from aapg.grpo.reward import IpSAEGatedReward
from aapg.grpo.selection import SelectionPipeline


def _pipeline(aggregate: str = "max") -> SelectionPipeline:
    reward = IpSAEGatedReward(
        primary="ipsae", hard={"has_clash": {"max": 0}, "ipsae": {"min": 0.5}}
    )
    cfg = {"aggregate": aggregate}
    # sampler/designer/scorers are unused by the methods under test.
    return SelectionPipeline(sampler=None, designer=None, scorers=[], reward=reward, cfg=cfg)


def test_backbone_reward_max_prefers_gated_passing_sequence() -> None:
    """A high-ipSAE sequence that fails a hard gate must not mask a passing lower-ipSAE one.

    Sequence 0 has the higher ipSAE (0.9) but clashes (fails the has_clash gate); sequence 1 has
    a lower ipSAE (0.7) and passes every gate. Best-of-N selection should reward this backbone by
    the PASSING sequence (0.7), not the rejected max-ipSAE sequence.
    """
    seq_scores = [
        Score(values={"ipsae": 0.9, "has_clash": 1.0}),  # clashes -> rejected
        Score(values={"ipsae": 0.7, "has_clash": 0.0}),  # passes -> reward 0.7
    ]
    reward = _pipeline("max")._backbone_reward(seq_scores)
    assert reward == 0.7


def test_backbone_reward_empty_returns_floor() -> None:
    reward = _pipeline("max")._backbone_reward([])
    # No sequences -> reward of an empty score, which is the hard-fail floor (no primary metric).
    assert reward < -100.0


def test_scrmsd_failure_is_fail_closed_infinite() -> None:
    """A shape mismatch between designed and predicted CA traces yields inf, not a crash."""

    class _BadScorer:
        name = "bad"

        def score(self, candidate):  # noqa: ANN001
            # Predict a CA trace with a DIFFERENT residue count than the designed backbone.
            return Score(
                values={"ipsae": 0.8},
                metadata={"predicted_ca_coords": np.zeros((3, 3))},
            )

    pipe = SelectionPipeline(
        sampler=None,
        designer=None,
        scorers=[_BadScorer()],
        reward=IpSAEGatedReward(primary="ipsae"),
        cfg={},
    )
    # 10 designed CA vs 3 predicted CA -> Kabsch superposition raises -> fail-closed inf.
    cand = Candidate(
        structure_path=None,
        binder_chains=("A",),
        target_chains=("B",),
        name="c0",
        metadata={"backbone_ca_coords": np.zeros((10, 3))},
    )
    merged = pipe._candidate_score(cand)
    assert merged.values["scrmsd"] == float("inf")
    assert "scrmsd_error" in merged.metadata


def test_cascade_threads_predicted_cif_to_downstream_scorer() -> None:
    """A structure stage's cif_path must become the next stage's candidate.structure_path."""
    seen: dict[str, object] = {}

    class _StructScorer:
        name = "struct"

        def score(self, candidate):  # noqa: ANN001
            return Score(
                values={"ipsae": 0.8},
                metadata={
                    "cif_path": "/tmp/predicted_complex.cif",
                    "predicted_ca_coords": np.zeros((5, 3)),
                },
            )

    class _DownstreamScorer:
        name = "down"

        def score(self, candidate):  # noqa: ANN001
            seen["structure_path"] = candidate.structure_path
            return Score(values={"interface_dg": -10.0})

    pipe = SelectionPipeline(
        sampler=None,
        designer=None,
        scorers=[_StructScorer(), _DownstreamScorer()],
        reward=IpSAEGatedReward(primary="ipsae"),
        cfg={},
    )
    cand = Candidate(
        structure_path="/backbone.pdb",
        binder_chains=("A",),
        target_chains=("B",),
        name="c0",
        metadata={"backbone_ca_coords": np.zeros((5, 3))},
    )
    pipe._candidate_score(cand)
    # Rosetta et al. must analyse the predicted complex, not the sequence-less RFD3 backbone.
    assert seen["structure_path"] == "/tmp/predicted_complex.cif"


def test_scrmsd_not_fabricated_for_real_coordless_scorer() -> None:
    """A real backend emitting no coordinates must leave scRMSD unset, never a fabricated pass."""

    class _RealHead:
        name = "structure_head"

        def score(self, candidate):  # noqa: ANN001
            return Score(values={"ipsae": 0.8}, metadata={})  # real, no coords, no mock flag

    pipe = SelectionPipeline(
        sampler=None,
        designer=None,
        scorers=[_RealHead()],
        reward=IpSAEGatedReward(primary="ipsae"),
        cfg={},
    )
    cand = Candidate(
        binder_chains=("A",),
        target_chains=("B",),
        name="c0",
        metadata={"backbone_ca_coords": np.zeros((8, 3))},
    )
    merged = pipe._candidate_score(cand)
    assert "scrmsd" not in merged.values


def test_scrmsd_fabricated_only_for_mock_scorer() -> None:
    class _Mock:
        name = "m"

        def score(self, candidate):  # noqa: ANN001
            return Score(values={"ipsae": 0.8}, metadata={"mock": True})

    pipe = SelectionPipeline(
        sampler=None,
        designer=None,
        scorers=[_Mock()],
        reward=IpSAEGatedReward(primary="ipsae"),
        cfg={},
    )
    cand = Candidate(
        binder_chains=("A",),
        target_chains=("B",),
        name="c0",
        metadata={"backbone_ca_coords": np.zeros((8, 3))},
    )
    merged = pipe._candidate_score(cand)
    assert "scrmsd" in merged.values and merged.values["scrmsd"] >= 0.0


def test_aggregate_mean_ors_has_clash() -> None:
    """mean/topk aggregation must OR the boolean has_clash, not average it to a fraction."""
    from aapg.grpo.reward import aggregate_sequences

    scores = [
        Score(values={"ipsae": 0.8, "has_clash": 0.0}),
        Score(values={"ipsae": 0.6, "has_clash": 1.0}),
    ]
    agg = aggregate_sequences(scores, how="mean")
    assert agg.values["has_clash"] == 1.0  # OR over the group, not 0.5
    assert agg.values["ipsae"] == 0.7  # genuine mean of the continuous metric
