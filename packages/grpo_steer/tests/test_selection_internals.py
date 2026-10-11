"""Unit tests for SelectionPipeline internals: max-reward aggregation and scRMSD fail-closed."""

from __future__ import annotations

import numpy as np

from aapg.core.scoring import Score
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

    class _Cand:
        name = "c0"
        metadata = {"backbone_ca_coords": np.zeros((10, 3))}  # 10 != 3 -> RMSD superposition fails

    merged = pipe._candidate_score(_Cand())
    assert merged.values["scrmsd"] == float("inf")
    assert "scrmsd_error" in merged.metadata
