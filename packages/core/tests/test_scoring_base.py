"""Tests for the scoring base: Candidate, combine_scores, and registered Scorer/Reward."""

from __future__ import annotations

from aapg.core.registry import Registry
from aapg.core.scoring import Candidate, Reward, Score, Scorer, combine_scores


def test_candidate_defaults() -> None:
    cand = Candidate()
    assert cand.structure_path is None
    assert cand.binder_chains == ("A",)
    assert cand.target_chains == ("B",)
    assert cand.binder_sequence is None
    assert cand.name is None
    assert cand.metadata == {}


def test_candidate_metadata_not_shared() -> None:
    a = Candidate()
    b = Candidate()
    a.metadata["x"] = 1
    assert b.metadata == {}


def test_combine_scores_merges_values_and_metadata() -> None:
    s1 = Score(values={"a": 1.0, "b": 2.0}, metadata={"src": "one"})
    s2 = Score(values={"c": 3.0}, metadata={"step": 7})
    combined = combine_scores([s1, s2])
    assert combined.values == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert combined.metadata == {"src": "one", "step": 7}


def test_combine_scores_override_on_clash() -> None:
    s1 = Score(values={"a": 1.0}, metadata={"k": "first"})
    s2 = Score(values={"a": 9.0}, metadata={"k": "second"})
    combined = combine_scores([s1, s2])
    assert combined.values == {"a": 9.0}
    assert combined.metadata == {"k": "second"}


def test_combine_scores_empty_list() -> None:
    combined = combine_scores([])
    assert combined.values == {}
    assert combined.metadata == {}


def test_combine_scores_does_not_mutate_inputs() -> None:
    s1 = Score(values={"a": 1.0})
    s2 = Score(values={"b": 2.0})
    combine_scores([s1, s2])
    assert s1.values == {"a": 1.0}
    assert s2.values == {"b": 2.0}


def test_concrete_scorer_and_reward_via_local_registries() -> None:
    scorers: Registry[Scorer] = Registry("scorer")
    rewards: Registry[Reward] = Registry("reward")

    @scorers.register("length")
    class LengthScorer(Scorer):
        name = "length"

        def score(self, candidate: Candidate) -> Score:
            seq = candidate.binder_sequence or ""
            return Score(values={"length": float(len(seq))}, metadata={"name": candidate.name})

    @rewards.register("sum")
    class SumReward(Reward):
        name = "sum"

        def __call__(self, scores: list[Score]) -> float:
            return sum(v for s in scores for v in s.values.values())

    scorer = scorers.create("length")
    cand = Candidate(binder_sequence="ACDEF", name="cand0")
    score = scorer.score(cand)
    assert score.values == {"length": 5.0}
    assert score.metadata == {"name": "cand0"}

    reward = rewards.create("sum")
    assert reward([score, Score(values={"extra": 2.0})]) == 7.0
