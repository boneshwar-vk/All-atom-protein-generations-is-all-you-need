"""Tests for group-relative advantage normalization."""

from __future__ import annotations

import numpy as np

from aapg.grpo.advantage import group_relative_advantage


def test_varied_input_is_mean_zero_unit_std():
    rewards = [1.0, 2.0, 3.0, 4.0, 10.0]
    adv = group_relative_advantage(rewards)
    arr = np.asarray(adv)
    assert arr.shape == (5,)
    assert abs(float(arr.mean())) < 1e-9
    # Population std (the normalization used) is 1 for the standardized values.
    assert abs(float(arr.std()) - 1.0) < 1e-9


def test_ordering_preserved():
    adv = group_relative_advantage([1.0, 5.0, 3.0])
    assert adv[1] > adv[2] > adv[0]


def test_single_element_returns_zero():
    assert group_relative_advantage([7.0]) == [0.0]


def test_empty_returns_empty():
    assert group_relative_advantage([]) == []


def test_zero_variance_returns_zeros():
    assert group_relative_advantage([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]


def test_inf_reward_does_not_poison_group():
    adv = group_relative_advantage([1.0, 2.0, float("inf")])
    assert adv == [0.0, 0.0, 0.0]
    assert np.all(np.isfinite(adv))


def test_negative_inf_reward_does_not_poison_group():
    adv = group_relative_advantage([float("-inf"), 2.0, 3.0])
    assert adv == [0.0, 0.0, 0.0]
    assert np.all(np.isfinite(adv))


def test_nan_reward_does_not_poison_group():
    adv = group_relative_advantage([1.0, float("nan"), 3.0, 4.0])
    assert adv == [0.0, 0.0, 0.0, 0.0]
    assert not np.any(np.isnan(adv))
