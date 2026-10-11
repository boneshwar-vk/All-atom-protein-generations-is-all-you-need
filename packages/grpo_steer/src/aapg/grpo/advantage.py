"""Group-relative advantage: the core GRPO normalization, reused by the future RL trainer.

GRPO normalizes each rollout's reward *within its group* rather than against a learned value
baseline: the advantage of rollout ``i`` is ``(r_i - mean(r)) / std(r)``. In v1 this ranks the
best-of-N candidates within a group; in the later policy-gradient trainer the same quantity
multiplies the per-step log-probs to form the surrogate loss (see the FUTURE-RL HOOK in the
rfdiffusion3 adapter). Keeping the math in one place keeps the two consistent.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["group_relative_advantage"]

# Floor on the group std to avoid dividing by ~0 when rewards are nearly constant.
_STD_EPSILON = 1e-8


def group_relative_advantage(rewards: Sequence[float]) -> list[float]:
    """Return the group-relative advantages of ``rewards``.

    Computes ``(r - mean(r)) / std(r)`` with a small epsilon floor on the std. Degenerate
    groups return all zeros: fewer than two rewards (no meaningful group) and zero-variance
    groups (every reward equal) both yield a flat baseline, so no candidate is preferred.

    Args:
        rewards: the per-rollout scalar rewards of one group.

    Returns:
        A list of advantages, one per input reward, in the same order.
    """
    arr = np.asarray(list(rewards), dtype=float)
    if arr.size < 2:
        return [0.0] * arr.size

    std = float(arr.std())
    if std <= _STD_EPSILON:
        return [0.0] * arr.size

    mean = float(arr.mean())
    return [float(a) for a in (arr - mean) / std]
