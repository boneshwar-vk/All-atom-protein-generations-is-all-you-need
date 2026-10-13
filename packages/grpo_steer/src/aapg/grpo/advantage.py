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

    Non-finite guard: if any reward is ``nan``/``inf`` (or the group mean/std come out
    non-finite), the whole group returns all zeros rather than propagating NaN/Inf into every
    advantage. A single poisoned reward would otherwise make ``mean``/``std`` non-finite and turn
    the entire group's advantages into NaN, silently corrupting ranking and (later) the RL
    surrogate loss. Rejecting the group to a flat baseline is the minimal safe behavior: no
    candidate is preferred, and the caller's gates/filters still drop the bad candidate upstream.
    This is defensive hardening -- the scorers clamp their outputs, so non-finite rewards are not
    expected to reach here in normal operation.

    Args:
        rewards: the per-rollout scalar rewards of one group.

    Returns:
        A list of advantages, one per input reward, in the same order.
    """
    arr = np.asarray(list(rewards), dtype=float)
    if arr.size < 2:
        return [0.0] * arr.size

    # Any non-finite reward poisons mean/std -> reject the group to a flat baseline.
    if not np.all(np.isfinite(arr)):
        return [0.0] * arr.size

    std = float(arr.std())
    mean = float(arr.mean())
    if not np.isfinite(std) or std <= _STD_EPSILON or not np.isfinite(mean):
        return [0.0] * arr.size

    return [float(a) for a in (arr - mean) / std]
