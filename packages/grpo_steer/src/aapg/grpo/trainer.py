"""GRPO trainer loop (placeholder).

Sketches the online loop: sample a group of rollouts from the RFD3 policy, score them, compute
group-relative advantages, and apply the policy update with a KL penalty. Reward is resolved by
name from config via ``aapg.core.scoring``.
"""

from __future__ import annotations

from typing import Any


class GRPOTrainer:
    """Online GRPO trainer over an RFD3 sampling policy (placeholder)."""

    def __init__(self, policy: Any = None, reward: Any = None, **cfg: Any) -> None:
        self.policy = policy
        self.reward = reward
        self.cfg = cfg

    def step(self) -> dict[str, float]:
        """Run one GRPO update step and return metrics (placeholder)."""
        raise NotImplementedError

    def train(self) -> None:
        """Run the full training loop (placeholder)."""
        raise NotImplementedError
