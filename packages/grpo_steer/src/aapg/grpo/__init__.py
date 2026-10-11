"""aapg.grpo — module 1: GRPO steering of RFDiffusion3.

v1 is best-of-N / hill-climb **selection** (no sampler weight updates): backbones are sampled,
sequences designed, candidates scored through a cost-ordered cascade (via ``aapg.core.scoring``
registries / ``aapg.toolkit``), and the best are kept by group-relative advantage. The reward
hill-climbs toward Protenix v2 ipSAE with calibrated gates on the other metrics.

The ``rl`` mode (policy-gradient GRPO that updates the sampler from group-relative advantages
with a KL anchor to the base model) is future work; the advantage math and the per-step log-prob
recording hook are in place (see :mod:`aapg.grpo.advantage` and ``rfdiffusion3.adapter``).
"""

from aapg.grpo.advantage import group_relative_advantage
from aapg.grpo.candidate import build_candidate
from aapg.grpo.reward import IpSAEGatedReward, aggregate_sequences
from aapg.grpo.scrmsd import self_consistency_rmsd
from aapg.grpo.selection import METRIC_COLUMNS, SelectionPipeline
from aapg.grpo.trainer import GRPOTrainer

__version__ = "0.0.0"

__all__ = [
    "group_relative_advantage",
    "build_candidate",
    "IpSAEGatedReward",
    "aggregate_sequences",
    "self_consistency_rmsd",
    "METRIC_COLUMNS",
    "SelectionPipeline",
    "GRPOTrainer",
]
