"""aapg.grpo — module 1: GRPO steering of RFDiffusion3 (placeholder).

Online Group-Relative Policy Optimization that steers RFD3 sampling. Rollouts are generated in
groups, structures are scored (via ``aapg.core.scoring`` registries / ``aapg.toolkit``), and the
sampler policy is updated from group-relative advantages with a KL anchor to the base model.
"""

__version__ = "0.0.0"
