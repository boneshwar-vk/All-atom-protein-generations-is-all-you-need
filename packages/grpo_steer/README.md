# aapg-grpo (module 1)

Online **GRPO** (Group-Relative Policy Optimization) that steers RFDiffusion3 sampling toward
high-reward structures.

Pieces (placeholders): a policy wrapper over the RFD3 sampler, a group rollout sampler, the
advantage/KL computation, and `trainer.py` (the online loop). Reward is composed from
`aapg.core.scoring` scorers, resolved by name from config.

Config groups: [`configs/steering/`](../../configs/steering), [`configs/reward/`](../../configs/reward).

**Status:** placeholder stubs only.
