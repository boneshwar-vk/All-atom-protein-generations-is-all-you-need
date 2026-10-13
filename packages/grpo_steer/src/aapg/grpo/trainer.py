"""GRPO driver: resolves components from config and runs the selection pipeline.

:class:`GRPOTrainer` has two modes. ``selection`` (v1, implemented) is best-of-N / hill-climb
selection -- no sampler weights are updated; it builds the sampler, designer, scorer cascade
and reward from config (scorers and reward resolved by name from the shared registries) and
delegates to :class:`aapg.grpo.selection.SelectionPipeline`. ``rl`` (not implemented) is the
future policy-gradient GRPO trainer; it raises :class:`NotImplementedError` pointing at the
group-relative advantage math in :mod:`aapg.grpo.advantage` and the per-step log-prob recording
hook documented in the rfdiffusion3 adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aapg.core.scoring import REWARDS, SCORERS, Reward, Scorer

if TYPE_CHECKING:
    from rfdiffusion3.adapter import Target

__all__ = ["GRPOTrainer"]

# Reward-config keys that are consumed by the pipeline (aggregation) rather than the reward
# constructor, and so must not be forwarded to REWARDS.create.
_REWARD_NON_CTOR_KEYS = frozenset({"name", "aggregate", "top_k"})


class GRPOTrainer:
    """Resolve GRPO components from config and run selection (or defer to the RL stub)."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg: dict[str, Any] = dict(cfg or {})
        self.mode: str = str(self.cfg.get("mode", "selection"))

    # -- component resolution ------------------------------------------------------------
    def _resolve_scorer_name(self, name: str, mock: bool) -> str:
        """Map a scorer name to its mock variant when ``mock`` and one is registered."""
        if mock:
            mock_name = f"{name}_mock"
            if mock_name in SCORERS:
                return mock_name
        return name

    def _build_scorers(self, mock: bool) -> list[Scorer]:
        """Build the cost-ordered scorer cascade from ``cfg['cascade']`` / ``cfg['scorers']``."""
        names = self.cfg.get("cascade") or self.cfg.get("scorers") or []
        scorer_cfgs: dict[str, Any] = self.cfg.get("scorer_cfg", {})
        scorers: list[Scorer] = []
        for name in names:
            resolved = self._resolve_scorer_name(str(name), mock)
            scorers.append(SCORERS.create(resolved, **scorer_cfgs.get(str(name), {})))
        return scorers

    def _build_reward(self) -> Reward:
        """Build the reward from ``cfg['reward']`` via the REWARDS registry."""
        reward_cfg: dict[str, Any] = dict(self.cfg.get("reward", {}))
        name = str(reward_cfg.get("name", "ipsae_gated"))
        ctor_kwargs = {k: v for k, v in reward_cfg.items() if k not in _REWARD_NON_CTOR_KEYS}
        return REWARDS.create(name, **ctor_kwargs)

    def _build_target(self) -> Target:
        """Build the :class:`Target` from ``cfg['target']`` (lazy adapter import)."""
        from rfdiffusion3.adapter import Target

        t: dict[str, Any] = dict(self.cfg.get("target", {}))
        return Target(
            structure_path=t.get("structure_path"),
            chains=tuple(t.get("chains", ("B",))),
            hotspots=tuple(t.get("hotspots", ())),
            binder_length=int(t.get("binder_length", 80)),
            metadata=dict(t.get("metadata", {})),
        )

    def _pipeline_cfg(self) -> dict[str, Any]:
        """Assemble the SelectionPipeline cfg from the trainer cfg (+ reward aggregation keys)."""
        reward_cfg: dict[str, Any] = dict(self.cfg.get("reward", {}))
        target_cfg: dict[str, Any] = dict(self.cfg.get("target", {}))
        return {
            "n_backbones": self.cfg.get("n_backbones", 4),
            "seqs_per_backbone": self.cfg.get("seqs_per_backbone", 8),
            "n_rounds": self.cfg.get("n_rounds", 1),
            "top_k": self.cfg.get("top_k", 1),
            "group_size": self.cfg.get("group_size", self.cfg.get("n_backbones", 4)),
            "seed": self.cfg.get("seed", 0),
            "binder_chains": tuple(target_cfg.get("binder_chains", ("A",))),
            "aggregate": reward_cfg.get("aggregate", "max"),
            "aggregate_top_k": reward_cfg.get("top_k", 1),
            "output_path": self.cfg.get("output_path"),
            "interim_gates": self.cfg.get("interim_gates"),
        }

    # -- drivers -------------------------------------------------------------------------
    def run(self, target: Target | None = None) -> Any:
        """Run the configured mode. ``selection`` returns the ranked table; ``rl`` raises."""
        if self.mode == "rl":
            raise NotImplementedError(
                "GRPOTrainer mode 'rl' (policy-gradient GRPO) is not implemented in v1. It will "
                "reuse group_relative_advantage from aapg.grpo.advantage and the per-step "
                "log-prob recording hook documented in rfdiffusion3.adapter (the sampler's "
                "random realignment must be disabled during policy-gradient training)."
            )
        if self.mode != "selection":
            raise ValueError(f"unknown GRPOTrainer mode {self.mode!r}; expected 'selection'/'rl'")

        from aapg.grpo.selection import SelectionPipeline

        mock = bool(self.cfg.get("mock", False))
        sampler = self._build_sampler(mock)
        designer = self._build_designer(mock)
        scorers = self._build_scorers(mock)
        reward = self._build_reward()
        pipeline = SelectionPipeline(sampler, designer, scorers, reward, self._pipeline_cfg())
        return pipeline.run(target if target is not None else self._build_target())

    def _build_sampler(self, mock: bool) -> Any:
        from rfdiffusion3.adapter import get_sampler

        return get_sampler(mock=mock, **dict(self.cfg.get("sampler", {})))

    def _build_designer(self, mock: bool) -> Any:
        """Build the sequence designer from ``cfg['designer']`` (name-switchable backend).

        ``designer.name`` picks the backend (default ``proteinmpnn``; ``protonmpnn`` is the
        placeholder alternate); the remaining kwargs are forwarded to the backend constructor. A
        config with no ``designer.name`` still uses ProteinMPNN (backward compatible).
        """
        from aapg.grpo.designers import get_designer

        designer_cfg = dict(self.cfg.get("designer", {}))
        name = str(designer_cfg.pop("name", "proteinmpnn"))
        return get_designer(name, mock=mock, **designer_cfg)

    # -- backward-compatible surface -----------------------------------------------------
    def train(self) -> Any:
        """Alias for :meth:`run` over the config-built target (backward compatible)."""
        return self.run()

    def step(self) -> dict[str, float]:
        """A single policy-update step. Undefined for pure selection; the RL step is future work."""
        raise NotImplementedError(
            "GRPOTrainer.step applies to the future 'rl' mode's policy update; v1 'selection' "
            "has no weight-update step -- call run()/train() to execute the selection loop."
        )
