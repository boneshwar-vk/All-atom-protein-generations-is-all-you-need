"""Entry point for the GRPO selection pipeline (``aapg-grpo``).

Uses Hydra when it is installed (composing from the repo's ``configs/``), and otherwise falls
back to a dependency-free argparse runner that still executes the mock pipeline -- so the
``aapg-grpo`` console script works in CI without Hydra. Importing this module imports
``aapg.toolkit`` for its scorer-registration side effects, and ``aapg.grpo.reward`` to register
the gated reward.
"""

from __future__ import annotations

import sys
from typing import Any

import aapg.grpo.reward  # noqa: F401  (registers "ipsae_gated" into REWARDS)
import aapg.score_head  # noqa: F401  (registers "structure_head"[_mock] so the swap works)
import aapg.toolkit  # noqa: F401  (registers the concrete + mock scorers into SCORERS)
from aapg.grpo.trainer import GRPOTrainer

__all__ = ["main", "default_cfg"]


def default_cfg() -> dict[str, Any]:
    """A self-contained mock-runnable config mirroring configs/steering + configs/reward."""
    return {
        "mode": "selection",
        "mock": True,
        "n_backbones": 4,
        "seqs_per_backbone": 2,
        "n_rounds": 1,
        "top_k": 100,
        "group_size": 64,
        "seed": 0,
        "cascade": ["structure_auto", "rosetta_interface"],
        "scorer_cfg": {
            "structure_auto": {"boltz_weights_dir": None, "protenix_weights_dir": None},
        },
        "sampler": {},
        "designer": {"temperature": 0.1},
        "target": {
            "structure_path": None,
            "chains": ["B"],
            "binder_chains": ["A"],
            "hotspots": [],
            "binder_length": 80,
        },
        "reward": {
            "name": "ipsae_gated",
            "primary": "ipsae",
            "aggregate": "max",
            "top_k": 1,
            "penalty_weight": 1.0,
            "hard": {
                "has_clash": {"max": 0},
                "ipsae": {"min": 0.6},
                "ipae_mean": {"max": 10},
                "complex_plddt": {"min": 80},
                "scrmsd": {"max": 2.0},
            },
            "soft": {
                "iptm": {"min": 0.6, "scale": 0.3},
                "ipae_min": {"max": 5, "scale": 5},
                "interface_dg": {"max": 0, "scale": 15},
                "shape_complementarity": {"min": 0.55, "scale": 0.2},
                "contact_molecular_surface": {"min": 400, "scale": 400},
                "sap_normalized": {"max": 50, "scale": 50},
                "isoelectric_point": {"range": [6.0, 9.0], "scale": 3},
            },
        },
        "output_path": None,
    }


def _coerce(value: str) -> Any:
    """Coerce a CLI string to bool/int/float/None/list, else leave it as a string.

    A comma-separated value becomes a list (so ``cascade=structure_auto,rosetta_interface`` sets
    a list, not a string iterated character-by-character).
    """
    if "," in value:
        return [_coerce(part) for part in value.split(",")]
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for a dotted key, stripping a leading ``steering.``."""
    key = dotted_key
    if key.startswith("steering."):
        key = key[len("steering.") :]
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _run_fallback(argv: list[str]) -> int:
    """Argparse fallback: apply ``key=value`` overrides to the default cfg and run."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="aapg-grpo",
        description="Run the GRPO selection pipeline (Hydra-free fallback).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted key=value overrides, e.g. steering.mock=true n_backbones=8",
    )
    args = parser.parse_args(argv)

    cfg = default_cfg()
    for override in args.overrides:
        if "=" not in override:
            parser.error(f"override {override!r} is not of the form key=value")
        key, raw = override.split("=", 1)
        _set_nested(cfg, key, _coerce(raw))

    trainer = GRPOTrainer(cfg)
    table = trainer.run()
    n_rows = table.height if hasattr(table, "height") else len(table)
    print(f"[aapg-grpo] selection complete: {n_rows} candidate rows (mode={trainer.mode}).")
    if cfg.get("output_path"):
        print(f"[aapg-grpo] wrote ranked table to {cfg['output_path']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline. With Hydra installed and no ``argv``, use the Hydra app; else argparse.

    Returns a process exit code (``0`` on success) so it can be used directly as a console
    entry point and called in tests.
    """
    if argv is None:
        try:
            import hydra  # noqa: F401
        except ImportError:
            return _run_fallback(sys.argv[1:])
        return _run_hydra()
    return _run_fallback(argv)


def _run_hydra() -> int:  # pragma: no cover - exercised only when hydra is installed
    """Hydra entry point composing from the repo ``configs/`` tree."""
    from pathlib import Path

    import hydra
    from omegaconf import DictConfig, OmegaConf

    # packages/grpo_steer/src/aapg/grpo/cli.py -> repo root is five parents up.
    config_dir = str(Path(__file__).resolve().parents[5] / "configs")

    @hydra.main(version_base=None, config_path=config_dir, config_name="config")
    def _app(cfg: DictConfig) -> None:
        container = OmegaConf.to_container(cfg, resolve=True)
        container = container if isinstance(container, dict) else {}
        steering = container.get("steering") or {}
        reward = container.get("reward") or {}
        trainer_cfg = dict(steering) if isinstance(steering, dict) else {}
        if reward:
            trainer_cfg.setdefault("reward", reward)
        # Thread the top-level sampler/model groups into the sampler cfg so real-run settings
        # (e.g. the RFD3 checkpoint) are not silently dropped (they live outside `steering`).
        sampler_cfg = dict(trainer_cfg.get("sampler") or {})
        top_sampler = container.get("sampler") or {}
        if isinstance(top_sampler, dict):
            sampler_cfg.update(top_sampler)
        model_group = container.get("model") or {}
        if isinstance(model_group, dict) and model_group.get("checkpoint"):
            sampler_cfg.setdefault("checkpoint", model_group["checkpoint"])
        trainer_cfg["sampler"] = sampler_cfg

        # Thread the dedicated `scorer` group so it is not dead config: its `cascade` /
        # `scorer_cfg` (configs/scorer/*.yaml) take precedence over the copies in `steering`,
        # letting `scorer=boltz2` on the CLI actually pick the backend.
        scorer_group = container.get("scorer") or {}
        if isinstance(scorer_group, dict):
            if scorer_group.get("cascade"):
                trainer_cfg["cascade"] = scorer_group["cascade"]
            if scorer_group.get("scorer_cfg"):
                merged = dict(trainer_cfg.get("scorer_cfg") or {})
                merged.update(scorer_group["scorer_cfg"])
                trainer_cfg["scorer_cfg"] = merged

        # Thread the `designer` group (ProteinMPNN kwargs) so it reaches the designer adapter.
        designer_group = container.get("designer") or {}
        if isinstance(designer_group, dict) and designer_group:
            designer_cfg = dict(trainer_cfg.get("designer") or {})
            designer_cfg.update(designer_group)
            trainer_cfg["designer"] = designer_cfg

        GRPOTrainer(trainer_cfg).run()

    _app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
