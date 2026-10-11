"""The GRPO selection loop: best-of-N / hill-climb over sampled backbones (module 1 v1).

This is the heart of v1. It is pure *selection* -- no sampler weights are updated (see the
FUTURE-RL HOOK in the rfdiffusion3 adapter and :mod:`aapg.grpo.advantage` for what the later
policy-gradient trainer will reuse). Each round:

  1. sample ``n_backbones`` backbones for the target;
  2. design ``seqs_per_backbone`` sequences on each backbone and build candidates;
  3. run a COST-ORDERED SCORING CASCADE with early exit -- cheap scorers first, interim hard
     gates applied after each stage, expensive scorers run ONLY on survivors;
  4. compute scRMSD (the 11th metric) per candidate;
  5. aggregate the per-sequence scores to one representative score per backbone;
  6. compute the reward per backbone and rank backbones by group-relative advantage;
  7. keep the top-k backbones (optionally reseeding the next round around them).

Every candidate (backbone x sequence) is recorded with all 11 metrics, its reward and its
group-relative advantage. The ranked table is returned (and optionally written to Parquet via
polars, or JSONL via the stdlib when polars is absent).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aapg.core.scoring import Reward, Score, Scorer, combine_scores
from aapg.grpo.advantage import group_relative_advantage
from aapg.grpo.candidate import build_candidate
from aapg.grpo.reward import aggregate_sequences
from aapg.grpo.scrmsd import self_consistency_rmsd

if TYPE_CHECKING:
    from rfdiffusion3.adapter import Backbone, Target

__all__ = ["SelectionPipeline", "METRIC_COLUMNS"]

# The 10 scorer metrics + scRMSD, in recorded-column order.
METRIC_COLUMNS = [
    "ipsae",
    "iptm",
    "ipae_min",
    "ipae_mean",
    "sap_normalized",
    "isoelectric_point",
    "interface_dg",
    "shape_complementarity",
    "contact_molecular_surface",
    "complex_plddt",
    "scrmsd",
]

# Non-metric columns that precede the metrics in the output table.
_LEAD_COLUMNS = ["name", "binder_sequence", "round", "reward", "advantage", "backbone_path"]


def _bound_satisfied(value: float, bound: dict[str, Any]) -> bool:
    """Whether ``value`` satisfies a min/max/range ``bound`` (interim-gate check)."""
    if "range" in bound:
        lo, hi = float(bound["range"][0]), float(bound["range"][1])
        return lo <= value <= hi
    if "min" in bound and value < float(bound["min"]):
        return False
    if "max" in bound and value > float(bound["max"]):
        return False
    return True


def _predicted_ca(candidate_meta: dict[str, Any], scores: list[Score], seed_key: str) -> Any:
    """Return the predicted-complex binder CA trace for scRMSD.

    In a real run the structure predictor emits binder CA coordinates; any scorer that provides
    them stores them under ``metadata["predicted_ca_coords"]`` and they are used directly. In
    the mock / dry-run path no predictor runs, so a deterministic small perturbation of the
    designed backbone CA trace stands in, giving a reproducible, plausibly-small scRMSD.
    """
    for score in scores:
        predicted = score.metadata.get("predicted_ca_coords")
        if predicted is not None:
            return np.asarray(predicted, dtype=float)

    designed = candidate_meta.get("backbone_ca_coords")
    if designed is None:
        return None
    designed = np.asarray(designed, dtype=float)
    digest = hashlib.sha256(seed_key.encode()).digest()
    rng = np.random.default_rng(seed=int.from_bytes(digest[:8], "big"))
    # ~sub-angstrom per-atom jitter -> a small, deterministic scRMSD after superposition.
    return designed + rng.normal(loc=0.0, scale=0.6, size=designed.shape)


class SelectionPipeline:
    """Best-of-N / hill-climb selection over sampled backbones (no weight updates)."""

    def __init__(
        self,
        sampler: Any,
        designer: Any,
        scorers: list[Scorer],
        reward: Reward,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.sampler = sampler
        self.designer = designer
        # The scorer list IS the cost-ordered cascade: cheapest stage first.
        self.scorers = list(scorers)
        self.reward = reward
        self.cfg: dict[str, Any] = dict(cfg or {})

        self.n_backbones = int(self.cfg.get("n_backbones", 4))
        self.seqs_per_backbone = int(self.cfg.get("seqs_per_backbone", 8))
        self.n_rounds = int(self.cfg.get("n_rounds", 1))
        self.top_k = int(self.cfg.get("top_k", 1))
        self.group_size = int(self.cfg.get("group_size", self.n_backbones))
        self.seed = int(self.cfg.get("seed", 0))
        self.binder_chains = tuple(self.cfg.get("binder_chains", ("A",)))
        self.aggregate = str(self.cfg.get("aggregate", "max"))
        self.aggregate_top_k = int(self.cfg.get("aggregate_top_k", self.cfg.get("top_k", 1)))
        # Interim gates for early exit; default to the reward's own hard gates when present.
        interim = self.cfg.get("interim_gates")
        if interim is None:
            interim = getattr(self.reward, "hard", {})
        self.interim_gates: dict[str, Any] = dict(interim or {})

    # -- scoring cascade -----------------------------------------------------------------
    def _score_candidate(self, candidate: Any) -> tuple[list[Score], bool]:
        """Run the cost-ordered cascade on ``candidate`` with interim-gate early exit.

        Returns the accumulated per-stage scores and whether the candidate survived every
        stage (``False`` means an interim hard gate rejected it and expensive stages were
        skipped).
        """
        collected: list[Score] = []
        survived = True
        for i, scorer in enumerate(self.scorers):
            collected.append(scorer.score(candidate))
            # After every stage except the last, drop candidates already failing a hard gate.
            if i < len(self.scorers) - 1 and self.interim_gates:
                values = combine_scores(collected).values
                for metric, bound in self.interim_gates.items():
                    if metric in values and not _bound_satisfied(float(values[metric]), bound):
                        survived = False
                        break
                if not survived:
                    break
        return collected, survived

    def _candidate_score(self, candidate: Any) -> Score:
        """Score one candidate through the cascade and attach scRMSD, returning one Score."""
        collected, _survived = self._score_candidate(candidate)
        merged = combine_scores(collected)
        predicted = _predicted_ca(
            candidate.metadata, collected, seed_key=f"scrmsd:{candidate.name}"
        )
        designed = candidate.metadata.get("backbone_ca_coords")
        if predicted is not None and designed is not None:
            merged.values["scrmsd"] = self_consistency_rmsd(
                np.asarray(designed, dtype=float), predicted
            )
        return merged

    # -- one round -----------------------------------------------------------------------
    def _run_round(self, target: Target, round_idx: int, seed: int) -> list[dict[str, Any]]:
        """Sample, design, score, reward and rank one round; return per-candidate rows."""
        backbones: list[Backbone] = self.sampler.sample_backbones(
            target, n=self.n_backbones, seed=seed
        )

        # Per backbone: design sequences, score each candidate, aggregate, reward.
        backbone_reward: list[float] = []
        backbone_rows: list[list[dict[str, Any]]] = []
        for b_idx, backbone in enumerate(backbones):
            sequences = self.designer.design(
                backbone,
                n_seqs=self.seqs_per_backbone,
                binder_chains=self.binder_chains,
                target_chains=target.chains,
                seed=seed,
            )
            seq_scores: list[Score] = []
            rows: list[dict[str, Any]] = []
            for s_idx, sequence in enumerate(sequences):
                name = f"r{round_idx}_b{b_idx:04d}_s{s_idx:02d}"
                candidate = build_candidate(
                    backbone, sequence, target, name=name, binder_chains=self.binder_chains
                )
                score = self._candidate_score(candidate)
                seq_scores.append(score)
                rows.append(
                    {
                        "name": name,
                        "binder_sequence": sequence,
                        "round": round_idx,
                        "backbone_path": backbone.path,
                        **{m: score.values.get(m) for m in METRIC_COLUMNS},
                    }
                )
            representative = aggregate_sequences(
                seq_scores,
                how=self.aggregate,
                primary=getattr(self.reward, "primary", "ipsae"),
                top_k=self.aggregate_top_k,
            )
            backbone_reward.append(self.reward([representative]))
            backbone_rows.append(rows)

        # Group-relative advantage over backbones, chunked into groups of group_size.
        advantages = self._grouped_advantage(backbone_reward)

        out_rows: list[dict[str, Any]] = []
        for b_idx, rows in enumerate(backbone_rows):
            for row in rows:
                row["reward"] = backbone_reward[b_idx]
                row["advantage"] = advantages[b_idx]
                out_rows.append(row)
        return out_rows

    def _grouped_advantage(self, rewards: list[float]) -> list[float]:
        """Group-relative advantage, computed within consecutive chunks of ``group_size``."""
        if self.group_size <= 0:
            return group_relative_advantage(rewards)
        out: list[float] = []
        for start in range(0, len(rewards), self.group_size):
            out.extend(group_relative_advantage(rewards[start : start + self.group_size]))
        return out

    # -- driver --------------------------------------------------------------------------
    def run(self, target: Target) -> Any:
        """Run all rounds and return the ranked table (polars DataFrame or list of dicts)."""
        all_rows: list[dict[str, Any]] = []
        seed = self.seed
        for round_idx in range(self.n_rounds):
            all_rows.extend(self._run_round(target, round_idx, seed))
            # Reseed the next round; a real hill-climb would reseed around the round's winners
            # (its top-k backbones). v1 advances the seed deterministically.
            seed += 1

        # Rank the whole table by advantage then reward, best first.
        all_rows.sort(
            key=lambda r: (r.get("advantage") or 0.0, r.get("reward") or 0.0), reverse=True
        )
        self._write(all_rows)
        return self._as_table(all_rows)

    # -- output --------------------------------------------------------------------------
    def _ordered(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reorder each row's keys to the canonical column order."""
        columns = _LEAD_COLUMNS + [m for m in METRIC_COLUMNS if m not in _LEAD_COLUMNS]
        return [{c: row.get(c) for c in columns} for row in rows]

    def _as_table(self, rows: list[dict[str, Any]]) -> Any:
        """Return a polars DataFrame if polars is importable, else the ordered list of dicts."""
        ordered = self._ordered(rows)
        try:
            import polars as pl
        except ImportError:
            return ordered
        return pl.DataFrame(ordered)

    def _write(self, rows: list[dict[str, Any]]) -> str | None:
        """Write the ranked table to ``cfg['output_path']`` (Parquet via polars, else JSONL)."""
        output_path = self.cfg.get("output_path")
        if not output_path:
            return None
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = self._ordered(rows)
        try:
            import polars as pl

            out = path if path.suffix else path.with_suffix(".parquet")
            pl.DataFrame(ordered).write_parquet(out)
            return str(out)
        except ImportError:
            out = path if path.suffix else path.with_suffix(".jsonl")
            with out.open("w") as fh:
                for row in ordered:
                    fh.write(json.dumps(row) + "\n")
            return str(out)
