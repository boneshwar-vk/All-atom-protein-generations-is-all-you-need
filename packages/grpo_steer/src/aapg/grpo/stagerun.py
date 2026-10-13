"""Stage-wise, memory-safe, resumable runner for the GRPO binder pipeline (``aapg-grpo-stage``).

The online :class:`aapg.grpo.selection.SelectionPipeline` holds RFD3, ProteinMPNN, the structure
predictor and PyRosetta co-resident -- fine on an A100, but a 24 GB MacBook cannot keep them all
in memory at once. This module splits the pipeline into **one subcommand per stage**, each meant
to be run as its OWN process so the previous stage's model memory is fully released before the
next one loads:

    rfd3  ->  mpnn  ->  fold  ->  rosetta  ->  reward

Every stage persists its output to a manifest under the shared work dir (``--run DIR``) and is
**resumable**: re-running a stage skips the ids already present in its manifest, so an
interrupted run (or an out-of-memory bail-out) continues where it left off. The ``fold`` stage
additionally guards against OOM: before folding each complex it checks free memory and, if it
cannot stay above the floor, checkpoints (its progress is already on disk) and exits ``75``
(``EX_TEMPFAIL``) with a clear resume message instead of letting the OS kill the process.

Nothing here reimplements a model: every stage reuses the repository's existing adapters, the
shared scorer registry (:data:`aapg.core.scoring.SCORERS`) and the gated reward
(:class:`aapg.grpo.reward.IpSAEGatedReward`). Heavy backends (torch, boltz, PyRosetta) are
imported lazily inside the scorers, so the whole CLI -- and the full mock pipeline used by CI --
runs CPU-only with no GPU, weights or heavy dependencies.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import aapg.grpo.reward  # noqa: F401  (registers "ipsae_gated" into REWARDS)
import aapg.toolkit  # noqa: F401  (registers the concrete + mock scorers into SCORERS)
from aapg.core.scoring import REWARDS, SCORERS, Candidate, Score
from aapg.grpo.candidate import build_candidate
from aapg.grpo.scrmsd import self_consistency_rmsd

if TYPE_CHECKING:
    from rfdiffusion3.adapter import Backbone, Target

__all__ = ["main"]

# Process exit code used when the fold stage bails out under memory pressure. 75 is the classic
# EX_TEMPFAIL ("try again later"): progress is already checkpointed, so re-running resumes.
EXIT_MEMORY_BAILOUT = 75

# The structure-scorer metrics recorded per candidate by the ``fold`` stage, in order.
_STRUCTURE_METRICS = ["ipsae", "iptm", "ipae_min", "ipae_mean", "complex_plddt", "has_clash"]
# The Rosetta interface metrics recorded per candidate by the ``rosetta`` stage, in order.
_ROSETTA_METRICS = [
    "interface_dg",
    "shape_complementarity",
    "contact_molecular_surface",
    "sap_normalized",
    "isoelectric_point",
]

# Reward-config keys consumed by the selection aggregation rather than the reward constructor;
# they must not be forwarded to ``REWARDS.create`` (mirrors aapg.grpo.trainer).
_REWARD_NON_CTOR_KEYS = frozenset({"name", "aggregate", "top_k"})

# Embedded fallback used only if the reward YAML cannot be read (PyYAML absent / file missing),
# so the CPU/mock path never hard-fails on a missing optional dependency. Mirrors
# configs/reward/ipsae_gated.yaml.
_DEFAULT_REWARD_CFG: dict[str, Any] = {
    "name": "ipsae_gated",
    "primary": "ipsae",
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
}


# --------------------------------------------------------------------------------------------
# small on-disk manifest helpers (append + dedupe => resumable)
# --------------------------------------------------------------------------------------------
def _run_dir(args: argparse.Namespace) -> Path:
    """Return the run (work) directory, creating it if needed."""
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    return run


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-Lines manifest into a list of dicts (empty if the file is absent)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _done_ids(path: Path, key: str) -> set[str]:
    """Return the set of values under ``key`` already recorded in the JSON-Lines ``path``."""
    return {str(row[key]) for row in _read_jsonl(path) if key in row}


def _append_row(fh: Any, row: dict[str, Any]) -> None:
    """Append one row to an open JSON-Lines file handle and flush (crash-safe checkpointing)."""
    fh.write(json.dumps(row) + "\n")
    fh.flush()


# --------------------------------------------------------------------------------------------
# target + reward resolution
# --------------------------------------------------------------------------------------------
def _resolve_target(args: argparse.Namespace) -> tuple[Target, tuple[str, ...]]:
    """Build the :class:`Target` and default binder chains from ``--target-json`` or flags.

    A ``--target-json`` file (keys: ``structure_path``, ``chains``, ``binder_chains``,
    ``hotspots``, ``binder_length``) supplies the base; any individual target flag that is set
    overrides the corresponding field. Returns ``(target, binder_chains)`` -- the binder chains
    are a *default* only; once a backbone is sampled, the chain ids the sampler recorded in its
    manifest row are the single source of truth downstream.
    """
    from rfdiffusion3.adapter import Target

    spec: dict[str, Any] = {}
    if args.target_json:
        loaded = json.loads(Path(args.target_json).read_text())
        if isinstance(loaded, dict):
            spec.update(loaded)

    def _csv(value: str | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        return tuple(part for part in (p.strip() for p in value.split(",")) if part)

    structure_path = (
        args.structure_path if args.structure_path is not None else spec.get("structure_path")
    )
    chains = _csv(args.chains) or tuple(spec.get("chains", ("B",)))
    binder_chains = _csv(args.binder_chains) or tuple(spec.get("binder_chains", ("A",)))
    hotspots = _csv(args.hotspots) or tuple(spec.get("hotspots", ()))
    binder_length = (
        args.binder_length if args.binder_length is not None else int(spec.get("binder_length", 80))
    )

    target = Target(
        structure_path=structure_path,
        chains=tuple(chains),
        hotspots=tuple(hotspots),
        binder_length=int(binder_length),
    )
    return target, tuple(binder_chains)


def _repo_root() -> Path:
    """Repo root for locating default configs (…/packages/grpo_steer/src/aapg/grpo/stagerun.py)."""
    return Path(__file__).resolve().parents[5]


def _load_reward_cfg(path: str | None) -> dict[str, Any]:
    """Load the reward config from YAML, returning only the reward *constructor* kwargs.

    Falls back to :data:`_DEFAULT_REWARD_CFG` when the YAML file is missing or PyYAML is not
    installed, so the reward stage works CPU-only with the minimal install.
    """
    cfg_path = Path(path) if path else _repo_root() / "configs" / "reward" / "ipsae_gated.yaml"
    raw: Any = None
    try:
        import yaml

        raw = yaml.safe_load(cfg_path.read_text())
    except (ImportError, OSError):
        raw = None
    if not isinstance(raw, dict):
        raw = _DEFAULT_REWARD_CFG
    return {k: v for k, v in raw.items() if k not in _REWARD_NON_CTOR_KEYS}


# --------------------------------------------------------------------------------------------
# memory guard (fold stage)
# --------------------------------------------------------------------------------------------
def available_memory_gb() -> float | None:
    """Best-effort available system memory in GB, or ``None`` if it cannot be determined.

    Prefers ``psutil`` when importable; otherwise parses ``vm_stat`` on macOS and
    ``/proc/meminfo`` on Linux. Returning ``None`` means "cannot tell" (the caller treats that as
    enough memory rather than bailing out spuriously).
    """
    try:
        import psutil

        return float(psutil.virtual_memory().available) / 1e9
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - a flaky psutil must not crash the stage; fall through
        return None

    if sys.platform == "darwin":
        return _vm_stat_available_gb()
    if sys.platform.startswith("linux"):
        return _meminfo_available_gb()
    return None


def _vm_stat_available_gb() -> float | None:
    """Parse macOS ``vm_stat`` for (free + inactive + speculative) pages -> GB."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page_size = 4096
    pages = 0
    try:
        for line in out.splitlines():
            if "page size of" in line:
                page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
            elif line.startswith(("Pages free", "Pages inactive", "Pages speculative")):
                pages += int(line.rsplit(":", 1)[1].strip().rstrip("."))
    except (ValueError, IndexError):
        return None
    return pages * page_size / 1e9


def _meminfo_available_gb() -> float | None:
    """Parse Linux ``/proc/meminfo`` ``MemAvailable`` (kB) -> GB."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        return None
    return None


def enough_memory(min_free_gb: float) -> bool:
    """Whether at least ``min_free_gb`` of memory is available. Undeterminable -> ``True``."""
    available = available_memory_gb()
    if available is None:
        return True
    return available >= min_free_gb


def _release_memory() -> None:
    """Collect garbage and empty torch's MPS/CUDA caches -- but only if torch is already loaded."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if getattr(torch, "backends", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - cache clearing is best-effort
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - cache clearing is best-effort
        pass


# --------------------------------------------------------------------------------------------
# stage 1: rfd3
# --------------------------------------------------------------------------------------------
def cmd_rfd3(args: argparse.Namespace) -> int:
    """Sample backbones and record each binder CA trace + a manifest row."""
    from rfdiffusion3.adapter import get_sampler

    run = _run_dir(args)
    bb_dir = run / "backbones"
    bb_dir.mkdir(parents=True, exist_ok=True)
    manifest = run / "backbones.json"
    target, default_binder_chains = _resolve_target(args)

    existing: dict[str, dict[str, Any]] = {}
    if manifest.exists():
        for row in json.loads(manifest.read_text()):
            existing[str(row["id"])] = row
    if len(existing) >= args.n and not args.force:
        print(f"[rfd3] {len(existing)} backbones already present (>= {args.n}); skipping.")
        return 0
    if args.force:
        existing = {}

    sampler_cfg: dict[str, Any] = {}
    if args.checkpoint:
        sampler_cfg["checkpoint"] = args.checkpoint
    if args.rfd3_command:
        sampler_cfg["command"] = args.rfd3_command.split()
    sampler = get_sampler(mock=args.mock, **sampler_cfg)

    backbones: list[Backbone] = sampler.sample_backbones(target, n=args.n, seed=args.seed)
    rows: dict[str, dict[str, Any]] = dict(existing)
    for i, backbone in enumerate(backbones):
        bb_id = f"bb{i:04d}"
        if bb_id in existing and (bb_dir / f"{bb_id}.npy").exists():
            continue  # already sampled and saved; resume without re-writing
        recorded = backbone.metadata.get("binder_chains")
        binder_chains = tuple(recorded) if recorded else default_binder_chains
        ca_coords = backbone.metadata.get("ca_coords")
        if ca_coords is None:
            raise RuntimeError(f"backbone {bb_id} has no CA coordinates to persist for scRMSD")
        np.save(bb_dir / f"{bb_id}.npy", np.asarray(ca_coords, dtype=float))
        rows[bb_id] = {
            "id": bb_id,
            "path": backbone.path,
            "length": int(backbone.length),
            "binder_chains": list(binder_chains),
        }

    ordered = [rows[k] for k in sorted(rows)]
    manifest.write_text(json.dumps(ordered, indent=2))
    print(f"[rfd3] wrote {len(ordered)} backbones to {manifest}")
    return 0


# --------------------------------------------------------------------------------------------
# stage 2: mpnn
# --------------------------------------------------------------------------------------------
def cmd_mpnn(args: argparse.Namespace) -> int:
    """Design ``--seqs-per`` sequences on each backbone not already in the sequences manifest."""
    from aapg.grpo.designers import get_designer
    from rfdiffusion3.adapter import Backbone

    run = _run_dir(args)
    backbones = json.loads((run / "backbones.json").read_text())
    target, _default_binder_chains = _resolve_target(args)
    seq_path = run / "sequences.jsonl"
    done = _done_ids(seq_path, "id")

    designer_cfg: dict[str, Any] = {}
    if args.weights_set:
        designer_cfg["weights_set"] = args.weights_set
    if args.checkpoint:
        designer_cfg["checkpoint"] = args.checkpoint
    designer = get_designer(args.designer, mock=args.mock, **designer_cfg)

    written = 0
    with seq_path.open("a") as fh:
        for bb in backbones:
            bb_id = str(bb["id"])
            if bb_id in done:
                continue  # this backbone's sequences are already recorded; resume
            binder_chains = tuple(bb["binder_chains"])
            backbone = Backbone(
                path=bb["path"],
                length=int(bb["length"]),
                metadata={"binder_chains": binder_chains},
            )
            sequences = designer.design(
                backbone,
                n_seqs=args.seqs_per,
                binder_chains=binder_chains,
                target_chains=target.chains,
                seed=args.seed,
            )
            for seq_idx, seq in enumerate(sequences):
                _append_row(
                    fh,
                    {
                        "id": bb_id,
                        "seq_idx": seq_idx,
                        "cand": f"{bb_id}_seq{seq_idx:02d}",
                        "seq": seq,
                        "binder_chains": list(binder_chains),
                    },
                )
                written += 1
    print(f"[mpnn] wrote {written} new sequences to {seq_path}")
    return 0


# --------------------------------------------------------------------------------------------
# stage 3: fold + score
# --------------------------------------------------------------------------------------------
def _build_structure_scorer(args: argparse.Namespace, structures_dir: Path) -> Any:
    """Construct the structure scorer with ``work_dir=structures_dir`` so its CIF persists.

    The CIF must outlive this process (the ``rosetta`` stage reads it), so an ephemeral temp dir
    is deliberately NOT used. Mock backends ignore the extra kwargs.
    """
    backend = str(args.backend)
    if args.mock:
        mock_name = f"{backend}_mock"
        name = mock_name if mock_name in SCORERS else backend
        return SCORERS.create(name)

    work_dir = str(structures_dir)
    if backend == "structure_auto":
        # structure_auto delegates to boltz2 (or protenix_v2 when licensed weights are present);
        # thread the persist dir into BOTH sub-configs so whichever backend runs keeps its CIF.
        return SCORERS.create(
            "structure_auto",
            boltz_weights_dir=args.weights_dir,
            protenix_weights_dir=args.protenix_weights_dir,
            boltz_cfg={
                "work_dir": work_dir,
                "recycling_steps": args.recycling_steps,
                "diffusion_samples": args.diffusion_samples,
            },
            protenix_cfg={"work_dir": work_dir},
        )
    if backend == "boltz2":
        return SCORERS.create(
            "boltz2",
            work_dir=work_dir,
            weights_dir=args.weights_dir,
            recycling_steps=args.recycling_steps,
            diffusion_samples=args.diffusion_samples,
        )
    if backend == "protenix_v2":
        return SCORERS.create(
            "protenix_v2",
            work_dir=work_dir,
            weights_dir=args.protenix_weights_dir or args.weights_dir,
        )
    return SCORERS.create(backend, work_dir=work_dir, weights_dir=args.weights_dir)


def _bb_dir_npy(run: Path, bb_id: str) -> Path:
    """Path to a backbone's saved binder CA trace."""
    return run / "backbones" / f"{bb_id}.npy"


def _scrmsd(designed_ca: np.ndarray, predicted: Any) -> float | None:
    """scRMSD of the designed vs predicted binder CA trace; ``inf`` on shape error, ``None`` if
    no predicted coordinates were emitted (e.g. a mock scorer)."""
    if predicted is None:
        return None
    try:
        return self_consistency_rmsd(designed_ca, np.asarray(predicted, dtype=float))
    except Exception:  # noqa: BLE001 - a shape mismatch must fail closed, not crash the stage
        return float("inf")


def cmd_fold(args: argparse.Namespace) -> int:
    """Fold + score each candidate one at a time, persisting the CIF and a metrics row."""
    from rfdiffusion3.adapter import Backbone

    run = _run_dir(args)
    structures_dir = run / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    target, _default_binder_chains = _resolve_target(args)

    bb_by_id = {str(b["id"]): b for b in json.loads((run / "backbones.json").read_text())}
    sequences = _read_jsonl(run / "sequences.jsonl")
    struct_path = run / "structure.jsonl"
    done = _done_ids(struct_path, "cand")

    scorer = _build_structure_scorer(args, structures_dir)
    folded = 0
    with struct_path.open("a") as fh:
        for seq_row in sequences:
            cand = str(seq_row["cand"])
            if cand in done:
                continue
            bb_id = str(seq_row["id"])
            bb = bb_by_id[bb_id]
            binder_chains = tuple(seq_row["binder_chains"])

            # MEMORY GUARD: never start a fold we cannot afford. Progress is already checkpointed,
            # so if we cannot free enough memory we exit 75 and the next invocation resumes.
            if not enough_memory(args.min_free_gb):
                _release_memory()
                if not enough_memory(args.min_free_gb):
                    print(
                        f"[fold] low memory (< {args.min_free_gb} GB free) before folding {cand}; "
                        f"checkpointed {folded} complexes this run. Re-run `fold` to resume.",
                        file=sys.stderr,
                    )
                    return EXIT_MEMORY_BAILOUT

            designed_ca = np.load(_bb_dir_npy(run, bb_id))
            backbone = Backbone(
                path=bb["path"],
                length=int(bb["length"]),
                metadata={"binder_chains": binder_chains, "ca_coords": designed_ca},
            )
            candidate = build_candidate(
                backbone, seq_row["seq"], target, name=cand, binder_chains=binder_chains
            )
            # Thread the persist dir via metadata too, so every structure backend keeps its CIF.
            candidate = replace(
                candidate, metadata={**candidate.metadata, "work_dir": str(structures_dir)}
            )
            score = scorer.score(candidate)
            values = dict(score.values)
            cif = score.metadata.get("cif_path")
            scrmsd = _scrmsd(designed_ca, score.metadata.get("predicted_ca_coords"))

            row: dict[str, Any] = {"cand": cand, "id": bb_id, "seq": seq_row["seq"], "cif": cif}
            for metric in _STRUCTURE_METRICS:
                row[metric] = values.get(metric)
            row["scrmsd"] = scrmsd
            _append_row(fh, row)
            folded += 1
            print(f"[fold] scored {cand}")
    print(f"[fold] scored {folded} new complexes into {struct_path}")
    return 0


# --------------------------------------------------------------------------------------------
# stage 4: rosetta
# --------------------------------------------------------------------------------------------
def cmd_rosetta(args: argparse.Namespace) -> int:
    """Score the PyRosetta interface on the predicted complex (or the backbone, in mock)."""
    run = _run_dir(args)
    target, _default_binder_chains = _resolve_target(args)
    bb_by_id = {str(b["id"]): b for b in json.loads((run / "backbones.json").read_text())}
    structure_rows = _read_jsonl(run / "structure.jsonl")
    out_path = run / "rosetta.jsonl"
    done = _done_ids(out_path, "cand")

    name = "rosetta_interface_mock" if args.mock else "rosetta_interface"
    scorer = SCORERS.create(name)

    scored = 0
    with out_path.open("a") as fh:
        for row in structure_rows:
            cand = str(row["cand"])
            if cand in done:
                continue
            bb = bb_by_id[str(row["id"])]
            binder_chains = tuple(bb["binder_chains"])
            cif = row.get("cif")
            # Score the PREDICTED complex when a CIF exists; otherwise (mock) the mock scorer
            # ignores the structure, so the backbone path is a harmless placeholder.
            structure_path = cif if cif else bb["path"]
            candidate = Candidate(
                structure_path=structure_path,
                binder_chains=binder_chains,
                target_chains=target.chains,
                binder_sequence=row.get("seq"),
                name=cand,
            )
            values = scorer.score(candidate).values
            out: dict[str, Any] = {"cand": cand}
            for metric in _ROSETTA_METRICS:
                out[metric] = values.get(metric)
            _append_row(fh, out)
            scored += 1
            print(f"[rosetta] scored {cand}")
    print(f"[rosetta] scored {scored} new interfaces into {out_path}")
    return 0


# --------------------------------------------------------------------------------------------
# stage 5: reward + rank
# --------------------------------------------------------------------------------------------
def _write_table(rows: list[dict[str, Any]], output_path: Path) -> str:
    """Write ranked ``rows`` honoring ``output_path``'s extension (mirrors SelectionPipeline).

    ``.jsonl``/``.json`` -> JSON Lines (stdlib, regardless of polars); ``.parquet`` -> Parquet
    (requires polars); no/other extension -> Parquet when polars is available, else JSONL.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    def _jsonl(path: Path) -> str:
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return str(path)

    if suffix in (".jsonl", ".json"):
        return _jsonl(output_path)
    try:
        import polars as pl
    except ImportError:
        if suffix == ".parquet":
            raise RuntimeError(
                f"writing {output_path} as Parquet needs polars; install the grpo_steer 'full' "
                "extra or use a .jsonl output path."
            ) from None
        return _jsonl(output_path if suffix else output_path.with_suffix(".jsonl"))
    out = output_path if suffix else output_path.with_suffix(".parquet")
    pl.DataFrame(rows).write_parquet(out)
    return str(out)


def cmd_reward(args: argparse.Namespace) -> int:
    """Join the structure + Rosetta manifests, score the gated reward, and write a ranked table."""
    run = _run_dir(args)
    structure_rows = _read_jsonl(run / "structure.jsonl")
    rosetta_by_cand = {str(r["cand"]): r for r in _read_jsonl(run / "rosetta.jsonl")}

    reward_cfg = _load_reward_cfg(args.reward_config)
    reward = REWARDS.create("ipsae_gated", **reward_cfg)

    ranked: list[dict[str, Any]] = []
    for srow in structure_rows:
        cand = str(srow["cand"])
        merged: dict[str, float] = {}
        for metric in (*_STRUCTURE_METRICS, "scrmsd"):
            value = srow.get(metric)
            if isinstance(value, (int, float)):
                merged[metric] = float(value)
        rrow = rosetta_by_cand.get(cand, {})
        for metric in _ROSETTA_METRICS:
            value = rrow.get(metric)
            if isinstance(value, (int, float)):
                merged[metric] = float(value)
        reward_value = reward([Score(values=merged)])
        ranked.append({"cand": cand, "reward": reward_value, **merged})

    ranked.sort(key=lambda r: r["reward"], reverse=True)

    output_path = Path(args.output) if args.output else run / "ranked.parquet"
    written = _write_table(ranked, output_path)
    print(f"[reward] ranked {len(ranked)} candidates -> {written}")
    for row in ranked[: args.top]:
        print(f"  {row['cand']}  reward={row['reward']:.4f}  ipsae={row.get('ipsae')}")
    return 0


# --------------------------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------------------------
def _add_common(parser: argparse.ArgumentParser) -> None:
    """Add the global args shared by every stage (run dir, mock switch, target spec)."""
    parser.add_argument("--run", required=True, help="Work directory holding the stage manifests.")
    parser.add_argument(
        "--mock", action="store_true", help="Use the CPU mock adapters (no GPU/weights)."
    )
    parser.add_argument(
        "--target-json",
        default=None,
        help="JSON with structure_path/chains/binder_chains/hotspots/binder_length.",
    )
    parser.add_argument("--structure-path", default=None, help="Target structure path (override).")
    parser.add_argument("--chains", default=None, help="Comma-separated target chain ids.")
    parser.add_argument(
        "--binder-chains", default=None, help="Comma-separated default binder chain ids."
    )
    parser.add_argument("--hotspots", default=None, help="Comma-separated target hotspot residues.")
    parser.add_argument(
        "--binder-length", type=int, default=None, help="Binder length for de-novo sampling."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``aapg-grpo-stage`` argument parser (one subcommand per stage)."""
    parser = argparse.ArgumentParser(
        prog="aapg-grpo-stage",
        description=(
            "Stage-wise, memory-safe, resumable GRPO binder pipeline: run each stage "
            "(rfd3 -> mpnn -> fold -> rosetta -> reward) as its own process."
        ),
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    p_rfd3 = sub.add_parser("rfd3", help="Sample binder backbones.")
    _add_common(p_rfd3)
    p_rfd3.add_argument("--n", type=int, default=8, help="Number of backbones to sample.")
    p_rfd3.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p_rfd3.add_argument("--checkpoint", default=None, help="RFD3 checkpoint path (real run).")
    p_rfd3.add_argument("--rfd3-command", default=None, help="RFD3 argv prefix (e.g. 'rfd3').")
    p_rfd3.add_argument("--force", action="store_true", help="Re-sample even if N already exist.")
    p_rfd3.set_defaults(func=cmd_rfd3)

    p_mpnn = sub.add_parser("mpnn", help="Design sequences on each backbone.")
    _add_common(p_mpnn)
    p_mpnn.add_argument("--seqs-per", type=int, default=2, help="Sequences designed per backbone.")
    p_mpnn.add_argument("--designer", default="proteinmpnn", help="Designer backend name.")
    p_mpnn.add_argument("--weights-set", default=None, help="MPNN weight set (e.g. soluble).")
    p_mpnn.add_argument("--checkpoint", default=None, help="Designer checkpoint path (real run).")
    p_mpnn.add_argument("--seed", type=int, default=0, help="Design seed.")
    p_mpnn.set_defaults(func=cmd_mpnn)

    p_fold = sub.add_parser("fold", help="Fold + score each candidate complex.")
    _add_common(p_fold)
    p_fold.add_argument(
        "--backend",
        default="boltz2",
        help="Structure scorer: boltz2 | structure_auto | protenix_v2.",
    )
    p_fold.add_argument("--recycling-steps", type=int, default=1, help="Folding recycling steps.")
    p_fold.add_argument("--diffusion-samples", type=int, default=1, help="Diffusion samples.")
    p_fold.add_argument(
        "--min-free-gb",
        type=float,
        default=4.0,
        help="Bail out (exit 75) before a fold if free memory drops below this floor.",
    )
    p_fold.add_argument("--weights-dir", default=None, help="Boltz-2 weights dir (optional).")
    p_fold.add_argument(
        "--protenix-weights-dir", default=None, help="Protenix v2 weights dir (licensed)."
    )
    p_fold.set_defaults(func=cmd_fold)

    p_ros = sub.add_parser("rosetta", help="Score the PyRosetta interface.")
    _add_common(p_ros)
    p_ros.set_defaults(func=cmd_rosetta)

    p_rew = sub.add_parser("reward", help="Join scores, apply the gated reward, rank.")
    _add_common(p_rew)
    p_rew.add_argument("--output", default=None, help="Ranked table path (ext picks the format).")
    p_rew.add_argument("--reward-config", default=None, help="Reward YAML (default: configs/).")
    p_rew.add_argument("--top", type=int, default=10, help="How many top rows to print.")
    p_rew.set_defaults(func=cmd_reward)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one stage. Returns a process exit code (``0`` ok, ``75`` memory bail-out)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Any = args.func
    result = handler(args)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
