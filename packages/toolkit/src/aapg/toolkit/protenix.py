"""Protenix v2 structure-prediction scorer.

Wraps the ``protenix`` prediction CLI to fold a binder/target complex, then derives interface
confidence metrics from its outputs: the model's own ipTM / PAE / pLDDT summaries plus the
ipSAE interface score computed from the full token-pair PAE (see :mod:`aapg.toolkit.ipsae`).

All heavy dependencies (``gemmi`` and the ``protenix`` binary) are imported/invoked lazily
inside the methods, so importing this module and registering the scorers works even when those
tools are absent (e.g. in CI). A deterministic ``MockProtenixScorer`` is provided for CPU tests
and dry runs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from aapg.core.scoring import SCORERS, Candidate, Score, Scorer

__all__ = ["ProtenixScorer", "MockProtenixScorer"]

# Canonical names returned by the protenix summary confidence JSON.
_SUMMARY_GLOB = "*_summary_confidence_*.json"
_FULL_GLOB = "*_full_data_*.json"


def _derive_unit(seed: str, salt: str) -> float:
    """Return a deterministic float in ``[0, 1)`` from ``seed`` and ``salt``."""
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _scale(unit: float, low: float, high: float) -> float:
    return low + unit * (high - low)


def _to_plddt100(value: float) -> float:
    """Normalise a scalar pLDDT to the 0-100 scale (×100 if it looks like a 0-1 value)."""
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _plddt_array_100(arr: Any) -> Any:
    """Normalise a per-residue pLDDT array to 0-100 (×100 if it looks like a 0-1 scale)."""
    import numpy as np

    a = np.asarray(arr, dtype=float)
    if a.size and float(np.nanmax(a)) <= 1.0:
        return a * 100.0
    return a


@SCORERS.register("protenix_v2")
class ProtenixScorer(Scorer):
    """Fold a binder/target complex with Protenix v2 and score the interface."""

    name = "protenix_v2"

    def __init__(
        self,
        model_name: str = "protenix-v2",
        seeds: tuple[int, ...] = (101,),
        n_step: int = 200,
        n_cycle: int = 10,
        work_dir: str | None = None,
        weights_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.seeds = tuple(seeds)
        self.n_step = n_step
        self.n_cycle = n_cycle
        self.work_dir = work_dir
        # Protenix v2 weights are proprietary and must be supplied by a licensed user here; v1
        # (Apache-2.0) auto-downloads. See configs/scorer/protenix.yaml.
        self.weights_dir = weights_dir
        self.extra = kwargs

    def _check_licensed_weights(self) -> None:
        """Fail loudly if a v2 run has no licensed weights in place (never silently wrong)."""
        from pathlib import Path

        is_v2 = "v2" in self.model_name or self.model_name == "protenix-v2"
        if not is_v2:
            return
        wd = self.weights_dir
        if not wd or not Path(wd).is_dir() or not any(Path(wd).iterdir()):
            raise RuntimeError(
                "Protenix v2 weights are proprietary and were not found. Set scorer.weights_dir "
                f"to your licensed v2 weights (looked in {wd!r}), or use the default open Boltz-2 "
                "backend ('boltz2'), or Protenix v1 (model_name=protenix_base_default_v1.0.0)."
            )

    # -- sequence extraction -------------------------------------------------------------
    def _sequences_from_structure(self, path: str, chains: tuple[str, ...]) -> dict[str, str]:
        """Read one-letter sequences for ``chains`` from a structure file (lazy gemmi)."""
        try:
            import gemmi
        except ImportError as exc:  # pragma: no cover - requires gemmi
            raise RuntimeError(
                "gemmi is required to derive sequences from a structure; install the "
                "'tools' extra of aapg-toolkit."
            ) from exc

        structure = gemmi.read_structure(path)
        model = structure[0]
        out: dict[str, str] = {}
        for chain_id in chains:
            chain = model[chain_id]
            out[chain_id] = gemmi.one_letter_code([res.name for res in chain]).upper()
        return out

    def _build_input_json(self, candidate: Candidate, out_dir: Path) -> Path:
        """Assemble the single-job Protenix input JSON and write it to ``out_dir``."""
        sequences: dict[str, str] = {}
        target_seqs = candidate.metadata.get("target_sequences", {})
        if isinstance(target_seqs, dict):
            sequences.update(target_seqs)
        if candidate.binder_sequence is not None:
            for chain_id in candidate.binder_chains:
                sequences[chain_id] = candidate.binder_sequence

        missing = [
            c for c in (*candidate.target_chains, *candidate.binder_chains) if c not in sequences
        ]
        if missing:
            if candidate.structure_path is None:
                raise RuntimeError(
                    f"no sequence available for chains {missing} and no structure to derive "
                    "them from"
                )
            sequences.update(
                self._sequences_from_structure(candidate.structure_path, tuple(missing))
            )

        protein_chains = [
            {"proteinChain": {"sequence": sequences[c], "count": 1, "chainId": c}}
            for c in (*candidate.target_chains, *candidate.binder_chains)
        ]
        job = {"name": candidate.name or "aapg_complex", "sequences": protein_chains}
        input_path = out_dir / "input.json"
        input_path.write_text(json.dumps([job], indent=2))
        return input_path

    # -- prediction ----------------------------------------------------------------------
    def _run_protenix(self, input_path: Path, out_dir: Path) -> None:
        """Shell out to the ``protenix`` CLI (lazy; raises if the binary is missing)."""
        seed_str = ",".join(str(s) for s in self.seeds)
        cmd = [
            "protenix",
            "pred",
            "-i",
            str(input_path),
            "-o",
            str(out_dir),
            "-s",
            seed_str,
            "-n",
            self.model_name,
            "--need_atom_confidence",
            "true",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "the 'protenix' binary was not found on PATH; install the Protenix tool "
                "(aapg-toolkit 'tools' extra) or run its container."
            ) from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover - requires protenix
            raise RuntimeError(f"protenix prediction failed: {exc.stderr}") from exc

    def _read_outputs(self, out_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
        """Locate and load the summary JSON, full-data JSON and prediction CIF."""
        summary_files = sorted(out_dir.rglob(_SUMMARY_GLOB))
        full_files = sorted(out_dir.rglob(_FULL_GLOB))
        cif_files = sorted(out_dir.rglob("*.cif"))
        if not summary_files or not full_files or not cif_files:
            raise RuntimeError(f"protenix outputs not found under {out_dir}")
        summary = json.loads(summary_files[0].read_text())
        full = json.loads(full_files[0].read_text())
        return summary, full, str(cif_files[0])

    def _residue_repr_from_cif(self, cif_path: str) -> tuple[Any, list[str], Any]:
        """Read per-residue representative coords, chain ids and pLDDT from a CIF (lazy gemmi).

        One representative atom per residue (CB, or CA for glycine). Protenix writes per-atom
        pLDDT (0-100) into the B-factor column, so the representative atom's B-factor is the
        residue pLDDT used by pDockQ. Returned arrays share one residue-axis length.
        """
        import gemmi  # pragma: no cover - requires gemmi
        import numpy as np

        structure = gemmi.read_structure(cif_path)
        model = structure[0]
        coords: list[list[float]] = []
        chain_ids: list[str] = []
        plddt: list[float] = []
        for chain in model:
            for res in chain:
                atom = res.find_atom("CB", "*") or res.find_atom("CA", "*")
                if atom is None:
                    continue
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                chain_ids.append(chain.name)
                plddt.append(float(atom.b_iso))
        return np.asarray(coords, dtype=float), chain_ids, np.asarray(plddt, dtype=float)

    def score(self, candidate: Candidate) -> Score:
        import numpy as np

        from aapg.toolkit.ipsae import compute_ipsae, interface_pae

        self._check_licensed_weights()
        base = Path(self.work_dir) if self.work_dir else Path.cwd() / "protenix_runs"
        out_dir = base / (candidate.name or "aapg_complex")
        out_dir.mkdir(parents=True, exist_ok=True)

        input_path = self._build_input_json(candidate, out_dir)
        self._run_protenix(input_path, out_dir)
        summary, full, cif_path = self._read_outputs(out_dir)

        iptm = float(summary.get("iptm", 0.0))
        complex_plddt = _to_plddt100(float(summary.get("plddt", 0.0)))
        has_clash = float(summary.get("has_clash", 0.0) or 0.0)

        pae = np.asarray(full["token_pair_pae"], dtype=float)
        # token_asym_id is aligned to the PAE token axis, so it is the authoritative chain map
        # for the PAE-indexed metrics. coords/pLDDT come from the CIF (residue axis); for a
        # standard protein complex the token and residue axes are 1:1 and compute_ipsae's length
        # guard raises loudly if that assumption is violated rather than scoring silently wrong.
        token_chain = [str(c) for c in full.get("token_asym_id", [])]
        coords, cif_chains, res_plddt = self._residue_repr_from_cif(cif_path)
        res_plddt = _plddt_array_100(res_plddt)
        chains = token_chain if len(token_chain) == pae.shape[0] else cif_chains

        ipsae = compute_ipsae(pae, coords, chains, res_plddt)
        # Interface iPAE from the token PAE cross-chain block (same helper as Boltz, so the
        # ipae gates mean the same thing across backends; this avoids the chain_pair_pae summary
        # which includes intra-chain diagonal blocks that dilute the mean / zero the min).
        ipae_min, ipae_mean = interface_pae(
            pae, chains, candidate.binder_chains, candidate.target_chains
        )

        return Score(
            values={
                "ipsae": float(ipsae["ipsae"]),
                "iptm": iptm,
                "ipae_min": ipae_min,
                "ipae_mean": ipae_mean,
                "complex_plddt": complex_plddt,
                "has_clash": has_clash,
            },
            metadata={
                "scorer": self.name,
                "model_name": self.model_name,
                "seeds": list(self.seeds),
                "out_dir": str(out_dir),
                "cif_path": cif_path,
                "ipsae_pairs": {k: v for k, v in ipsae.items() if "_" in k},
            },
        )


@SCORERS.register("protenix_v2_mock")
class MockProtenixScorer(Scorer):
    """Deterministic CPU stand-in for :class:`ProtenixScorer` (no heavy imports)."""

    name = "protenix_v2_mock"

    def __init__(self, **kwargs: Any) -> None:
        self.extra = kwargs

    def score(self, candidate: Candidate) -> Score:
        seed = candidate.binder_sequence or candidate.name or ""
        return Score(
            values={
                "ipsae": round(_scale(_derive_unit(seed, "ipsae"), 0.3, 0.9), 4),
                "iptm": round(_scale(_derive_unit(seed, "iptm"), 0.4, 0.95), 4),
                "ipae_min": round(_scale(_derive_unit(seed, "ipae_min"), 2.0, 15.0), 4),
                "ipae_mean": round(_scale(_derive_unit(seed, "ipae_mean"), 5.0, 25.0), 4),
                "complex_plddt": round(_scale(_derive_unit(seed, "plddt"), 60.0, 95.0), 4),
                "has_clash": float(_derive_unit(seed, "clash") < 0.5),
            },
            metadata={"scorer": self.name, "mock": True},
        )
