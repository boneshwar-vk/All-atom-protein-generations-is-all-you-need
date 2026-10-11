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
import tempfile
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
        """Locate and load the summary JSON, full-data JSON and CIF for the SAME sample.

        Protenix emits one ``*_summary_confidence_*.json``, ``*_full_data_*.json`` and ``*.cif``
        per (seed, sample). Picking each glob independently could mix files from different seeds
        when several are requested; instead choose one summary file and match the full-data JSON
        and CIF that share its ``<prefix>`` (the text before ``_summary_confidence_``), falling
        back to the first available file only when no stem match exists.
        """
        summary_files = sorted(out_dir.rglob(_SUMMARY_GLOB))
        full_files = sorted(out_dir.rglob(_FULL_GLOB))
        cif_files = sorted(out_dir.rglob("*.cif"))
        if not summary_files or not full_files or not cif_files:
            raise RuntimeError(f"protenix outputs not found under {out_dir}")

        summary_path = summary_files[0]
        prefix = summary_path.name.split("_summary_confidence_")[0]
        full_path = next((p for p in full_files if p.name.startswith(prefix)), full_files[0])
        cif_path = next((p for p in cif_files if p.name.startswith(prefix)), cif_files[0])

        summary = json.loads(summary_path.read_text())
        full = json.loads(full_path.read_text())
        return summary, full, str(cif_path)

    def _residue_repr_from_cif(self, cif_path: str) -> tuple[Any, Any, list[str], Any]:
        """Read per-residue coords, chain ids and pLDDT from a CIF (lazy gemmi).

        Returns ``(repr_coords, ca_coords, chain_ids, plddt)``. ``repr_coords`` is the CB (CA for
        glycine / when CB is absent) used for ipSAE interface distances; ``ca_coords`` is the CA
        **only**, used for scRMSD against the designed CA backbone. A residue is counted only if
        it has a CA, so the arrays stay index-aligned. Protenix writes per-atom pLDDT (0-100) into
        the B-factor column, so the CA B-factor is the residue pLDDT used by pDockQ.
        """
        import gemmi  # pragma: no cover - requires gemmi
        import numpy as np

        structure = gemmi.read_structure(cif_path)
        model = structure[0]
        repr_coords: list[list[float]] = []
        ca_coords: list[list[float]] = []
        chain_ids: list[str] = []
        plddt: list[float] = []
        for chain in model:
            for res in chain:
                ca = res.find_atom("CA", "*")
                if ca is None:
                    continue
                rep = res.find_atom("CB", "*") or ca
                repr_coords.append([rep.pos.x, rep.pos.y, rep.pos.z])
                ca_coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
                chain_ids.append(chain.name)
                plddt.append(float(ca.b_iso))
        return (
            np.asarray(repr_coords, dtype=float),
            np.asarray(ca_coords, dtype=float),
            chain_ids,
            np.asarray(plddt, dtype=float),
        )

    def score(self, candidate: Candidate) -> Score:
        import numpy as np

        from aapg.toolkit.ipsae import compute_ipsae, entity_order_index, interface_pae

        self._check_licensed_weights()

        # Everything needed downstream is parsed into memory below, so unless the caller pins a
        # persistent work_dir, run under a TemporaryDirectory that is removed once parsed -- at
        # GRPO scale the per-complex outputs would otherwise fill the disk.
        ephemeral = self.work_dir is None
        tmp = tempfile.TemporaryDirectory(prefix="aapg_protenix_") if ephemeral else None
        base = Path(tmp.name) if tmp is not None else Path(self.work_dir)  # type: ignore[arg-type]
        try:
            out_dir = base / (candidate.name or "aapg_complex")
            out_dir.mkdir(parents=True, exist_ok=True)
            input_path = self._build_input_json(candidate, out_dir)
            self._run_protenix(input_path, out_dir)
            summary, full, cif_path = self._read_outputs(out_dir)
            repr_coords, ca_coords, cif_chains, res_plddt = self._residue_repr_from_cif(cif_path)
            out_dir_str: str | None = None if ephemeral else str(out_dir)
            cif_str: str | None = None if ephemeral else cif_path
        finally:
            if tmp is not None:
                tmp.cleanup()

        iptm = float(summary.get("iptm", 0.0))
        complex_plddt = _to_plddt100(float(summary.get("plddt", 0.0)))

        pae = np.asarray(full["token_pair_pae"], dtype=float)
        res_plddt = _plddt_array_100(res_plddt)

        # token_asym_id is aligned to the PAE token axis, so it is the authoritative chain map for
        # the PAE-indexed metrics. The CIF arrays are on the residue axis; for a standard protein
        # complex the two axes are 1:1 but NOT necessarily in the same chain order, so reorder
        # BOTH the token axis (PAE + token chains) and the CIF arrays to one canonical entity order
        # (target chains, then binder) before pairing them. compute_ipsae's length guard raises
        # loudly if the 1:1 assumption is violated rather than scoring silently wrong.
        token_chain = [str(c) for c in full.get("token_asym_id", [])]
        entity_order = (*candidate.target_chains, *candidate.binder_chains)
        if len(token_chain) == pae.shape[0]:
            tok = entity_order_index(token_chain, entity_order)
            pae = pae[np.ix_(tok, tok)]
            chains = [token_chain[i] for i in tok.tolist()]
        else:
            # No usable token map: fall back to CIF chain order for the PAE axis.
            chains = cif_chains
        cif_order = entity_order_index(cif_chains, entity_order)
        repr_coords = repr_coords[cif_order] if repr_coords.size else repr_coords
        ca_coords = ca_coords[cif_order] if ca_coords.size else ca_coords
        res_plddt = res_plddt[cif_order] if res_plddt.size else res_plddt
        cif_chains = [cif_chains[i] for i in cif_order.tolist()]

        ipsae = compute_ipsae(pae, repr_coords, chains, res_plddt)
        # Interface iPAE from the token PAE cross-chain block (same helper as Boltz, so the
        # ipae gates mean the same thing across backends; this avoids the chain_pair_pae summary
        # which includes intra-chain diagonal blocks that dilute the mean / zero the min).
        ipae_min, ipae_mean = interface_pae(
            pae, chains, candidate.binder_chains, candidate.target_chains
        )

        # Predicted binder CA trace (CA-only, comparable to the designed backbone) for scRMSD.
        chain_arr = np.asarray(cif_chains)
        binder_mask = np.isin(chain_arr, list(candidate.binder_chains))
        predicted_ca = ca_coords[binder_mask] if binder_mask.any() else ca_coords

        values: dict[str, float] = {
            "ipsae": float(ipsae["ipsae"]),
            "iptm": iptm,
            "ipae_min": ipae_min,
            "ipae_mean": ipae_mean,
            "complex_plddt": complex_plddt,
        }
        # Only emit has_clash when Protenix actually reports it: a defaulted 0.0 would fake a
        # "no clash" pass through the reward's hard clash gate.
        if "has_clash" in summary and summary.get("has_clash") is not None:
            values["has_clash"] = float(summary["has_clash"])

        return Score(
            values=values,
            metadata={
                "scorer": self.name,
                "model_name": self.model_name,
                "seeds": list(self.seeds),
                "out_dir": out_dir_str,  # None when run under a cleaned-up temp dir
                "cif_path": cif_str,  # None when ephemeral (file already removed)
                "ipsae_pairs": {k: v for k, v in ipsae.items() if "_" in k},
                "predicted_ca_coords": predicted_ca,
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
