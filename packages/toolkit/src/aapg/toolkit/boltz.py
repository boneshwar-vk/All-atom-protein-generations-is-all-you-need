"""Boltz-2 structure-prediction scorer (default open-weights interface backend).

Wraps the ``boltz`` prediction CLI to fold a binder/target complex, then derives the same
interface confidence metrics as the Protenix backend -- the model's own ipTM / PAE / pLDDT plus
the ipSAE interface score computed from the token-pair PAE (see :mod:`aapg.toolkit.ipsae`). This
is the project's **default** structure scorer because Boltz-2 is MIT-licensed with openly
downloadable, commercially-usable weights (Boltz weights auto-download from Hugging Face on first
run), so it keeps the pipeline fully open. See ``protenix.py`` for the licensed Protenix v2
alternative.

All heavy dependencies (``gemmi``, ``numpy`` and the ``boltz`` binary) are imported/invoked
lazily inside the methods, so importing this module and registering the scorers works even when
those tools are absent (e.g. in CI). A deterministic ``MockBoltzScorer`` is provided for CPU
tests and dry runs.

Boltz output layout (version-dependent -- confirm against your installed Boltz): predictions land
under ``<out>/predictions/<name>/`` as ``<name>_model_0.cif`` plus a ``confidence_*.json``
(keys ``complex_plddt``/``iptm``/``ptm``/``complex_pde`` and ``pair_chains_iptm``) and
``pae_<name>_model_0.npz`` (the token-pair PAE). The CIF B-factor column carries per-residue
pLDDT.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from aapg.core.scoring import SCORERS, Candidate, Score, Scorer

__all__ = ["BoltzScorer", "MockBoltzScorer"]

_CONFIDENCE_GLOB = "confidence_*.json"
_PAE_GLOB = "pae_*.npz"


def _derive_unit(seed: str, salt: str) -> float:
    """Return a deterministic float in ``[0, 1)`` from ``seed`` and ``salt``."""
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _scale(unit: float, low: float, high: float) -> float:
    return low + unit * (high - low)


def _reduce(value: Any, how: str) -> float:
    """Reduce a confidence field to a scalar: pass a scalar through, else min/mean an array."""
    if value is None:
        return 0.0
    import numpy as np

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 0:
        return 0.0
    return float(np.nanmin(arr) if how == "min" else np.nanmean(arr))


def _to_plddt100(value: float) -> float:
    """Normalise pLDDT to the 0-100 scale (Boltz reports 0-1; gates/metrics expect 0-100)."""
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _plddt_array_100(arr: Any) -> Any:
    """Normalise a per-residue pLDDT array to 0-100 (×100 if it looks like a 0-1 scale)."""
    import numpy as np

    a = np.asarray(arr, dtype=float)
    if a.size and float(np.nanmax(a)) <= 1.0:
        return a * 100.0
    return a


@SCORERS.register("boltz2")
class BoltzScorer(Scorer):
    """Fold a binder/target complex with Boltz-2 and score the interface (default backend)."""

    name = "boltz2"

    def __init__(
        self,
        model: str = "boltz2",
        weights_dir: str | None = None,
        recycling_steps: int = 3,
        diffusion_samples: int = 1,
        use_msa: bool = False,
        extra_args: list[str] | None = None,
        work_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        # Boltz auto-downloads its (open, MIT) weights to its cache on first use; set weights_dir
        # to pin a local copy (e.g. data/weights/boltz2) for offline/container runs.
        self.weights_dir = weights_dir
        self.recycling_steps = recycling_steps
        self.diffusion_samples = diffusion_samples
        # Default to single-sequence (msa: empty): at GRPO scale (thousands of complexes) an MSA
        # server is impractical and non-deterministic, and binder pipelines run single-sequence.
        # Set use_msa=True to let Boltz build MSAs (adds --use_msa_server).
        self.use_msa = use_msa
        self.extra_args = list(extra_args or [])
        self.work_dir = work_dir
        self.extra = kwargs

    # -- inputs ---------------------------------------------------------------------------
    def _sequences(self, candidate: Candidate) -> dict[str, str]:
        """Resolve {chain_id: sequence} for the target + binder chains (lazy gemmi)."""
        seqs: dict[str, str] = {}
        target_seqs = candidate.metadata.get("target_sequences", {})
        if isinstance(target_seqs, dict):
            seqs.update(target_seqs)
        if candidate.binder_sequence is not None:
            for chain_id in candidate.binder_chains:
                seqs[chain_id] = candidate.binder_sequence

        missing = [c for c in (*candidate.target_chains, *candidate.binder_chains) if c not in seqs]
        if missing:
            if candidate.structure_path is None:
                raise RuntimeError(
                    f"no sequence available for chains {missing} and no structure to derive them"
                )
            import gemmi  # pragma: no cover - requires gemmi

            structure = gemmi.read_structure(candidate.structure_path)
            model = structure[0]
            for chain_id in missing:
                chain = model[chain_id]
                seqs[chain_id] = gemmi.one_letter_code([res.name for res in chain]).upper()
        return seqs

    def _write_input_yaml(self, candidate: Candidate, out_dir: Path) -> Path:
        """Write the Boltz YAML job (one protein entity per target/binder chain)."""
        seqs = self._sequences(candidate)
        lines = ["version: 1", "sequences:"]
        for chain_id in (*candidate.target_chains, *candidate.binder_chains):
            lines += [
                "  - protein:",
                f"      id: {chain_id}",
                f"      sequence: {seqs[chain_id]}",
            ]
            if not self.use_msa:
                lines.append("      msa: empty")  # single-sequence; no MSA server
        input_path = out_dir / "input.yaml"
        input_path.write_text("\n".join(lines) + "\n")
        return input_path

    # -- prediction -----------------------------------------------------------------------
    def _run_boltz(self, input_path: Path, out_dir: Path) -> None:
        """Shell out to the ``boltz`` CLI (lazy; raises if the binary is missing)."""
        cmd = [
            "boltz",
            "predict",
            str(input_path),
            "--out_dir",
            str(out_dir),
            "--output_format",
            "mmcif",
            "--recycling_steps",
            str(self.recycling_steps),
            "--diffusion_samples",
            str(self.diffusion_samples),
            "--write_full_pae",
        ]
        if self.weights_dir:
            cmd += ["--cache", self.weights_dir]
        if self.use_msa:
            cmd += ["--use_msa_server"]
        cmd += self.extra_args
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "the 'boltz' binary was not found on PATH; install the Boltz-2 tool "
                "(aapg-toolkit 'tools' extra: `pip install boltz`) or run its container. "
                "Boltz weights (MIT) download automatically on first run."
            ) from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover - requires boltz
            raise RuntimeError(f"boltz prediction failed: {exc.stderr}") from exc

    def _read_outputs(self, out_dir: Path) -> tuple[dict[str, Any], Any, str]:
        """Load the confidence JSON, PAE array and CIF for the SAME predicted model.

        Boltz names its files ``confidence_<stem>.json`` / ``pae_<stem>.npz`` / ``<stem>.cif`` per
        sampled model (``<stem>`` ends in ``_model_<i>``; ``model_0`` is Boltz's top-ranked
        prediction). Picking each glob's first match independently could mix files from different
        samples when ``diffusion_samples > 1``; instead select the best-ranked confidence file and
        derive the PAE and CIF from its stem so all three describe one structure.
        """
        import numpy as np

        conf_files = sorted(out_dir.rglob(_CONFIDENCE_GLOB))
        if not conf_files:
            raise RuntimeError(f"boltz confidence output not found under {out_dir}")
        # Prefer model_0 (top-ranked); fall back to the lexicographically first confidence file.
        conf_path = next((p for p in conf_files if "model_0" in p.name), conf_files[0])
        stem = conf_path.name[len("confidence_") : -len(".json")]

        pae_path = conf_path.with_name(f"pae_{stem}.npz")
        cif_path = conf_path.with_name(f"{stem}.cif")
        if not pae_path.exists():  # fall back to any matching PAE under the run dir
            pae_candidates = sorted(out_dir.rglob(_PAE_GLOB))
            if not pae_candidates:
                raise RuntimeError(f"boltz PAE output not found under {out_dir}")
            pae_path = pae_candidates[0]
        if not cif_path.exists():
            cif_candidates = sorted(out_dir.rglob("*.cif"))
            if not cif_candidates:
                raise RuntimeError(f"boltz CIF output not found under {out_dir}")
            cif_path = cif_candidates[0]

        confidence = json.loads(conf_path.read_text())
        pae_npz = np.load(pae_path)
        pae = pae_npz[pae_npz.files[0]]
        return confidence, pae, str(cif_path)

    def _residue_repr_from_cif(self, cif_path: str) -> tuple[Any, Any, list[str], Any]:
        """Per-residue coords, chain ids and pLDDT via gemmi.

        Returns ``(repr_coords, ca_coords, chain_ids, plddt)``. ``repr_coords`` is the CB (CA for
        glycine / when CB is absent) used for interface-distance in ipSAE; ``ca_coords`` is the CA
        **only**, used for scRMSD against the designed CA backbone (a CB trace would inflate the
        RMSD). A residue is counted only if it has a CA (true for every amino acid), so the two
        arrays and the chain/pLDDT lists stay index-aligned.
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
        from aapg.toolkit.ipsae import compute_ipsae, entity_order_index, interface_pae

        # At GRPO scale (thousands of complexes) Boltz outputs would fill the disk, and everything
        # needed downstream (confidence, PAE, coordinates) is parsed into memory below. So unless
        # the caller pins a persistent work_dir, run under a TemporaryDirectory that is removed
        # once parsed. When work_dir IS set the outputs persist (and cif_path/out_dir stay valid).
        ephemeral = self.work_dir is None
        tmp = tempfile.TemporaryDirectory(prefix="aapg_boltz_") if ephemeral else None
        base = Path(tmp.name) if tmp is not None else Path(self.work_dir)  # type: ignore[arg-type]
        try:
            out_dir = base / (candidate.name or "aapg_complex")
            out_dir.mkdir(parents=True, exist_ok=True)
            input_path = self._write_input_yaml(candidate, out_dir)
            self._run_boltz(input_path, out_dir)
            confidence, pae, cif_path = self._read_outputs(out_dir)
            repr_coords, ca_coords, chains, res_plddt = self._residue_repr_from_cif(cif_path)
            out_dir_str: str | None = None if ephemeral else str(out_dir)
            cif_str: str | None = None if ephemeral else cif_path
        finally:
            if tmp is not None:
                tmp.cleanup()

        import numpy as np

        iptm = _reduce(confidence.get("iptm"), "mean")
        # Boltz reports complex_plddt on a 0-1 scale; normalise to 0-100 so it is comparable
        # across backends and matches the reward's complex_plddt gate (min 80).
        complex_plddt = _to_plddt100(_reduce(confidence.get("complex_plddt"), "mean"))

        # The token PAE is in input-entity order (target chains, then binder); permute the
        # CIF-derived per-residue arrays to that same order before pairing them with the PAE,
        # so ipSAE and the interface-PAE block index the right residues.
        order = entity_order_index(chains, (*candidate.target_chains, *candidate.binder_chains))
        repr_coords = repr_coords[order] if repr_coords.size else repr_coords
        ca_coords = ca_coords[order] if ca_coords.size else ca_coords
        chains = [chains[i] for i in order.tolist()]
        res_plddt = _plddt_array_100(res_plddt[order] if res_plddt.size else res_plddt)

        ipsae = compute_ipsae(pae, repr_coords, chains, res_plddt)

        # Interface PAE from the cross-chain block of the token PAE (shared with Protenix).
        ipae_min, ipae_mean = interface_pae(
            pae, chains, candidate.binder_chains, candidate.target_chains
        )

        # Predicted binder CA trace -> enables a REAL scRMSD (designed backbone vs this
        # prediction) in the selection pipeline, which reads metadata["predicted_ca_coords"].
        # Use the CA-only trace (not the CB representative) so it is comparable to the designed
        # backbone CA coordinates.
        chain_arr = np.asarray(chains)
        binder_mask = np.isin(chain_arr, list(candidate.binder_chains))
        predicted_ca = ca_coords[binder_mask] if binder_mask.any() else ca_coords

        values: dict[str, float] = {
            "ipsae": float(ipsae["ipsae"]),
            "iptm": iptm,
            "ipae_min": ipae_min,
            "ipae_mean": ipae_mean,
            "complex_plddt": complex_plddt,
        }
        # Only emit has_clash when Boltz actually reports it: defaulting a missing field to 0.0
        # would fake a "no clash" pass through the reward's hard clash gate (vacuously true).
        if "has_clash" in confidence and confidence.get("has_clash") is not None:
            values["has_clash"] = float(confidence["has_clash"])

        return Score(
            values=values,
            metadata={
                "scorer": self.name,
                "model": self.model,
                "out_dir": out_dir_str,  # None when run under a cleaned-up temp dir
                "cif_path": cif_str,  # None when ephemeral (file already removed)
                "ipsae_pairs": {k: v for k, v in ipsae.items() if "_" in k},
                "predicted_ca_coords": predicted_ca,
            },
        )


# Plausible ranges for the deterministic mock, matching the real scorer's output keys.
_MOCK_RANGES: dict[str, tuple[float, float]] = {
    "ipsae": (0.3, 0.9),
    "iptm": (0.4, 0.95),
    "ipae_min": (2.0, 15.0),
    "ipae_mean": (5.0, 25.0),
    "complex_plddt": (60.0, 95.0),  # 0-100 scale, matching the real scorer's normalised output
}


@SCORERS.register("boltz2_mock")
class MockBoltzScorer(Scorer):
    """Deterministic CPU stand-in for :class:`BoltzScorer` (no heavy imports)."""

    name = "boltz2_mock"

    def __init__(self, **kwargs: Any) -> None:
        self.extra = kwargs

    def score(self, candidate: Candidate) -> Score:
        seed = candidate.binder_sequence or candidate.name or ""
        values = {
            k: round(_scale(_derive_unit(seed, k), lo, hi), 4)
            for k, (lo, hi) in _MOCK_RANGES.items()
        }
        values["has_clash"] = float(_derive_unit(seed, "clash") < 0.5)
        return Score(values=values, metadata={"scorer": self.name, "mock": True})
