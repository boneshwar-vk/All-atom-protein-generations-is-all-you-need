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
        work_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        # Boltz auto-downloads its (open, MIT) weights to its cache on first use; set weights_dir
        # to pin a local copy (e.g. data/weights/boltz2) for offline/container runs.
        self.weights_dir = weights_dir
        self.recycling_steps = recycling_steps
        self.diffusion_samples = diffusion_samples
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
        """Load the confidence JSON, the PAE array and the predicted CIF."""
        import numpy as np

        conf_files = sorted(out_dir.rglob(_CONFIDENCE_GLOB))
        pae_files = sorted(out_dir.rglob(_PAE_GLOB))
        cif_files = sorted(out_dir.rglob("*.cif"))
        if not conf_files or not pae_files or not cif_files:
            raise RuntimeError(f"boltz outputs not found under {out_dir}")
        confidence = json.loads(conf_files[0].read_text())
        pae_npz = np.load(pae_files[0])
        pae = pae_npz[pae_npz.files[0]]
        return confidence, pae, str(cif_files[0])

    def _residue_repr_from_cif(self, cif_path: str) -> tuple[Any, list[str], Any]:
        """Per-residue representative coords, chain ids and pLDDT (from B-factor) via gemmi."""
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
        from aapg.toolkit.ipsae import compute_ipsae

        base = Path(self.work_dir) if self.work_dir else Path.cwd() / "boltz_runs"
        out_dir = base / (candidate.name or "aapg_complex")
        out_dir.mkdir(parents=True, exist_ok=True)

        input_path = self._write_input_yaml(candidate, out_dir)
        self._run_boltz(input_path, out_dir)
        confidence, pae, cif_path = self._read_outputs(out_dir)

        iptm = _reduce(confidence.get("iptm"), "mean")
        # Boltz reports complex_plddt on a 0-1 scale; normalise to 0-100 so it is comparable
        # across backends and matches the reward's complex_plddt gate (min 80).
        complex_plddt = _to_plddt100(_reduce(confidence.get("complex_plddt"), "mean"))
        has_clash = float(confidence.get("has_clash", 0.0) or 0.0)

        coords, chains, res_plddt = self._residue_repr_from_cif(cif_path)
        res_plddt = _plddt_array_100(res_plddt)
        ipsae = compute_ipsae(pae, coords, chains, res_plddt)

        # Interface PAE summaries from the cross-chain block of the token PAE.
        ipae_min, ipae_mean = _interface_pae(pae, chains, candidate)

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
                "model": self.model,
                "out_dir": str(out_dir),
                "cif_path": cif_path,
                "ipsae_pairs": {k: v for k, v in ipsae.items() if "_" in k},
            },
        )


def _interface_pae(pae: Any, chains: list[str], candidate: Candidate) -> tuple[float, float]:
    """Min and mean PAE over the binder<->target cross-chain block (0 if unavailable)."""
    import numpy as np

    arr = np.asarray(pae, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or len(chains) != arr.shape[0]:
        return 0.0, 0.0
    chain_arr = np.asarray(chains)
    binder = np.isin(chain_arr, list(candidate.binder_chains))
    target = np.isin(chain_arr, list(candidate.target_chains))
    if not binder.any() or not target.any():
        return 0.0, 0.0
    block = np.concatenate(
        [arr[np.ix_(binder, target)].ravel(), arr[np.ix_(target, binder)].ravel()]
    )
    if block.size == 0:
        return 0.0, 0.0
    return float(np.nanmin(block)), float(np.nanmean(block))


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
