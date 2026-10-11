"""Adapter over the vendored ProteinMPNN sequence designer (module 1).

Gives the GRPO selection pipeline a stable surface for designing binder sequences onto a
generated backbone. The real :class:`MPNNDesigner` drives the vendored ``protein_mpnn_run.py``
as a subprocess (so torch is only needed in that subprocess, never imported into this process),
and raises a clear error when the backend/weights are unavailable. A dependency-free
:class:`MockMPNNDesigner` returns deterministic plausible sequences for tests and dry runs. Pick
one through :func:`get_designer`.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "MPNNDesigner",
    "MockMPNNDesigner",
    "get_designer",
    "CANONICAL_AMINO_ACIDS",
]

# The 20 canonical amino acids, in a fixed order, used to decode mock sequences.
CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Vendored weight sets (dauparas/ProteinMPNN); model_name selects the noise level, e.g. v_48_020.
_WEIGHT_SETS = {"vanilla": "vanilla_model_weights", "soluble": "soluble_model_weights"}


def _vendor_dir() -> Path:
    """Absolute path to the vendored ProteinMPNN source directory."""
    import proteinmpnn

    return Path(proteinmpnn.__file__).resolve().parent / "vendor"


def _parse_mpnn_fasta(path: Path) -> list[str]:
    """Return the designed sequences from a ProteinMPNN ``seqs/*.fa`` file, in order.

    The first record is the native/input sequence; subsequent records are the sampled designs.
    Multiple designed chains are '/'-separated in a record; they are concatenated here.
    """
    records: list[str] = []
    seq_parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if seq_parts:
                records.append("".join(seq_parts))
                seq_parts = []
        elif line.strip():
            seq_parts.append(line.strip())
    if seq_parts:
        records.append("".join(seq_parts))
    designs = records[1:]  # drop the native/input record
    return [d.replace("/", "") for d in designs]


class MPNNDesigner:
    """Design binder sequences onto a backbone with the vendored ProteinMPNN model.

    Runs ``protein_mpnn_run.py`` as a subprocess using the current interpreter (which, on a GPU
    box, has torch installed); this process never imports torch. ``checkpoint`` selects the
    model name (default ``v_48_020``), and ``weights_set`` picks ``vanilla`` or ``soluble``.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        temperature: float = 0.1,
        weights_set: str = "vanilla",
        python: str | None = None,
        **cfg: Any,
    ) -> None:
        self.model_name = checkpoint or "v_48_020"
        self.temperature = temperature
        if weights_set not in _WEIGHT_SETS:
            raise ValueError(f"unknown weights_set {weights_set!r}; expected {list(_WEIGHT_SETS)}")
        self.weights_set = weights_set
        self.python = python or sys.executable
        self.cfg = cfg

    def design(
        self,
        backbone: Any,
        n_seqs: int = 8,
        binder_chains: tuple[str, ...] = ("A",),
        target_chains: tuple[str, ...] = ("B",),
        seed: int = 0,
    ) -> list[str]:
        """Design ``n_seqs`` binder sequences for ``backbone`` via vendored ProteinMPNN.

        The binder chains are the ones redesigned; the remaining (target) chains are kept fixed
        as context. Raises :class:`RuntimeError` with guidance when the vendored source/weights
        or the backbone PDB are missing, or when the subprocess fails (e.g. torch absent).
        """
        vendor = _vendor_dir()
        run_script = vendor / "protein_mpnn_run.py"
        weights_dir = vendor / _WEIGHT_SETS[self.weights_set]
        if not run_script.exists() or not weights_dir.exists():
            raise RuntimeError(
                f"vendored ProteinMPNN not found under {vendor}; re-vendor the source/weights "
                "(see VENDORING.md) or use get_designer(mock=True)."
            )
        pdb_path = getattr(backbone, "path", None)
        if not pdb_path or not Path(pdb_path).exists():
            raise RuntimeError(
                f"real ProteinMPNN design needs a backbone PDB on disk; got {pdb_path!r}. "
                "Use get_designer(mock=True) for CPU dry runs."
            )

        # The designed sequences are parsed into memory below, so the run outputs are transient;
        # a TemporaryDirectory keeps them from accumulating across thousands of GRPO designs.
        with tempfile.TemporaryDirectory(prefix="aapg_mpnn_") as tmp:
            out_dir = Path(tmp)
            cmd = [
                self.python,
                str(run_script),
                "--pdb_path",
                str(pdb_path),
                "--pdb_path_chains",
                " ".join(binder_chains),
                "--path_to_model_weights",
                str(weights_dir),
                "--model_name",
                self.model_name,
                "--out_folder",
                str(out_dir),
                "--num_seq_per_target",
                str(n_seqs),
                "--sampling_temp",
                str(self.temperature),
                "--seed",
                str(seed),
                "--batch_size",
                "1",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(vendor))
            except FileNotFoundError as exc:  # pragma: no cover - interpreter missing
                raise RuntimeError(f"could not launch ProteinMPNN with {self.python!r}") from exc
            except subprocess.CalledProcessError as exc:  # pragma: no cover - requires torch
                raise RuntimeError(
                    "ProteinMPNN run failed (is torch installed in this interpreter?):\n"
                    + (exc.stderr or "")[-2000:]
                ) from exc

            fastas = sorted((out_dir / "seqs").glob("*.fa"))
            if not fastas:
                raise RuntimeError(f"ProteinMPNN produced no sequences under {out_dir}/seqs")
            return _parse_mpnn_fasta(fastas[0])[:n_seqs]


class MockMPNNDesigner(MPNNDesigner):
    """Deterministic, dependency-free stand-in for :class:`MPNNDesigner`.

    Returns ``n_seqs`` plausible binder sequences of ``backbone.length`` residues over the
    canonical amino acids, derived deterministically from ``seed`` and the backbone identity.
    No heavy imports or subprocesses.
    """

    def __init__(self, **cfg: Any) -> None:
        # Mock ignores weights/interpreter; keep a permissive signature.
        self.model_name = str(cfg.get("checkpoint") or "mock")
        self.temperature = float(cfg.get("temperature", 0.1))
        self.weights_set = "vanilla"
        self.python = sys.executable
        self.cfg = cfg

    def design(
        self,
        backbone: Any,
        n_seqs: int = 8,
        binder_chains: tuple[str, ...] = ("A",),
        target_chains: tuple[str, ...] = ("B",),
        seed: int = 0,
    ) -> list[str]:
        length = int(getattr(backbone, "length", 0))
        backbone_key = str(getattr(backbone, "path", backbone))
        sequences: list[str] = []
        for s in range(n_seqs):
            residues: list[str] = []
            counter = 0
            while len(residues) < length:
                digest = hashlib.sha256(f"{backbone_key}:{seed}:{s}:{counter}".encode()).digest()
                for byte in digest:
                    if len(residues) >= length:
                        break
                    residues.append(CANONICAL_AMINO_ACIDS[byte % len(CANONICAL_AMINO_ACIDS)])
                counter += 1
            sequences.append("".join(residues))
        return sequences


def get_designer(mock: bool = False, **cfg: Any) -> MPNNDesigner:
    """Return a :class:`MockMPNNDesigner` when ``mock`` else a real :class:`MPNNDesigner`."""
    if mock:
        return MockMPNNDesigner(**cfg)
    return MPNNDesigner(**cfg)
