"""Rosetta (PyRosetta) interface scorer for binder/target complexes.

Ported from the project's ``local_binder_interface_scores.py``: relaxes the complex, runs the
InterfaceAnalyzerMover, and reports interface energetics and biophysical descriptors
(interface dG, shape complementarity, contact molecular surface, normalised SAP, isoelectric
point, and a handful of cheap extras).

PyRosetta and Biopython are imported lazily inside the methods, so importing this module and
registering the scorers succeeds without them. A deterministic ``MockRosettaScorer`` is
provided for CPU tests and dry runs.
"""

from __future__ import annotations

import hashlib
from typing import Any

from aapg.core.scoring import SCORERS, Candidate, Score, Scorer

__all__ = ["RosettaInterfaceScorer", "MockRosettaScorer"]

# Module-level guard so PyRosetta is initialised at most once per process.
_PYROSETTA_INITIALISED = False


def _ensure_pyrosetta() -> Any:
    """Lazily import and initialise PyRosetta, returning the ``pyrosetta`` module."""
    global _PYROSETTA_INITIALISED
    try:
        import pyrosetta as pr
    except ImportError as exc:  # pragma: no cover - requires pyrosetta
        raise RuntimeError(
            "PyRosetta is required for the rosetta_interface scorer; install the 'tools' "
            "extra of aapg-toolkit (PyRosetta is separately licensed)."
        ) from exc
    if not _PYROSETTA_INITIALISED:
        pr.init(
            "-beta_nov16 -relax:default_repeats 1 -mute all -use_input_sc -ignore_unrecognized_res"
        )
        _PYROSETTA_INITIALISED = True
    return pr


def relax(pose: Any, steps: int = 200, repeats: int = 2) -> None:  # pragma: no cover
    """FastRelax the pose in place, then realign it back onto its starting coordinates."""
    pr = _ensure_pyrosetta()
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover

    start_pose = pose.clone()
    mmf = MoveMap()
    mmf.set_chi(True)
    mmf.set_bb(True)
    mmf.set_jump(False)
    fr = FastRelax(pr.get_fa_scorefxn(), repeats)
    fr.set_movemap(mmf)
    fr.max_iter(steps)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)
    align = AlignChainMover()
    align.source_chain(0)
    align.target_chain(0)
    align.pose(start_pose)
    align.apply(pose)


def contact_molecular_surface(
    pose: Any, binder_sel: Any, target_sel: Any
) -> float:  # pragma: no cover
    """Contact molecular surface across the binder/target interface (NaN on failure)."""
    try:
        from pyrosetta.rosetta.protocols.simple_filters import (
            ContactMolecularSurfaceFilter,
        )
    except ImportError:
        from pyrosetta.rosetta.protocols.denovo_design.filters import (
            ContactMolecularSurfaceFilter,
        )

    cms = ContactMolecularSurfaceFilter()
    cms.selector1(binder_sel)
    cms.selector2(target_sel)
    cms.distance_weight(0.5)
    try:
        r = float(cms.report_sm(pose))
        return r if r >= 0 else float("nan")
    except RuntimeError:
        return float("nan")


def _make_interface_analyzer(interface: str, scorefxn: Any) -> Any:  # pragma: no cover
    """Build an InterfaceAnalyzerMover for ``interface`` (e.g. ``"B_A"``)."""
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    try:
        from pyrosetta.rosetta.protocols.analysis import DockingPartners

        iam = InterfaceAnalyzerMover(DockingPartners.docking_partners_from_string(interface))
    except (ImportError, AttributeError, TypeError):
        iam = InterfaceAnalyzerMover(interface)
    iam.set_scorefunction(scorefxn)
    iam.set_compute_interface_energy(True)
    iam.set_compute_packstat(True)
    iam.set_pack_separated(True)
    return iam


def process_file(
    file: str,
    binder_chains: tuple[str, ...],
    target_chains: tuple[str, ...],
    relax_perform: bool = True,
    relax_steps: int = 200,
    relax_repeats: int = 2,
) -> dict[str, float]:  # pragma: no cover - requires pyrosetta
    """Relax and analyse the binder/target interface of a complex structure file."""
    pr = _ensure_pyrosetta()
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

    binder_str = "".join(binder_chains)
    target_str = "".join(target_chains)
    binder_sel = ChainSelector(binder_str)
    target_sel = ChainSelector(target_str)

    pose = pr.pose_from_file(file)
    if relax_perform:
        relax(pose, steps=relax_steps, repeats=relax_repeats)

    scorefxn = pr.get_fa_scorefxn()
    interface = f"{target_str}_{binder_str}"
    iam = _make_interface_analyzer(interface, scorefxn)
    iam.apply(pose)
    data = iam.get_all_data()

    cms = contact_molecular_surface(pose, binder_sel, target_sel)
    n_binder = int(binder_sel.apply(pose).count(True)) if hasattr(binder_sel, "apply") else 0

    out = {
        "interface_dg": float(iam.get_interface_dG()),
        "interface_dsasa": float(iam.get_interface_delta_sasa()),
        "interface_hbonds": float(data.interface_hbonds),
        "shape_complementarity": float(data.sc_value),
        "interface_packstat": float(iam.get_interface_packstat()),
        "contact_molecular_surface": float(cms),
        "per_res_reu": float(scorefxn(pose) / max(pose.total_residue(), 1)),
        "_n_binder": float(n_binder),
    }
    return out


def _sap_normalized(pose: Any, binder_chains: tuple[str, ...]) -> float:  # pragma: no cover
    """Spatial aggregation propensity over the binder, normalised to 65 residues."""
    from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import calculate_sap
    from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

    binder_sel = ChainSelector("".join(binder_chains))
    sap = float(calculate_sap(pose, binder_sel, binder_sel, binder_sel))
    n_binder = int(binder_sel.apply(pose).count(True))
    return sap * 65.0 / max(n_binder, 1)


def _isoelectric_point(sequence: str) -> float:
    """Isoelectric point of ``sequence`` via Biopython (lazy import)."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return float(ProteinAnalysis(sequence).isoelectric_point())


@SCORERS.register("rosetta_interface")
class RosettaInterfaceScorer(Scorer):
    """Relax a complex and score its binder/target interface with PyRosetta."""

    name = "rosetta_interface"

    def __init__(
        self,
        relax: bool = True,
        relax_steps: int = 200,
        relax_repeats: int = 2,
        weights: str = "ref2015",
        **kwargs: Any,
    ) -> None:
        self.relax = relax
        self.relax_steps = relax_steps
        self.relax_repeats = relax_repeats
        self.weights = weights
        self.extra = kwargs

    def score(self, candidate: Candidate) -> Score:  # pragma: no cover - requires pyrosetta
        if candidate.structure_path is None:
            raise RuntimeError("rosetta_interface requires candidate.structure_path")
        pr = _ensure_pyrosetta()

        metrics = process_file(
            candidate.structure_path,
            candidate.binder_chains,
            candidate.target_chains,
            relax_perform=self.relax,
            relax_steps=self.relax_steps,
            relax_repeats=self.relax_repeats,
        )

        pose = pr.pose_from_file(candidate.structure_path)
        if self.relax:
            relax(pose, steps=self.relax_steps, repeats=self.relax_repeats)
        sap_norm = _sap_normalized(pose, candidate.binder_chains)

        binder_seq = candidate.binder_sequence
        if binder_seq is None:
            from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

            sel = ChainSelector("".join(candidate.binder_chains)).apply(pose)
            binder_seq = "".join(
                pose.residue(i + 1).name1() for i in range(pose.total_residue()) if sel[i + 1]
            )
        pi = _isoelectric_point(binder_seq)

        return Score(
            values={
                "interface_dg": metrics["interface_dg"],
                "shape_complementarity": metrics["shape_complementarity"],
                "contact_molecular_surface": metrics["contact_molecular_surface"],
                "sap_normalized": sap_norm,
                "isoelectric_point": pi,
                "interface_dsasa": metrics["interface_dsasa"],
                "interface_hbonds": metrics["interface_hbonds"],
                "interface_packstat": metrics["interface_packstat"],
                "per_res_reu": metrics["per_res_reu"],
            },
            metadata={
                "scorer": self.name,
                "weights": self.weights,
                "relaxed": self.relax,
                "structure_path": candidate.structure_path,
            },
        )


def _derive_unit(seed: str, salt: str) -> float:
    """Deterministic float in ``[0, 1)`` from ``seed`` and ``salt``."""
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _scale(unit: float, low: float, high: float) -> float:
    return low + unit * (high - low)


@SCORERS.register("rosetta_interface_mock")
class MockRosettaScorer(Scorer):
    """Deterministic CPU stand-in for :class:`RosettaInterfaceScorer` (no heavy imports)."""

    name = "rosetta_interface_mock"

    def __init__(self, **kwargs: Any) -> None:
        self.extra = kwargs

    def score(self, candidate: Candidate) -> Score:
        seed = candidate.binder_sequence or candidate.name or ""
        return Score(
            values={
                "interface_dg": round(_scale(_derive_unit(seed, "dg"), -45.0, -5.0), 4),
                "shape_complementarity": round(_scale(_derive_unit(seed, "sc"), 0.5, 0.78), 4),
                "contact_molecular_surface": round(
                    _scale(_derive_unit(seed, "cms"), 150.0, 600.0), 4
                ),
                "sap_normalized": round(_scale(_derive_unit(seed, "sap"), 10.0, 60.0), 4),
                "isoelectric_point": round(_scale(_derive_unit(seed, "pi"), 4.5, 9.5), 4),
            },
            metadata={"scorer": self.name, "mock": True},
        )
