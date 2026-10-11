"""Thin adapter over the vendored RFdiffusion3 backbone sampler (module 1).

This module gives the GRPO selection pipeline a stable, importable surface for generating
binder backbones against a fixed target, without forcing the heavy RFD3 stack (torch, the
vendored ``rfd3`` package, model weights) to be importable here. The real sampler
(:class:`RFD3Sampler`) imports ``rfd3`` **lazily** inside :meth:`RFD3Sampler.sample_backbones`
and raises a clear, actionable error when it is absent, and a dependency-free
:class:`MockRFD3Sampler` emits tiny deterministic backbones for tests and dry runs. Pick one
through :func:`get_sampler`.

FUTURE-RL HOOK (NOT IMPLEMENTED)
--------------------------------
v1 is best-of-N / hill-climb *selection*: backbones are sampled i.i.d. and ranked, and no
sampler parameters are updated. A later policy-gradient GRPO trainer (see
``aapg.grpo.advantage`` for the group-relative advantage math and ``aapg.grpo.trainer`` for the
``rl`` mode) will need the *log-probability of each sampling step* under the current policy to
form the surrogate loss. The recording point is the per-step reverse-diffusion update inside
``rfd3``'s ``inference_sampler``: each step draws a Gaussian "churn" perturbation
``x_{t-1} = mu_theta(x_t, t) + sigma_t * eps`` with ``eps ~ N(0, I)``, whose log-prob
``log N(x_{t-1}; mu_theta, sigma_t^2 I)`` is what GRPO differentiates. When that trainer is
built it must:

  * record those per-step log-probs into ``Backbone.metadata["step_logprobs"]`` during
    sampling, and
  * **disable the sampler's random realignment / re-centering** (any post-hoc rigid transform
    of the generated coordinates), because it is applied outside the recorded transition
    density and would make the stored log-probs inconsistent with the returned coordinates.

Neither is done here; v1 never reads a log-prob.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Backbone",
    "Target",
    "RFD3Sampler",
    "MockRFD3Sampler",
    "get_sampler",
    "parse_pdb_ca",
]


def parse_pdb_ca(
    path: str,
    include_chains: tuple[str, ...] | None = None,
    exclude_chains: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Parse CA coordinates and their chain ids from a PDB file (stdlib, no heavy deps).

    Reads fixed-column ``ATOM`` records, keeping one CA per residue. ``include_chains`` keeps only
    those chains; ``exclude_chains`` drops them (used to isolate the binder = everything that is
    not the target). Returns ``(coords (N,3), chain_ids length N)``.
    """
    include = set(include_chains) if include_chains else None
    exclude = set(exclude_chains) if exclude_chains else set()
    coords: list[list[float]] = []
    chains: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        # Exclude calcium ions (atom name "CA", element "CA"): keep only carbon CA (the
        # protein backbone CA, element "C"). If the element column is absent, only trust ATOM.
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if element:
            if element != "C":
                continue
        elif not line.startswith("ATOM"):
            continue
        chain = line[21].strip() or "A"
        if (include is not None and chain not in include) or chain in exclude:
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:  # pragma: no cover - malformed line
            continue
        coords.append([x, y, z])
        chains.append(chain)
    return np.asarray(coords, dtype=float), chains


@dataclass
class Backbone:
    """A generated binder backbone.

    Attributes:
        path: Path to the backbone structure (CIF/PDB). For mock backbones this is a synthetic
            ``mock://`` string that is never written to disk.
        length: Number of binder residues in the backbone.
        metadata: Free-form provenance. The mock sampler stores a deterministic pseudo-CA
            coordinate array under ``metadata["ca_coords"]`` (numpy, shape ``(length, 3)``) so
            downstream scRMSD / mock scoring has geometry to consume.
    """

    path: str
    length: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Target:
    """The fixed target a binder is designed against.

    Attributes:
        structure_path: Path to the target structure (CIF/PDB), or ``None`` for sequence-only
            / de-novo specification.
        chains: Chain IDs belonging to the target.
        hotspots: Target residue ids the binder should engage, e.g. ``("A52", "A56")``.
        binder_length: Desired length of the generated binder chain.
        metadata: Free-form extra context carried through sampling.
    """

    structure_path: str | None = None
    chains: tuple[str, ...] = ("B",)
    hotspots: tuple[str, ...] = ()
    binder_length: int = 80
    metadata: dict[str, Any] = field(default_factory=dict)


class RFD3Sampler:
    """Sample binder backbones by driving the RFdiffusion3 (foundry) CLI as a subprocess.

    RFD3 ships via RosettaCommons/foundry and is configured with Hydra-style ``key=value``
    overrides, so this adapter runs it out-of-process (no torch imported here) and parses the
    generated complex structures. The adapter owns the mechanics it can get right -- running the
    command, collecting the output PDBs, and extracting the binder CA trace -- while the exact
    config KEYS for the target/hotspots/contig are **version-dependent** and supplied through
    config, since RFD3 is new and its flag names must be matched to the installed build.

    Config (``**cfg`` / constructor args):
        command: argv prefix to invoke RFD3, e.g. ``["rfd3"]`` (default), ``["foundry", "run",
            "rfd3"]`` or ``["python", "-m", "rfd3"]``.
        config_name / config_path: optional Hydra ``--config-name`` / ``--config-dir`` to pass.
        spec: dict of Hydra overrides describing the design task for YOUR RFD3 version, e.g.
            ``{"ppi.hotspot_res": "[A52,A56]", "contigmap.contigs": ...}``. These keys are not
            hardcoded because they differ across RFD3 releases.
        overrides_fn: optional callable ``(target, n, seed, out_dir) -> list[str]`` that fully
            builds the override list, if ``spec`` is not expressive enough.
        extra_args: extra argv appended verbatim.
        output_glob: glob for generated structures under the run dir (default ``**/*.pdb``).

    The heavy stack lives in the subprocess interpreter; nothing here imports torch/rfd3.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        command: list[str] | None = None,
        config_name: str | None = None,
        config_path: str | None = None,
        spec: dict[str, Any] | None = None,
        overrides_fn: Any = None,
        extra_args: list[str] | None = None,
        output_glob: str = "**/*.pdb",
        work_dir: str | None = None,
        **cfg: Any,
    ) -> None:
        self.checkpoint = checkpoint
        self.command = list(command) if command else ["rfd3"]
        self.config_name = config_name
        self.config_path = config_path
        self.spec: dict[str, Any] = dict(spec or {})
        self.overrides_fn = overrides_fn
        self.extra_args = list(extra_args or [])
        self.output_glob = output_glob
        self.work_dir = work_dir
        self.cfg = cfg

    def _build_command(self, target: Target, n: int, seed: int, out_dir: Path) -> list[str]:
        """Assemble the RFD3 argv (Hydra-style overrides). Override ``spec``/``overrides_fn``
        to match your RFD3 version's exact config keys."""
        cmd = list(self.command)
        if self.config_path:
            cmd += ["--config-dir", self.config_path]
        if self.config_name:
            cmd += ["--config-name", self.config_name]
        if self.overrides_fn is not None:
            overrides = list(self.overrides_fn(target, n, seed, out_dir))
        else:
            # Minimal, widely-needed knobs as Hydra overrides; the design SPEC (target pdb,
            # hotspots, contig/binder length) comes from self.spec because its keys are
            # version-dependent. These three names also vary across releases -- override `spec`
            # (which is appended last and wins) if your build differs.
            overrides = [
                f"inference.num_designs={n}",
                f"inference.output_prefix={out_dir / 'design'}",
                f"inference.seed={seed}",
            ]
            if self.checkpoint:
                overrides.append(f"inference.ckpt_path={self.checkpoint}")
            if target.structure_path:
                overrides.append(f"inference.input_pdb={target.structure_path}")
            if target.hotspots:
                overrides.append("ppi.hotspot_res=[" + ",".join(target.hotspots) + "]")
            overrides.append(f"contigmap.length={target.binder_length}-{target.binder_length}")
        overrides += [f"{k}={v}" for k, v in self.spec.items()]
        return cmd + overrides + self.extra_args

    def sample_backbones(self, target: Target, n: int, seed: int = 0, **cfg: Any) -> list[Backbone]:
        """Generate ``n`` binder backbones for ``target`` via the RFD3 CLI.

        Returns one :class:`Backbone` per generated complex PDB, with the binder CA trace (every
        chain NOT in ``target.chains``) stored in ``metadata["ca_coords"]`` for scRMSD.

        Raises:
            RuntimeError: if the RFD3 command is missing/fails or produces no structures.
        """
        base = Path(self.work_dir) if self.work_dir else Path(tempfile.mkdtemp(prefix="aapg_rfd3_"))
        out_dir = base / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_command(target, n, seed, out_dir)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"RFdiffusion3 command {self.command!r} not found. Install/vendor RFD3 "
                "(scripts/vendor_rfd3.sh; `foundry install rfd3`) and set the adapter's "
                "`command`, or use get_sampler(mock=True) for CPU dry runs."
            ) from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover - requires rfd3
            raise RuntimeError(
                "RFdiffusion3 run failed (check the Hydra overrides match your version):\n"
                + (exc.stderr or "")[-2000:]
            ) from exc

        pdbs = sorted(out_dir.glob(self.output_glob))
        if not pdbs:
            raise RuntimeError(
                f"RFD3 produced no structures matching {self.output_glob!r} under {out_dir}"
            )
        backbones: list[Backbone] = []
        for i, pdb in enumerate(pdbs):
            ca, _chains = parse_pdb_ca(str(pdb), exclude_chains=tuple(target.chains))
            # The actual binder chain ids as RFD3 wrote them, in first-seen order, are the single
            # source of truth downstream (build_candidate / selection must not re-guess them).
            binder_chains = tuple(dict.fromkeys(_chains))
            if not binder_chains:
                raise RuntimeError(
                    f"RFD3 output {pdb} has no binder chain (every CA chain is in the target "
                    f"chains {tuple(target.chains)!r}); cannot build a binder candidate."
                )
            overlap = set(binder_chains) & set(target.chains)
            if overlap:  # pragma: no cover - defensive; parse_pdb_ca already excludes target
                raise RuntimeError(
                    f"binder chains {binder_chains} overlap target chains "
                    f"{tuple(target.chains)!r} in {pdb}: {sorted(overlap)}"
                )
            backbones.append(
                Backbone(
                    path=str(pdb),
                    length=int(ca.shape[0]),
                    metadata={
                        "seed": seed,
                        "index": i,
                        "target_chains": tuple(target.chains),
                        "binder_chains": binder_chains,
                        "hotspots": tuple(target.hotspots),
                        "ca_coords": ca,
                    },
                )
            )
        return backbones


class MockRFD3Sampler(RFD3Sampler):
    """Deterministic, dependency-free stand-in for :class:`RFD3Sampler`.

    Emits tiny seed-derived backbones without writing structure files. Each backbone carries a
    deterministic pseudo-CA coordinate array in ``metadata["ca_coords"]`` so downstream scRMSD
    and mock scoring have geometry to work with. No heavy imports.
    """

    def sample_backbones(self, target: Target, n: int, seed: int = 0, **cfg: Any) -> list[Backbone]:
        length = int(target.binder_length)
        # Pick a deterministic single binder chain id disjoint from the target chains, so the mock
        # exposes the same metadata["binder_chains"] contract as the real sampler.
        binder_chain = next(
            (c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in set(target.chains)),
            "Z",
        )
        backbones: list[Backbone] = []
        for i in range(n):
            rng = np.random.default_rng(seed=(seed * 1_000_003 + i))
            # A crude but deterministic pseudo-backbone: a jittered 3.8 A-spaced CA walk.
            steps = rng.normal(loc=0.0, scale=1.0, size=(length, 3))
            norms = np.linalg.norm(steps, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            ca_coords = np.cumsum(steps / norms * 3.8, axis=0).astype(float)
            backbones.append(
                Backbone(
                    path=f"mock://rfd3/seed{seed}/backbone_{i:04d}.cif",
                    length=length,
                    metadata={
                        "mock": True,
                        "seed": seed,
                        "index": i,
                        "target_chains": tuple(target.chains),
                        "binder_chains": (binder_chain,),
                        "hotspots": tuple(target.hotspots),
                        "ca_coords": ca_coords,
                    },
                )
            )
        return backbones


def get_sampler(mock: bool = False, **cfg: Any) -> RFD3Sampler:
    """Return a :class:`MockRFD3Sampler` when ``mock`` else a real :class:`RFD3Sampler`."""
    if mock:
        return MockRFD3Sampler(**cfg)
    return RFD3Sampler(**cfg)
