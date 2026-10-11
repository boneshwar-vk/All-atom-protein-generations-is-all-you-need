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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Backbone",
    "Target",
    "RFD3Sampler",
    "MockRFD3Sampler",
    "get_sampler",
]


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
    """Sample binder backbones with the vendored RFdiffusion3 model.

    The heavy backend is imported lazily in :meth:`sample_backbones`, so constructing the
    sampler (and importing this module) never requires ``rfd3`` or model weights.
    """

    def __init__(self, checkpoint: str | None = None, **cfg: Any) -> None:
        self.checkpoint = checkpoint
        self.cfg = cfg

    def sample_backbones(self, target: Target, n: int, seed: int = 0, **cfg: Any) -> list[Backbone]:
        """Generate ``n`` backbones for ``target`` (lazy heavy import).

        Raises:
            RuntimeError: if the vendored ``rfd3`` package cannot be imported, with guidance on
                vendoring RFD3 and providing weights.
        """
        try:
            from rfd3.model import inference_sampler  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "RFdiffusion3 is not available: could not import 'rfd3.model.inference_sampler'. "
                "Vendor the upstream source with scripts/vendor_rfd3.sh and provide model "
                "weights (set 'checkpoint'), or use get_sampler(mock=True) for CPU dry runs."
            ) from exc
        # Real sampling wiring is added when the RFD3 source is vendored; see the FUTURE-RL
        # HOOK in the module docstring for the log-prob recording this call will grow.
        raise NotImplementedError(
            "RFD3Sampler.sample_backbones is wired up when rfd3 is vendored; the lazy import "
            "above succeeded but the real sampling loop is not implemented in v1."
        )


class MockRFD3Sampler(RFD3Sampler):
    """Deterministic, dependency-free stand-in for :class:`RFD3Sampler`.

    Emits tiny seed-derived backbones without writing structure files. Each backbone carries a
    deterministic pseudo-CA coordinate array in ``metadata["ca_coords"]`` so downstream scRMSD
    and mock scoring have geometry to work with. No heavy imports.
    """

    def sample_backbones(self, target: Target, n: int, seed: int = 0, **cfg: Any) -> list[Backbone]:
        length = int(target.binder_length)
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
