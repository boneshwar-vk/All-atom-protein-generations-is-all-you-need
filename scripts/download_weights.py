#!/usr/bin/env python3
"""Fetch model weights into the git-ignored ``data/weights/`` tree.

Weights are NEVER committed to this repo (size + licensing). This script puts the OPEN weights
"in place" under ``data/weights/<model>/`` and tells a licensed user exactly where to drop the
proprietary ones (Protenix v2).

Policy per model (the authoritative manifest is the ``MANIFEST`` list below; the sibling
``weights_manifest.yaml`` mirrors it for humans):

- **boltz2** (MIT, open): Boltz downloads its own weights to its cache on first ``boltz predict``;
  nothing to fetch here. Point ``data/weights/boltz2`` at that cache (or let Boltz manage it).
- **proteinmpnn** (MIT, open): weights ship WITH the ``packages/proteinmpnn`` fork once it is
  vendored (the fork is a placeholder today).
- **rfd3** (BSD-3, open): fetched by the Foundry CLI — ``foundry install rfd3`` (into ~/.foundry);
  run that where you have the RFD3 stack.
- **protenix_v1** (Apache-2.0, open): auto-downloads via the ``protenix`` CLI on first run.
- **protenix_v2** (PROPRIETARY): not downloadable here and never committed. If you are licensed,
  download the v2 weights yourself and place them under ``data/weights/protenix_v2/``.
- **esm2** (MIT, open): optional; auto-downloads via huggingface on first use.

With ``--fetch`` and an entry that has a direct ``url``, the file is downloaded and (if a sha256
is given) checksum-verified. Everything else prints its instruction. The script is safe to run
offline: it only creates the destination directories and reports the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WeightSpec:
    name: str
    license: str
    open_weights: bool
    method: str  # "auto" | "foundry" | "in_repo" | "manual" | "url"
    dest: str
    note: str = ""
    url: str | None = None
    sha256: str | None = None
    files: list[str] = field(default_factory=list)


MANIFEST: list[WeightSpec] = [
    WeightSpec(
        "boltz2",
        "MIT",
        True,
        "auto",
        "boltz2",
        note="Boltz downloads weights to its cache on first `boltz predict`; or point here.",
    ),
    WeightSpec(
        "proteinmpnn",
        "MIT",
        True,
        "in_repo",
        "proteinmpnn",
        note="Ships WITH the ProteinMPNN fork once vendored (packages/proteinmpnn is a "
        "placeholder today); no separate download then.",
    ),
    WeightSpec(
        "rfd3",
        "BSD-3-Clause",
        True,
        "foundry",
        "rfd3",
        note="Install with `foundry install rfd3` (RosettaCommons/foundry) into ~/.foundry.",
    ),
    WeightSpec(
        "protenix_v1",
        "Apache-2.0",
        True,
        "auto",
        "protenix_v1",
        note="Auto-downloads via the `protenix` CLI (model protenix_base_default_v1.0.0).",
    ),
    WeightSpec(
        "protenix_v2",
        "PROPRIETARY",
        False,
        "manual",
        "protenix_v2",
        note="LICENSED ONLY. Not downloaded/committed. If licensed, place the v2 weights here.",
    ),
    WeightSpec(
        "esm2",
        "MIT",
        True,
        "auto",
        "esm2",
        note="Optional. Auto-downloads via huggingface (facebook/esm2_*) on first use.",
    ),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch/place model weights under data/weights/.")
    parser.add_argument("--dest", type=Path, default=Path("data/weights"))
    parser.add_argument("--fetch", action="store_true", help="download entries that have a URL")
    parser.add_argument("--only", nargs="*", default=None, help="limit to these model names")
    args = parser.parse_args(argv)

    specs = [s for s in MANIFEST if args.only is None or s.name in args.only]
    print(f"Weights root: {args.dest}  (git-ignored; weights are never committed)\n")
    for spec in specs:
        target = args.dest / spec.dest
        target.mkdir(parents=True, exist_ok=True)
        tag = "OPEN" if spec.open_weights else "PROPRIETARY"
        print(f"- {spec.name} [{spec.license}, {tag}] -> {target}")
        print(f"    {spec.note}")
        if args.fetch and spec.method == "url" and spec.url:
            for fname in spec.files or [spec.url.rsplit("/", 1)[-1]]:
                out = target / fname
                print(f"    downloading {spec.url} -> {out}")
                urllib.request.urlretrieve(spec.url, out)  # noqa: S310 (manifest-controlled URL)
                if spec.sha256 and _sha256(out) != spec.sha256:
                    raise SystemExit(f"checksum mismatch for {out}")
        elif args.fetch and spec.method != "url":
            print(f"    (skip --fetch: '{spec.method}' weights are obtained via their own tool)")
    print("\nDone. Proprietary weights (protenix_v2) must be placed by a licensed user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
