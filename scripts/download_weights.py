#!/usr/bin/env python3
"""Download model weights listed in the manifest (placeholder).

Reads ``scripts/weights_manifest.yaml`` (name, url, sha256, dest) and fetches each file under
``data/weights/``, verifying checksums. Not implemented yet — this is a stub so the entrypoint
and manifest format exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MANIFEST = Path(__file__).parent / "weights_manifest.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dest", type=Path, default=Path("data/weights"))
    args = parser.parse_args()
    print(f"[placeholder] would download weights from {args.manifest} into {args.dest}")
    print("Not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
