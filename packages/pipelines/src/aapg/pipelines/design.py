"""Sequence-design entrypoint (placeholder).

Iterative ProteinMPNN sequence design over backbones (module 4). Hydra app stub.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point (placeholder — not yet implemented)."""
    argv = sys.argv[1:] if argv is None else argv
    print("aapg design pipeline: placeholder — not implemented yet.")
    print(f"(received args: {argv})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
