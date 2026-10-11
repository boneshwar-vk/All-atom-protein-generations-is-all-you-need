"""End-to-end pipeline entrypoint (placeholder).

Will become a Hydra app (``@hydra.main(config_path=..., config_name="config")``) that composes
the ``experiment/`` preset: generate -> design -> validate. For now it is an import-safe stub.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point (placeholder — not yet implemented)."""
    argv = sys.argv[1:] if argv is None else argv
    print("aapg end2end pipeline: placeholder — not implemented yet.")
    print("When implemented, this will be a Hydra app reading configs/experiment/*.")
    print(f"(received args: {argv})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
