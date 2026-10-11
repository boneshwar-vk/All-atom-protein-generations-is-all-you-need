"""aapg.toolkit — adapters to external protein tools.

Each adapter wraps a tool that runs in its own container/binary (Rosetta, Protenix v2, OpenDDE,
ESM, TM-align, Foldseek, ...) and implements an ``aapg.core.scoring`` interface so it can be
registered and resolved by name.

Importing this package runs the registration side effects of the concrete scorer modules, so
the scorers (and their CPU mock variants) are available via ``aapg.core.scoring.SCORERS`` after
``import aapg.toolkit``. Heavy backends are imported lazily inside the scorers' methods, so the
imports below succeed even when those tools are absent.
"""

from __future__ import annotations

# Registration side effects: importing these modules registers the scorers into SCORERS.
from aapg.toolkit import ipsae, protenix, rosetta  # noqa: F401

__version__ = "0.0.0"
