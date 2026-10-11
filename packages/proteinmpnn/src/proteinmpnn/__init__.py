"""proteinmpnn — vendored fork of ProteinMPNN.

The upstream ProteinMPNN source and its (small, MIT-licensed) model weights are vendored under
:mod:`proteinmpnn.vendor` (see the repo-root ``VENDORING.md`` for the exact commit). The GRPO
sequence-design step uses them through :mod:`proteinmpnn.adapter`.

Importing this package stays light: the vendored modules import torch/numpy at module load, so
they are imported lazily by the adapter (never here). Module 4 (iterative ProteinMPNN) will edit
the vendored source directly.
"""

from __future__ import annotations

__version__ = "0.0.0"
# Upstream source + weights are vendored (dauparas/ProteinMPNN, MIT). See VENDORING.md.
__vendored__ = True
