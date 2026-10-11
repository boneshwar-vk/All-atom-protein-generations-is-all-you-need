"""aapg.mpnn_iter — module 4: iterative ProteinMPNN (placeholder).

Wraps the vendored ``proteinmpnn`` fork in an optimization loop: design sequences, score them
(Rosetta + sequence models via ``aapg.core.scoring`` / ``aapg.toolkit``), then re-design under
the feedback with an acceptance/annealing rule, repeating until a budget or convergence.
"""

__version__ = "0.0.0"
