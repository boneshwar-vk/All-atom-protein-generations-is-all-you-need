"""aapg.pipelines — end-to-end orchestration (placeholder).

Composes the four modules into runnable pipelines:

- ``generate``  : ellipsoid-constrained + GRPO-steered backbone generation (modules 1 + 2)
- ``design``    : iterative ProteinMPNN sequence design (module 4)
- ``validate``  : complex validation (module 3)
- ``end2end``   : generate -> design -> validate

Each is a Hydra entrypoint reading the shared ``configs/`` tree.
"""

__version__ = "0.0.0"
