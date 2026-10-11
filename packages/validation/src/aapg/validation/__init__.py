"""aapg.validation — module 3: complex validation pipeline (placeholder).

Takes backbones and validates the resulting complexes through a generic staged pipeline:
ProteinMPNN (sequence design) -> Protenix v2 (structure prediction) -> OpenDDE (energy/binding
evaluation) -> verdict. Stages are typed and composed from config so new stages slot in.
"""

__version__ = "0.0.0"
