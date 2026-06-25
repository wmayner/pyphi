"""msgspec schema types for serializing PyPhi results.

Each serializable type has one frozen ``msgspec.Struct`` carrying a unique
string ``tag``. ``Schema`` is the tagged union of all of them; msgspec uses the
tag to validate and dispatch on decode. Adding a type means adding its Struct
here and registering its converter in :mod:`pyphi.serialize.convert`.
"""

from typing import Any

import msgspec


class DirectionSchema(msgspec.Struct, frozen=True, tag="direction"):
    name: str


# --- Simple value types -------------------------------------------------------


class PyPhiFloatSchema(msgspec.Struct, frozen=True, tag="pyphi_float"):
    value: float


class DistanceResultSchema(msgspec.Struct, frozen=True, tag="distance_result"):
    value: float
    aux: dict[str, Any] = msgspec.field(default_factory=dict)


# A phi value is either a plain float or a distance result with auxiliary data;
# the Struct tag preserves which type to reconstruct.
PhiSchema = PyPhiFloatSchema | DistanceResultSchema


class NodeLabelsSchema(msgspec.Struct, frozen=True, tag="node_labels"):
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]


class StateSpecificationSchema(msgspec.Struct, frozen=True, tag="state_specification"):
    direction: DirectionSchema
    purview: tuple[int, ...]
    state: tuple[int, ...]
    intrinsic_information: PhiSchema
    repertoire: bytes
    unconstrained_repertoire: bytes
    tie_peers: tuple["StateSpecificationSchema", ...] = ()


class SystemStateSpecificationSchema(
    msgspec.Struct, frozen=True, tag="system_state_specification"
):
    cause: StateSpecificationSchema
    effect: StateSpecificationSchema


StateSpecSchema = StateSpecificationSchema | SystemStateSpecificationSchema


# The tagged union grows one member per serializable type.
Schema = (
    DirectionSchema
    | PyPhiFloatSchema
    | DistanceResultSchema
    | NodeLabelsSchema
    | StateSpecificationSchema
    | SystemStateSpecificationSchema
)
