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


# --- Partitions and edge cuts -------------------------------------------------


class PartSchema(msgspec.Struct, frozen=True, tag="part"):
    mechanism: tuple[int, ...]
    purview: tuple[int, ...]


class NullCutSchema(msgspec.Struct, frozen=True, tag="null_cut"):
    indices: tuple[int, ...]


class DirectedBipartitionSchema(msgspec.Struct, frozen=True, tag="directed_bipartition"):
    direction: DirectionSchema
    from_nodes: tuple[int, ...]
    to_nodes: tuple[int, ...]


class JointPartitionSchema(msgspec.Struct, frozen=True, tag="joint_partition"):
    parts: tuple[PartSchema, ...]


class JointBipartitionSchema(msgspec.Struct, frozen=True, tag="joint_bipartition"):
    part0: PartSchema
    part1: PartSchema


class JointTripartitionSchema(msgspec.Struct, frozen=True, tag="joint_tripartition"):
    parts: tuple[PartSchema, ...]


JointPartitionSchemas = (
    JointPartitionSchema | JointBipartitionSchema | JointTripartitionSchema
)


class DirectedJointPartitionSchema(
    msgspec.Struct, frozen=True, tag="directed_joint_partition"
):
    direction: DirectionSchema
    partition: JointPartitionSchemas


class EdgeCutSchema(msgspec.Struct, frozen=True, tag="edge_cut"):
    node_indices: tuple[int, ...]
    cut_matrix: bytes
    node_labels: NodeLabelsSchema | None


class CompleteEdgeCutSchema(msgspec.Struct, frozen=True, tag="complete_edge_cut"):
    node_indices: tuple[int, ...]
    node_labels: NodeLabelsSchema | None


class DirectedSetPartitionSchema(
    msgspec.Struct, frozen=True, tag="directed_set_partition"
):
    node_indices: tuple[int, ...]
    cut_matrix: bytes
    set_partition: tuple[tuple[int, ...], ...]
    node_labels: NodeLabelsSchema | None


# Any concrete partition / edge cut (the building-block Part is separate).
PartitionSchema = (
    NullCutSchema
    | DirectedBipartitionSchema
    | DirectedJointPartitionSchema
    | EdgeCutSchema
    | CompleteEdgeCutSchema
    | DirectedSetPartitionSchema
    | JointPartitionSchema
    | JointBipartitionSchema
    | JointTripartitionSchema
)


# The tagged union grows one member per serializable type.
Schema = (
    DirectionSchema
    | PyPhiFloatSchema
    | DistanceResultSchema
    | NodeLabelsSchema
    | StateSpecificationSchema
    | SystemStateSpecificationSchema
    | PartSchema
    | NullCutSchema
    | DirectedBipartitionSchema
    | DirectedJointPartitionSchema
    | EdgeCutSchema
    | CompleteEdgeCutSchema
    | DirectedSetPartitionSchema
    | JointPartitionSchema
    | JointBipartitionSchema
    | JointTripartitionSchema
)
