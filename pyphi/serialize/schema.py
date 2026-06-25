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


# --- RIA and MICE -------------------------------------------------------------


class RIASchema(msgspec.Struct, frozen=True, tag="ria"):
    phi: PhiSchema
    direction: DirectionSchema
    mechanism: tuple[int, ...]
    mechanism_state: tuple[int, ...] | None
    purview: tuple[int, ...]
    purview_state: tuple[int, ...] | None
    partition: PartitionSchema
    repertoire: bytes | None
    partitioned_repertoire: bytes | None
    specified_state: StateSpecificationSchema | None
    node_labels: NodeLabelsSchema | None
    partition_tie_peers: tuple["RIASchema", ...] = ()
    state_tie_peers: tuple["RIASchema", ...] = ()


class MICESchema(msgspec.Struct, frozen=True, tag="mice"):
    ria: RIASchema


class MICECauseSchema(msgspec.Struct, frozen=True, tag="mice_cause"):
    ria: RIASchema


class MICEEffectSchema(msgspec.Struct, frozen=True, tag="mice_effect"):
    ria: RIASchema


MICEAnySchema = MICESchema | MICECauseSchema | MICEEffectSchema


# --- Distinctions -------------------------------------------------------------


class DistinctionSchema(msgspec.Struct, frozen=True, tag="distinction"):
    mechanism: tuple[int, ...] | None
    cause: MICEAnySchema
    effect: MICEAnySchema


# IIT 3.0 terminology calls a distinction a "concept".
ConceptSchema = DistinctionSchema


class DistinctionsSchema(msgspec.Struct, frozen=True, tag="distinctions"):
    concepts: tuple[DistinctionSchema, ...]


class UnresolvedDistinctionsSchema(
    msgspec.Struct, frozen=True, tag="unresolved_distinctions"
):
    concepts: tuple[DistinctionSchema, ...]


class ResolvedDistinctionsSchema(
    msgspec.Struct, frozen=True, tag="resolved_distinctions"
):
    concepts: tuple[DistinctionSchema, ...]


DistinctionsAnySchema = (
    DistinctionsSchema | UnresolvedDistinctionsSchema | ResolvedDistinctionsSchema
)


# --- Provenance, excluded candidates, and SIAs --------------------------------


class ProvenanceSchema(msgspec.Struct, frozen=True, tag="provenance"):
    pyphi_version: str
    git_sha: str | None
    git_dirty: bool | None
    timestamp: str
    python_version: str
    numpy_version: str
    scipy_version: str
    platform: str
    wall_time: float | None = None
    seed: int | None = None
    note: str | None = None


class ExcludedCandidateSchema(msgspec.Struct, frozen=True, tag="excluded_candidate"):
    node_indices: tuple[int, ...]
    phi: float


class IIT3SIASchema(msgspec.Struct, frozen=True, tag="iit3_sia"):
    phi: PhiSchema | None
    distinctions: DistinctionsAnySchema | None
    partitioned_distinctions: DistinctionsAnySchema | None
    partition: PartitionSchema | None
    node_indices: tuple[int, ...] | None
    node_labels: NodeLabelsSchema | None
    current_state: tuple[int, ...] | None
    tie_peers: tuple["IIT3SIASchema", ...] = ()


# Direction-keyed phi dict (e.g. intrinsic_differentiation) as ordered pairs.
DirectionPhiPairs = tuple[tuple[DirectionSchema, PhiSchema], ...]


class IIT4SIASchema(msgspec.Struct, frozen=True, tag="iit4_sia"):
    phi: PhiSchema
    partition: PartitionSchema
    normalized_phi: PhiSchema
    cause: RIASchema | None
    effect: RIASchema | None
    system_state: SystemStateSpecificationSchema | None
    current_state: tuple[int, ...] | None
    node_indices: tuple[int, ...] | None
    node_labels: NodeLabelsSchema | None
    intrinsic_differentiation: DirectionPhiPairs | None
    reasons: tuple[str, ...] | None
    signed_phi: PhiSchema | None
    signed_normalized_phi: PhiSchema | None
    config: dict[str, Any] | None
    provenance: ProvenanceSchema | None
    tie_peers: tuple["IIT4SIASchema", ...] = ()


class NullIIT4SIASchema(IIT4SIASchema, frozen=True, tag="null_iit4_sia"):
    pass


SIASchema = IIT3SIASchema | IIT4SIASchema | NullIIT4SIASchema


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
    | RIASchema
    | MICESchema
    | MICECauseSchema
    | MICEEffectSchema
    | DistinctionSchema
    | DistinctionsSchema
    | UnresolvedDistinctionsSchema
    | ResolvedDistinctionsSchema
    | ProvenanceSchema
    | ExcludedCandidateSchema
    | IIT3SIASchema
    | IIT4SIASchema
    | NullIIT4SIASchema
)
