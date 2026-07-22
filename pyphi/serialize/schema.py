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


class DistanceResultSchema(msgspec.Struct, frozen=True, tag="distance_result"):
    value: float
    aux: dict[str, Any] = msgspec.field(default_factory=dict)


# A φ value is either a native float or a distance result with auxiliary data;
# the Struct tag distinguishes the latter, so a bare number decodes as a float.
PhiSchema = float | DistanceResultSchema


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
    runner_up_state: tuple[int, ...] | None = None
    runner_up_intrinsic_information: PhiSchema | None = None


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
    partition_margin: PhiSchema | None = None
    signed_phi: PhiSchema | None = None
    selectivity: float | None = None
    reasons: tuple[str, ...] | None = None


class MICESchema(msgspec.Struct, frozen=True, tag="mice"):
    ria: RIASchema
    purview_margin: PhiSchema | None = None
    purview_tie_peers: tuple["MICEAnySchema", ...] | None = None


class MICECauseSchema(msgspec.Struct, frozen=True, tag="mice_cause"):
    ria: RIASchema
    purview_margin: PhiSchema | None = None
    purview_tie_peers: tuple["MICEAnySchema", ...] | None = None


class MICEEffectSchema(msgspec.Struct, frozen=True, tag="mice_effect"):
    ria: RIASchema
    purview_margin: PhiSchema | None = None
    purview_tie_peers: tuple["MICEAnySchema", ...] | None = None


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
    estimator: dict | None = None


class MacroUnitSchema(msgspec.Struct, frozen=True, tag="macro_unit"):
    constituents: tuple["MacroUnitSchema | int", ...]
    update_grain: int
    mapping: tuple[int, ...]
    background_apportionment: tuple[int, ...] = ()


class ExcludedCandidateSchema(msgspec.Struct, frozen=True, tag="excluded_candidate"):
    node_indices: tuple[int, ...]
    phi: float
    units: tuple[MacroUnitSchema, ...] | None = None


class RunnerUpSchema(msgspec.Struct, frozen=True, tag="runner_up"):
    partition: PartitionSchema
    phi: PhiSchema


class IIT3SIASchema(msgspec.Struct, frozen=True, tag="iit3_sia"):
    phi: PhiSchema | None
    distinctions: DistinctionsAnySchema | None
    partitioned_distinctions: DistinctionsAnySchema | None
    partition: PartitionSchema | None
    node_indices: tuple[int, ...] | None
    node_labels: NodeLabelsSchema | None
    current_state: tuple[int, ...] | None
    tie_peers: tuple["IIT3SIASchema", ...] = ()
    runner_up: RunnerUpSchema | None = None
    reasons: tuple[str, ...] | None = None
    config: dict[str, Any] | None = None
    provenance: ProvenanceSchema | None = None


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
    partition_margin: PhiSchema | None = None
    runner_up: RunnerUpSchema | None = None


class NullIIT4SIASchema(IIT4SIASchema, frozen=True, tag="null_iit4_sia"):
    pass


SIASchema = IIT3SIASchema | IIT4SIASchema | NullIIT4SIASchema


# --- Relations (standalone) ---------------------------------------------------

# A standalone relation embeds its member distinctions in full. Inside a
# CauseEffectStructure the distinctions are stored once and relations reference
# them by index (see the normalized CES schema below).


class RelationSchema(msgspec.Struct, frozen=True, tag="relation"):
    distinctions: tuple[DistinctionSchema, ...]


class ConcreteRelationsSchema(msgspec.Struct, frozen=True, tag="concrete_relations"):
    relations: tuple[RelationSchema, ...]


class NullRelationsSchema(msgspec.Struct, frozen=True, tag="null_relations"):
    pass


class AnalyticalRelationsSchema(msgspec.Struct, frozen=True, tag="analytical_relations"):
    distinctions: DistinctionsAnySchema


RelationsSchema = (
    ConcreteRelationsSchema | NullRelationsSchema | AnalyticalRelationsSchema
)


# --- Normalized cause-effect structure ----------------------------------------

# Within a CES the distinctions are stored once in a table; each relation
# references its members by their index into that table, removing the dominant
# redundancy of embedding every distinction in every relation.


class RelationRefSchema(msgspec.Struct, frozen=True, tag="relation_ref"):
    distinction_indices: tuple[int, ...]


class ConcreteRelationsRefSchema(
    msgspec.Struct, frozen=True, tag="concrete_relations_ref"
):
    relations: tuple[RelationRefSchema, ...]


class NullRelationsRefSchema(msgspec.Struct, frozen=True, tag="null_relations_ref"):
    pass


class AnalyticalRelationsRefSchema(
    msgspec.Struct, frozen=True, tag="analytical_relations_ref"
):
    pass


RelationsRefSchema = (
    ConcreteRelationsRefSchema | NullRelationsRefSchema | AnalyticalRelationsRefSchema
)


class CESSchema(msgspec.Struct, frozen=True, tag="ces"):
    sia: SIASchema
    distinctions: DistinctionsAnySchema
    relations: RelationsRefSchema
    config: dict[str, Any] | None = None
    provenance: ProvenanceSchema | None = None


class NullCESSchema(CESSchema, frozen=True, tag="null_ces"):
    pass


# --- Substrate, system, transition --------------------------------------------


class SubstrateSchema(msgspec.Struct, frozen=True, tag="substrate"):
    """Alphabet-general substrate encoding.

    One conditional factor array per node plus the per-node state space, so
    substrates with any alphabet sizes round-trip.
    """

    factors: tuple[bytes, ...]
    state_space: tuple[tuple[int | str, ...], ...]
    cm: bytes
    node_labels: NodeLabelsSchema | None


class SystemSchema(msgspec.Struct, frozen=True, tag="system"):
    substrate: SubstrateSchema
    state: tuple[int, ...]
    node_indices: tuple[int, ...]
    partition: PartitionSchema
    external_indices: tuple[int, ...]
    background_conditioning: str | None = None


class TransitionSchema(msgspec.Struct, frozen=True, tag="transition"):
    substrate: SubstrateSchema
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    cause_indices: tuple[int, ...]
    effect_indices: tuple[int, ...]
    partition: PartitionSchema
    noise_background: bool = False


# --- Actual causation ---------------------------------------------------------


class AcRIASchema(msgspec.Struct, frozen=True, tag="ac_ria"):
    alpha: float
    state: tuple[int, ...]
    direction: DirectionSchema
    mechanism: tuple[int, ...]
    purview: tuple[int, ...]
    partition: PartitionSchema
    probability: float
    partitioned_probability: float
    partition_tie_peers: tuple["AcRIASchema", ...] = ()
    node_labels: NodeLabelsSchema | None = None
    reasons: tuple[str, ...] | None = None


class CausalLinkSchema(msgspec.Struct, frozen=True, tag="causal_link"):
    ria: AcRIASchema
    extended_purview: tuple[tuple[int, ...], ...] | None
    purview_tie_peers: tuple[AcRIASchema, ...] = ()


class AccountSchema(msgspec.Struct, frozen=True, tag="account"):
    causal_links: tuple[CausalLinkSchema, ...]


class DirectedAccountSchema(msgspec.Struct, frozen=True, tag="directed_account"):
    causal_links: tuple[CausalLinkSchema, ...]


AccountAnySchema = AccountSchema | DirectedAccountSchema


class AcSIASchema(msgspec.Struct, frozen=True, tag="ac_sia"):
    alpha: float | None
    direction: DirectionSchema | None
    account: AccountAnySchema | None
    partitioned_account: AccountAnySchema | None
    partition: PartitionSchema | None
    before_state: tuple[int, ...] | None
    after_state: tuple[int, ...] | None
    size: int | None
    node_indices: tuple[int, ...] | None
    cause_indices: tuple[int, ...] | None
    effect_indices: tuple[int, ...] | None
    node_labels: NodeLabelsSchema | None
    reasons: tuple[str, ...] | None = None
    config: dict[str, Any] | None = None
    provenance: ProvenanceSchema | None = None
    tie_peers: tuple["AcSIASchema", ...] = ()


# --- Complex (embeds a substrate, hence after the substrate schema) -----------


class ComplexSchema(msgspec.Struct, frozen=True, tag="complex"):
    sia: SIASchema
    substrate: SubstrateSchema
    is_maximal: bool
    excluded: tuple[ExcludedCandidateSchema, ...]
    units: tuple[MacroUnitSchema, ...] | None = None
    node_indices: tuple[int, ...] | None = None


# --- Estimation-layer posteriors ----------------------------------------------


class CoverageReportSchema(msgspec.Struct, frozen=True, tag="coverage_report"):
    counts: bytes
    n_units: int


class SubstratePosteriorSchema(msgspec.Struct, frozen=True, tag="substrate_posterior"):
    alpha_on: bytes
    alpha_off: bytes
    regime: str
    prior: float
    coverage: CoverageReportSchema
    node_labels: tuple[str, ...] | None
    provenance: ProvenanceSchema


class PhiPosteriorSchema(msgspec.Struct, frozen=True, tag="phi_posterior"):
    samples: bytes
    complex_samples: tuple[tuple[int, ...], ...]
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    seed: int
    regime: str
    coverage: CoverageReportSchema
    provenance: ProvenanceSchema
    screen_margin: float | None = None
    screened: bool = False
    reference_margins: dict[str, float | None] | None = None


# --- Batch-run results --------------------------------------------------------


class DataFrameSchema(msgspec.Struct, frozen=True, tag="dataframe"):
    """A pandas DataFrame as embedded parquet.

    ``index_columns`` names the index levels reset to columns before the
    parquet write; ``tuple_columns`` names the object columns whose non-null
    cells are restored as tuples on decode (parquet represents them as
    lists).
    """

    parquet: bytes
    index_columns: tuple[str, ...] = ()
    tuple_columns: tuple[str, ...] = ()


class SweepResultSchema(msgspec.Struct, frozen=True, tag="sweep_result"):
    df: DataFrameSchema
    results: tuple["Schema | float", ...]
    skipped: tuple[tuple["str | int", str, tuple[int, ...], tuple[int, ...]], ...]


class CampaignTaskSchema(msgspec.Struct, frozen=True, tag="campaign_task"):
    task_id: int
    kind: str
    compute: "str | None"
    compute_ref: "str | None"
    config_overrides: dict[str, Any]
    cells: tuple[tuple["str | int", str, tuple[int, ...], tuple[int, ...]], ...]
    skip_uncomputable: bool


class CellOutputSchema(msgspec.Struct, frozen=True, tag="campaign_cell_output"):
    status: str
    result: "Schema | None"
    traceback: "str | None"
    aux: dict[str, Any] | None = None


class CampaignTaskOutputSchema(msgspec.Struct, frozen=True, tag="campaign_task_output"):
    task_id: int
    pyphi_version: str
    entries: tuple[CellOutputSchema, ...]


class AxisScopeSchema(msgspec.Struct, frozen=True, tag="axis_scope"):
    explicit: tuple[tuple[int, ...], ...] | None
    min_order: int | None
    max_order: int | None
    containing: tuple[int, ...] | None
    within: tuple[int, ...] | None


class CESScopeSchema(msgspec.Struct, frozen=True, tag="ces_scope"):
    mechanisms: AxisScopeSchema
    cause_purviews: AxisScopeSchema
    effect_purviews: AxisScopeSchema
    max_purview_order_by_mechanism_order: tuple[tuple[int, int], ...] | None = None


class ShardSpecSchema(msgspec.Struct, frozen=True, tag="shard_spec"):
    payload_kind: str
    mechanisms: tuple[tuple[int, ...], ...]
    mechanism: tuple[int, ...] | None
    direction: "str | None"
    purviews: tuple[tuple[int, ...], ...]
    purview: tuple[int, ...] | None
    stride: tuple[int, int] | None
    units: float
    memory_bytes: int = 0


class CESShardTaskSchema(msgspec.Struct, frozen=True, tag="campaign_ces_task"):
    task_id: int
    kind: str
    substrate_label: "str | int"
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    scope: CESScopeSchema
    config_overrides: dict[str, Any]
    formalism: str
    spec: ShardSpecSchema
    ordering: "str | None"


class SIAShardTaskSchema(msgspec.Struct, frozen=True, tag="campaign_sia_task"):
    task_id: int
    kind: str
    substrate_label: "str | int"
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    config_overrides: dict[str, Any]
    formalism: str
    stride: tuple[int, int]


class OptimizationResultSchema(msgspec.Struct, frozen=True, tag="optimization_result"):
    """An :func:`~pyphi.optimize.optimize` outcome.

    ``best_objective`` is stored as None exactly when the run had no
    reachable candidate (NaN on the domain object; JSON cannot carry NaN).
    """

    best_params: bytes
    best_objective: float | None
    best_substrate: SubstrateSchema
    best_sia: SIASchema | None
    trajectory: DataFrameSchema
    bounds: tuple[tuple[float, float], ...]
    seed: int
    direction: str
    objective_name: str
    settings: dict[str, Any]
    config_snapshot: dict[str, Any]
    n_evaluations: int
    n_unreachable: int


# The tagged union grows one member per serializable type.
Schema = (
    DirectionSchema
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
    | RunnerUpSchema
    | ExcludedCandidateSchema
    | MacroUnitSchema
    | IIT3SIASchema
    | IIT4SIASchema
    | NullIIT4SIASchema
    | RelationSchema
    | ConcreteRelationsSchema
    | NullRelationsSchema
    | AnalyticalRelationsSchema
    | CESSchema
    | NullCESSchema
    | SubstrateSchema
    | SystemSchema
    | TransitionSchema
    | AcRIASchema
    | CausalLinkSchema
    | AccountSchema
    | DirectedAccountSchema
    | AcSIASchema
    | ComplexSchema
    | CoverageReportSchema
    | SubstratePosteriorSchema
    | PhiPosteriorSchema
    | SweepResultSchema
    | OptimizationResultSchema
    | CampaignTaskSchema
    | CellOutputSchema
    | CampaignTaskOutputSchema
    | AxisScopeSchema
    | CESScopeSchema
    | ShardSpecSchema
    | CESShardTaskSchema
    | SIAShardTaskSchema
)
