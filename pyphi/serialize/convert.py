"""Convert between PyPhi domain objects and their msgspec schema Structs.

Two registries map a domain type to its encoder and a schema Struct type to its
decoder. This replaces the per-class ``to_json`` / ``from_json`` methods. Each
serializable type adds one ``_register_<type>()`` populating both registries,
all called at import time.
"""

from collections.abc import Callable
from typing import Any

import msgspec
import numpy as np

from pyphi.direction import Direction

from . import arrays
from . import schema

_ENCODERS: dict[type, Callable[[Any], Any]] = {}  # domain type   -> encode
_DECODERS: dict[type, Callable[[Any], Any]] = {}  # schema Struct  -> decode


def to_schema(obj: Any) -> Any:
    _ensure_registered()
    encode = _ENCODERS.get(type(obj))
    if encode is None:
        raise TypeError(f"No serializer registered for {type(obj).__name__}")
    return encode(obj)


def from_schema(struct: Any) -> Any:
    _ensure_registered()
    decode = _DECODERS.get(type(struct))
    if decode is None:
        raise TypeError(f"No deserializer registered for {type(struct).__name__}")
    return decode(struct)


def _enc_optional(obj: Any) -> Any:
    """Encode a nested domain object that may be ``None``."""
    return to_schema(obj) if obj is not None else None


def _dec_optional(struct: Any) -> Any:
    """Decode a nested schema struct that may be ``None``."""
    return from_schema(struct) if struct is not None else None


def _register_direction() -> None:
    _ENCODERS[Direction] = lambda d: schema.DirectionSchema(name=d.name)
    _DECODERS[schema.DirectionSchema] = lambda s: Direction[s.name]


def _register_pyphi_float() -> None:
    from pyphi.data_structures import PyPhiFloat

    _ENCODERS[PyPhiFloat] = lambda f: schema.PyPhiFloatSchema(value=float(f))
    _DECODERS[schema.PyPhiFloatSchema] = lambda s: PyPhiFloat(s.value)


def _register_distance_result() -> None:
    from pyphi.measures.distribution import DistanceResult

    _ENCODERS[DistanceResult] = lambda r: schema.DistanceResultSchema(
        value=float(r), aux=r._public_aux_data()
    )
    _DECODERS[schema.DistanceResultSchema] = lambda s: DistanceResult(s.value, **s.aux)


def _register_node_labels() -> None:
    from pyphi.labels import NodeLabels

    _ENCODERS[NodeLabels] = lambda n: schema.NodeLabelsSchema(
        labels=tuple(n.labels), node_indices=tuple(n.node_indices)
    )
    _DECODERS[schema.NodeLabelsSchema] = lambda s: NodeLabels(s.labels, s.node_indices)


def _encode_state_spec(spec: Any, *, include_peers: bool) -> Any:
    peers = tuple(t for t in spec.ties if t is not spec) if include_peers else ()
    return schema.StateSpecificationSchema(
        direction=schema.DirectionSchema(name=spec.direction.name),
        purview=tuple(spec.purview),
        state=tuple(spec.state),
        intrinsic_information=to_schema(spec.intrinsic_information),
        repertoire=arrays.array_to_bytes(np.asarray(spec.repertoire)),
        unconstrained_repertoire=arrays.array_to_bytes(
            np.asarray(spec.unconstrained_repertoire)
        ),
        tie_peers=tuple(_encode_state_spec(p, include_peers=False) for p in peers),
    )


def _decode_state_spec(struct: Any) -> Any:
    from pyphi.models.state_specification import StateSpecification

    instance = StateSpecification(
        direction=from_schema(struct.direction),
        purview=tuple(struct.purview),
        state=tuple(struct.state),
        intrinsic_information=from_schema(struct.intrinsic_information),
        repertoire=arrays.bytes_to_array(struct.repertoire),
        unconstrained_repertoire=arrays.bytes_to_array(struct.unconstrained_repertoire),
    )
    if struct.tie_peers:
        peers = tuple(_decode_state_spec(p) for p in struct.tie_peers)
        tied = (instance, *peers)
        instance.set_ties(tied)
        for peer in peers:
            peer.set_ties(tied)
    return instance


def _register_state_specification() -> None:
    from pyphi.models.state_specification import StateSpecification

    _ENCODERS[StateSpecification] = lambda s: _encode_state_spec(s, include_peers=True)
    _DECODERS[schema.StateSpecificationSchema] = _decode_state_spec


def _register_system_state_specification() -> None:
    from pyphi.models.state_specification import SystemStateSpecification

    _ENCODERS[SystemStateSpecification] = (
        lambda s: schema.SystemStateSpecificationSchema(
            cause=to_schema(s.cause),
            effect=to_schema(s.effect),
        )
    )
    _DECODERS[schema.SystemStateSpecificationSchema] = (
        lambda s: SystemStateSpecification(
            cause=from_schema(s.cause),
            effect=from_schema(s.effect),
        )
    )


def _register_part() -> None:
    from pyphi.models.partitions import Part

    _ENCODERS[Part] = lambda p: schema.PartSchema(
        mechanism=tuple(p.mechanism), purview=tuple(p.purview)
    )
    _DECODERS[schema.PartSchema] = lambda s: Part(tuple(s.mechanism), tuple(s.purview))


def _register_null_cut() -> None:
    from pyphi.models.partitions import NullCut

    _ENCODERS[NullCut] = lambda c: schema.NullCutSchema(indices=tuple(c.indices))
    _DECODERS[schema.NullCutSchema] = lambda s: NullCut(tuple(s.indices))


def _register_directed_bipartition() -> None:
    from pyphi.models.partitions import DirectedBipartition

    _ENCODERS[DirectedBipartition] = lambda p: schema.DirectedBipartitionSchema(
        direction=schema.DirectionSchema(name=p.direction.name),
        from_nodes=tuple(p.from_nodes),
        to_nodes=tuple(p.to_nodes),
    )
    _DECODERS[schema.DirectedBipartitionSchema] = lambda s: DirectedBipartition(
        from_schema(s.direction), tuple(s.from_nodes), tuple(s.to_nodes)
    )


def _register_joint_partition() -> None:
    from pyphi.models.partitions import JointPartition

    _ENCODERS[JointPartition] = lambda p: schema.JointPartitionSchema(
        parts=tuple(to_schema(part) for part in p.parts)
    )
    _DECODERS[schema.JointPartitionSchema] = lambda s: JointPartition(
        *(from_schema(p) for p in s.parts)
    )


def _register_joint_bipartition() -> None:
    from pyphi.models.partitions import JointBipartition

    _ENCODERS[JointBipartition] = lambda p: schema.JointBipartitionSchema(
        part0=to_schema(p[0]), part1=to_schema(p[1])
    )
    _DECODERS[schema.JointBipartitionSchema] = lambda s: JointBipartition(
        from_schema(s.part0), from_schema(s.part1)
    )


def _register_joint_tripartition() -> None:
    from pyphi.models.partitions import JointTripartition

    _ENCODERS[JointTripartition] = lambda p: schema.JointTripartitionSchema(
        parts=tuple(to_schema(part) for part in p.parts)
    )
    _DECODERS[schema.JointTripartitionSchema] = lambda s: JointTripartition(
        *(from_schema(p) for p in s.parts)
    )


def _register_directed_joint_partition() -> None:
    from pyphi.models.partitions import DirectedJointPartition

    _ENCODERS[DirectedJointPartition] = lambda p: schema.DirectedJointPartitionSchema(
        direction=schema.DirectionSchema(name=p.direction.name),
        partition=to_schema(p.partition),
    )
    _DECODERS[schema.DirectedJointPartitionSchema] = lambda s: DirectedJointPartition(
        from_schema(s.direction), from_schema(s.partition)
    )


def _register_edge_cut() -> None:
    from pyphi.models.partitions import EdgeCut

    _ENCODERS[EdgeCut] = lambda c: schema.EdgeCutSchema(
        node_indices=tuple(c.node_indices),
        cut_matrix=arrays.array_to_bytes(np.asarray(c._cut_matrix)),
        node_labels=_enc_optional(c.node_labels),
    )
    _DECODERS[schema.EdgeCutSchema] = lambda s: EdgeCut(
        tuple(s.node_indices),
        arrays.bytes_to_array(s.cut_matrix),
        _dec_optional(s.node_labels),
    )


def _register_complete_edge_cut() -> None:
    from pyphi.models.partitions import CompleteEdgeCut

    _ENCODERS[CompleteEdgeCut] = lambda c: schema.CompleteEdgeCutSchema(
        node_indices=tuple(c.node_indices),
        node_labels=_enc_optional(c.node_labels),
    )
    _DECODERS[schema.CompleteEdgeCutSchema] = lambda s: CompleteEdgeCut(
        tuple(s.node_indices), _dec_optional(s.node_labels)
    )


def _register_directed_set_partition() -> None:
    from pyphi.models.partitions import DirectedSetPartition

    _ENCODERS[DirectedSetPartition] = lambda c: schema.DirectedSetPartitionSchema(
        node_indices=tuple(c.node_indices),
        cut_matrix=arrays.array_to_bytes(np.asarray(c._cut_matrix)),
        set_partition=tuple(tuple(part) for part in c.set_partition),
        node_labels=_enc_optional(c.node_labels),
    )
    _DECODERS[schema.DirectedSetPartitionSchema] = lambda s: DirectedSetPartition(
        node_indices=tuple(s.node_indices),
        cut_matrix=arrays.bytes_to_array(s.cut_matrix),
        set_partition=[list(part) for part in s.set_partition],
        node_labels=_dec_optional(s.node_labels),
    )


def _enc_array(arr: Any) -> Any:
    """Encode an optional numpy array to ``.npy`` bytes (``None`` stays ``None``)."""
    return arrays.array_to_bytes(np.asarray(arr)) if arr is not None else None


def _dec_array(data: Any) -> Any:
    """Decode optional ``.npy`` bytes to a numpy array (``None`` stays ``None``)."""
    return arrays.bytes_to_array(data) if data is not None else None


def _opt_tuple(values: Any) -> Any:
    return tuple(values) if values is not None else None


def _encode_ria(ria: Any, *, include_peers: bool) -> Any:
    partition_peers = (
        tuple(t for t in ria._partition_ties if t is not ria) if include_peers else ()
    )
    state_peers = (
        tuple(t for t in ria._state_ties if t is not ria) if include_peers else ()
    )
    return schema.RIASchema(
        phi=to_schema(ria.phi),
        direction=schema.DirectionSchema(name=ria.direction.name),
        mechanism=tuple(ria.mechanism),
        mechanism_state=_opt_tuple(ria.mechanism_state),
        purview=tuple(ria.purview),
        purview_state=_opt_tuple(ria.purview_state),
        partition=to_schema(ria.partition),
        repertoire=_enc_array(ria.repertoire),
        partitioned_repertoire=_enc_array(ria.partitioned_repertoire),
        specified_state=_enc_optional(ria.specified_state),
        node_labels=_enc_optional(ria.node_labels),
        partition_tie_peers=tuple(
            _encode_ria(p, include_peers=False) for p in partition_peers
        ),
        state_tie_peers=tuple(_encode_ria(p, include_peers=False) for p in state_peers),
    )


def _decode_ria(struct: Any) -> Any:
    from pyphi.models.ria import RepertoireIrreducibilityAnalysis

    instance = RepertoireIrreducibilityAnalysis(
        phi=from_schema(struct.phi),
        direction=from_schema(struct.direction),
        mechanism=tuple(struct.mechanism),
        purview=tuple(struct.purview),
        partition=from_schema(struct.partition),
        repertoire=_dec_array(struct.repertoire),
        partitioned_repertoire=_dec_array(struct.partitioned_repertoire),
        specified_state=_dec_optional(struct.specified_state),
        mechanism_state=_opt_tuple(struct.mechanism_state),
        purview_state=_opt_tuple(struct.purview_state),
        node_labels=_dec_optional(struct.node_labels),
    )
    if struct.partition_tie_peers:
        peers = tuple(_decode_ria(p) for p in struct.partition_tie_peers)
        tied = (instance, *peers)
        instance._partition_ties = tied
        for peer in peers:
            peer._partition_ties = tied
    if struct.state_tie_peers:
        peers = tuple(_decode_ria(p) for p in struct.state_tie_peers)
        tied = (instance, *peers)
        instance._state_ties = tied
        for peer in peers:
            peer._state_ties = tied
    return instance


def _register_ria() -> None:
    from pyphi.models.ria import RepertoireIrreducibilityAnalysis

    _ENCODERS[RepertoireIrreducibilityAnalysis] = lambda r: _encode_ria(
        r, include_peers=True
    )
    _DECODERS[schema.RIASchema] = _decode_ria


def _decode_mice(cls: type, struct: Any) -> Any:
    instance = cls(from_schema(struct.ria))
    instance._purview_ties = (instance,)
    return instance


def _register_mice() -> None:
    from pyphi.models.mice import MaximallyIrreducibleCause
    from pyphi.models.mice import MaximallyIrreducibleCauseOrEffect
    from pyphi.models.mice import MaximallyIrreducibleEffect

    _ENCODERS[MaximallyIrreducibleCauseOrEffect] = lambda m: schema.MICESchema(
        ria=to_schema(m.ria)
    )
    _ENCODERS[MaximallyIrreducibleCause] = lambda m: schema.MICECauseSchema(
        ria=to_schema(m.ria)
    )
    _ENCODERS[MaximallyIrreducibleEffect] = lambda m: schema.MICEEffectSchema(
        ria=to_schema(m.ria)
    )
    _DECODERS[schema.MICESchema] = lambda s: _decode_mice(
        MaximallyIrreducibleCauseOrEffect, s
    )
    _DECODERS[schema.MICECauseSchema] = lambda s: _decode_mice(
        MaximallyIrreducibleCause, s
    )
    _DECODERS[schema.MICEEffectSchema] = lambda s: _decode_mice(
        MaximallyIrreducibleEffect, s
    )


def _register_distinction() -> None:
    from pyphi.models.distinction import Distinction

    _ENCODERS[Distinction] = lambda d: schema.DistinctionSchema(
        mechanism=_opt_tuple(d.mechanism),
        cause=to_schema(d.cause),
        effect=to_schema(d.effect),
    )
    _DECODERS[schema.DistinctionSchema] = lambda s: Distinction(
        mechanism=_opt_tuple(s.mechanism),
        cause=from_schema(s.cause),
        effect=from_schema(s.effect),
    )


def _register_distinctions() -> None:
    from pyphi.models.distinctions import Distinctions
    from pyphi.models.distinctions import ResolvedDistinctions
    from pyphi.models.distinctions import UnresolvedDistinctions

    def encoder(struct_cls):
        return lambda d: struct_cls(concepts=tuple(to_schema(c) for c in d.concepts))

    def decoder(domain_cls):
        return lambda s: domain_cls(tuple(from_schema(c) for c in s.concepts))

    _ENCODERS[Distinctions] = encoder(schema.DistinctionsSchema)
    _ENCODERS[UnresolvedDistinctions] = encoder(schema.UnresolvedDistinctionsSchema)
    _ENCODERS[ResolvedDistinctions] = encoder(schema.ResolvedDistinctionsSchema)
    _DECODERS[schema.DistinctionsSchema] = decoder(Distinctions)
    _DECODERS[schema.UnresolvedDistinctionsSchema] = decoder(UnresolvedDistinctions)
    _DECODERS[schema.ResolvedDistinctionsSchema] = decoder(ResolvedDistinctions)


def _register_provenance() -> None:
    from pyphi.provenance import Provenance

    _ENCODERS[Provenance] = lambda p: schema.ProvenanceSchema(
        pyphi_version=p.pyphi_version,
        git_sha=p.git_sha,
        git_dirty=p.git_dirty,
        timestamp=p.timestamp,
        python_version=p.python_version,
        numpy_version=p.numpy_version,
        scipy_version=p.scipy_version,
        platform=p.platform,
        wall_time=p.wall_time,
        seed=p.seed,
        note=p.note,
    )
    _DECODERS[schema.ProvenanceSchema] = lambda s: Provenance(
        **msgspec.structs.asdict(s)
    )


def _register_excluded_candidate() -> None:
    from pyphi.models.complex import ExcludedCandidate

    _ENCODERS[ExcludedCandidate] = lambda e: schema.ExcludedCandidateSchema(
        node_indices=tuple(e.node_indices), phi=float(e.phi)
    )
    _DECODERS[schema.ExcludedCandidateSchema] = lambda s: ExcludedCandidate(
        s.node_indices, s.phi
    )


def _encode_iit3_sia(sia: Any, *, include_peers: bool) -> Any:
    peers = tuple(t for t in sia.ties if t is not sia) if include_peers else ()
    return schema.IIT3SIASchema(
        phi=_enc_optional(sia.phi),
        distinctions=_enc_optional(sia.distinctions),
        partitioned_distinctions=_enc_optional(sia.partitioned_distinctions),
        partition=_enc_optional(sia.partition),
        node_indices=_opt_tuple(sia.node_indices),
        node_labels=_enc_optional(sia.node_labels),
        current_state=_opt_tuple(sia.current_state),
        tie_peers=tuple(_encode_iit3_sia(p, include_peers=False) for p in peers),
    )


def _decode_iit3_sia(struct: Any) -> Any:
    from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis

    instance = IIT3SystemIrreducibilityAnalysis(
        phi=_dec_optional(struct.phi),
        distinctions=_dec_optional(struct.distinctions),
        partitioned_distinctions=_dec_optional(struct.partitioned_distinctions),
        partition=_dec_optional(struct.partition),
        node_indices=_opt_tuple(struct.node_indices),
        node_labels=_dec_optional(struct.node_labels),
        current_state=_opt_tuple(struct.current_state),
    )
    if struct.tie_peers:
        peers = tuple(_decode_iit3_sia(p) for p in struct.tie_peers)
        tied = [instance, *peers]
        instance._ties = tied
        for peer in peers:
            peer._ties = tied
    return instance


def _register_iit3_sia() -> None:
    from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis

    _ENCODERS[IIT3SystemIrreducibilityAnalysis] = lambda s: _encode_iit3_sia(
        s, include_peers=True
    )
    _DECODERS[schema.IIT3SIASchema] = _decode_iit3_sia


def _enc_intrinsic_diff(diff: Any) -> Any:
    if diff is None:
        return None
    return tuple(
        (schema.DirectionSchema(name=k.name), to_schema(v)) for k, v in diff.items()
    )


def _dec_intrinsic_diff(pairs: Any) -> Any:
    if pairs is None:
        return None
    return {from_schema(k): from_schema(v) for k, v in pairs}


def _enc_reasons(reasons: Any) -> Any:
    # A reason is normally a NullResultReason enum, but some fixtures carry it
    # as a bare name string; store the name either way.
    if reasons is None:
        return None
    return tuple(r.name if hasattr(r, "name") else str(r) for r in reasons)


def _dec_reasons(names: Any) -> Any:
    if names is None:
        return None
    from pyphi.models.explanation import NullResultReason

    return [
        NullResultReason[n] if n in NullResultReason.__members__ else n for n in names
    ]


def _enc_config(config: Any) -> Any:
    if config is None:
        return None
    # ConfigSnapshot is a nested frozen-dataclass tree; encode to plain builtins
    # (config-as-Struct is out of scope, and decode keeps the dict form, which
    # matches the prior serializer's behaviour).
    return msgspec.to_builtins(config, enc_hook=str)


def _iit4_sia_struct_cls(sia: Any) -> Any:
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis

    if isinstance(sia, NullSystemIrreducibilityAnalysis):
        return schema.NullIIT4SIASchema
    return schema.IIT4SIASchema


def _encode_iit4_sia(sia: Any, *, include_peers: bool) -> Any:
    peers = tuple(t for t in sia.ties if t is not sia) if include_peers else ()
    struct_cls = _iit4_sia_struct_cls(sia)
    return struct_cls(
        phi=to_schema(sia.phi),
        partition=to_schema(sia.partition),
        normalized_phi=to_schema(sia.normalized_phi),
        cause=_enc_optional(sia.cause),
        effect=_enc_optional(sia.effect),
        system_state=_enc_optional(sia.system_state),
        current_state=_opt_tuple(sia.current_state),
        node_indices=_opt_tuple(sia.node_indices),
        node_labels=_enc_optional(sia.node_labels),
        intrinsic_differentiation=_enc_intrinsic_diff(sia.intrinsic_differentiation),
        reasons=_enc_reasons(sia.reasons),
        signed_phi=_enc_optional(sia.signed_phi),
        signed_normalized_phi=_enc_optional(sia.signed_normalized_phi),
        config=_enc_config(sia.config),
        provenance=_enc_optional(sia.provenance),
        tie_peers=tuple(_encode_iit4_sia(p, include_peers=False) for p in peers),
    )


def _decode_iit4_sia(struct: Any) -> Any:
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
    from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis

    kwargs = {
        "phi": from_schema(struct.phi),
        "partition": from_schema(struct.partition),
        "normalized_phi": from_schema(struct.normalized_phi),
        "cause": _dec_optional(struct.cause),
        "effect": _dec_optional(struct.effect),
        "system_state": _dec_optional(struct.system_state),
        "current_state": _opt_tuple(struct.current_state),
        "node_indices": _opt_tuple(struct.node_indices),
        "node_labels": _dec_optional(struct.node_labels),
        "intrinsic_differentiation": _dec_intrinsic_diff(
            struct.intrinsic_differentiation
        ),
        "reasons": _dec_reasons(struct.reasons),
        "signed_phi": _dec_optional(struct.signed_phi),
        "signed_normalized_phi": _dec_optional(struct.signed_normalized_phi),
        "config": struct.config,
        "provenance": _dec_optional(struct.provenance),
    }
    if type(struct) is schema.NullIIT4SIASchema:
        instance = object.__new__(NullSystemIrreducibilityAnalysis)
        SystemIrreducibilityAnalysis.__init__(instance, **kwargs)
    else:
        instance = SystemIrreducibilityAnalysis(**kwargs)
    if struct.tie_peers:
        peers = tuple(_decode_iit4_sia(p) for p in struct.tie_peers)
        tied = [instance, *peers]
        instance._ties = tied
        for peer in peers:
            peer._ties = tied
    return instance


def _register_iit4_sia() -> None:
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
    from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis

    _ENCODERS[SystemIrreducibilityAnalysis] = lambda s: _encode_iit4_sia(
        s, include_peers=True
    )
    _ENCODERS[NullSystemIrreducibilityAnalysis] = lambda s: _encode_iit4_sia(
        s, include_peers=True
    )
    _DECODERS[schema.IIT4SIASchema] = _decode_iit4_sia
    _DECODERS[schema.NullIIT4SIASchema] = _decode_iit4_sia


def _register_relation() -> None:
    from pyphi.relations import Relation

    _ENCODERS[Relation] = lambda r: schema.RelationSchema(
        distinctions=tuple(to_schema(d) for d in r)
    )
    _DECODERS[schema.RelationSchema] = lambda s: Relation(
        [from_schema(d) for d in s.distinctions]
    )


def _register_relations() -> None:
    from pyphi.relations import AnalyticalRelations
    from pyphi.relations import ConcreteRelations
    from pyphi.relations import NullRelations

    _ENCODERS[ConcreteRelations] = lambda rs: schema.ConcreteRelationsSchema(
        relations=tuple(to_schema(r) for r in rs)
    )
    _ENCODERS[NullRelations] = lambda _rs: schema.NullRelationsSchema()
    _ENCODERS[AnalyticalRelations] = lambda rs: schema.AnalyticalRelationsSchema(
        distinctions=to_schema(rs.distinctions)
    )
    _DECODERS[schema.ConcreteRelationsSchema] = lambda s: ConcreteRelations(
        [from_schema(r) for r in s.relations]
    )
    _DECODERS[schema.NullRelationsSchema] = lambda _s: NullRelations()
    _DECODERS[schema.AnalyticalRelationsSchema] = lambda s: AnalyticalRelations(
        from_schema(s.distinctions)
    )


def _relation_indices(relation: Any, table: list, by_id: dict) -> tuple[int, ...]:
    indices = []
    for distinction in relation:
        index = by_id.get(id(distinction))
        if index is None:
            # Fallback to value equality if the relation's distinction is not
            # the identity-shared instance from the CES table.
            index = next((j for j, d in enumerate(table) if d == distinction), None)
            if index is None:
                raise ValueError(
                    "relation references a distinction absent from the CES table"
                )
        indices.append(index)
    return tuple(sorted(indices))


def _encode_relations_ref(relations: Any, table: list, by_id: dict) -> Any:
    from pyphi.relations import AnalyticalRelations
    from pyphi.relations import ConcreteRelations
    from pyphi.relations import NullRelations

    if isinstance(relations, NullRelations):
        return schema.NullRelationsRefSchema()
    if isinstance(relations, AnalyticalRelations):
        return schema.AnalyticalRelationsRefSchema()
    if isinstance(relations, ConcreteRelations):
        refs = tuple(
            schema.RelationRefSchema(
                distinction_indices=_relation_indices(rel, table, by_id)
            )
            for rel in relations
        )
        return schema.ConcreteRelationsRefSchema(relations=refs)
    raise TypeError(f"Cannot normalize relations of type {type(relations).__name__}")


def _decode_relations_ref(struct: Any, table: list) -> Any:
    from pyphi.models.distinctions import ResolvedDistinctions
    from pyphi.relations import AnalyticalRelations
    from pyphi.relations import ConcreteRelations
    from pyphi.relations import NullRelations
    from pyphi.relations import Relation

    if type(struct) is schema.NullRelationsRefSchema:
        return NullRelations()
    if type(struct) is schema.AnalyticalRelationsRefSchema:
        return AnalyticalRelations(ResolvedDistinctions(table))
    relations = tuple(
        Relation([table[i] for i in ref.distinction_indices]) for ref in struct.relations
    )
    return ConcreteRelations(relations)


def _encode_ces(ces: Any, struct_cls: Any) -> Any:
    table = list(ces.distinctions)
    by_id = {id(d): i for i, d in enumerate(table)}
    return struct_cls(
        sia=to_schema(ces.sia),
        distinctions=to_schema(ces.distinctions),
        relations=_encode_relations_ref(ces.relations, table, by_id),
    )


def _decode_ces(struct: Any, domain_cls: Any) -> Any:
    distinctions = from_schema(struct.distinctions)
    table = list(distinctions)
    relations = _decode_relations_ref(struct.relations, table)
    return domain_cls(
        sia=from_schema(struct.sia),
        distinctions=distinctions,
        relations=relations,
    )


def _register_ces() -> None:
    from pyphi.formalism.iit4 import NullCauseEffectStructure
    from pyphi.models.ces import CauseEffectStructure

    _ENCODERS[CauseEffectStructure] = lambda c: _encode_ces(c, schema.CESSchema)
    _ENCODERS[NullCauseEffectStructure] = lambda c: _encode_ces(c, schema.NullCESSchema)
    _DECODERS[schema.CESSchema] = lambda s: _decode_ces(s, CauseEffectStructure)
    _DECODERS[schema.NullCESSchema] = lambda s: _decode_ces(s, NullCauseEffectStructure)


def _register_substrate() -> None:
    from pyphi.substrate import Substrate

    _ENCODERS[Substrate] = lambda s: schema.SubstrateSchema(
        tpm=arrays.array_to_bytes(np.asarray(s._legacy_binary_joint())),
        cm=arrays.array_to_bytes(np.asarray(s.cm)),
        node_labels=_enc_optional(s.node_labels),
    )
    _DECODERS[schema.SubstrateSchema] = lambda s: Substrate(
        tpm=arrays.bytes_to_array(s.tpm),
        cm=arrays.bytes_to_array(s.cm),
        node_labels=_dec_optional(s.node_labels),
    )


def _register_system() -> None:
    from pyphi.system import System

    _ENCODERS[System] = lambda s: schema.SystemSchema(
        substrate=to_schema(s.substrate),
        state=tuple(s.state),
        node_indices=tuple(s.node_indices),
        partition=to_schema(s.partition),
        external_indices=tuple(s.external_indices),
    )
    _DECODERS[schema.SystemSchema] = lambda s: System(
        substrate=from_schema(s.substrate),
        state=tuple(s.state),
        node_indices=tuple(s.node_indices),
        partition=from_schema(s.partition),
        external_indices=tuple(s.external_indices),
    )


def _register_transition() -> None:
    from pyphi.actual import Transition

    _ENCODERS[Transition] = lambda t: schema.TransitionSchema(
        substrate=to_schema(t.substrate),
        before_state=tuple(t.before_state),
        after_state=tuple(t.after_state),
        cause_indices=tuple(t.cause_indices),
        effect_indices=tuple(t.effect_indices),
        partition=to_schema(t.partition),
    )
    _DECODERS[schema.TransitionSchema] = lambda t: Transition(
        substrate=from_schema(t.substrate),
        before_state=tuple(t.before_state),
        after_state=tuple(t.after_state),
        cause_indices=tuple(t.cause_indices),
        effect_indices=tuple(t.effect_indices),
        partition=from_schema(t.partition),
    )


def _encode_ac_ria(ria: Any, *, include_peers: bool) -> Any:
    peers: tuple = ()
    if include_peers and ria._partition_ties is not None:
        peers = tuple(t for t in ria._partition_ties if t is not ria)
    return schema.AcRIASchema(
        alpha=float(ria.alpha),
        state=tuple(ria.state),
        direction=schema.DirectionSchema(name=ria.direction.name),
        mechanism=tuple(ria.mechanism),
        purview=tuple(ria.purview),
        partition=to_schema(ria.partition),
        probability=float(ria.probability),
        partitioned_probability=float(ria.partitioned_probability),
        partition_tie_peers=tuple(_encode_ac_ria(p, include_peers=False) for p in peers),
    )


def _decode_ac_ria(struct: Any) -> Any:
    from pyphi.models.actual_causation import AcRepertoireIrreducibilityAnalysis

    instance = AcRepertoireIrreducibilityAnalysis(
        alpha=struct.alpha,
        state=tuple(struct.state),
        direction=from_schema(struct.direction),
        mechanism=tuple(struct.mechanism),
        purview=tuple(struct.purview),
        partition=from_schema(struct.partition),
        probability=struct.probability,
        partitioned_probability=struct.partitioned_probability,
    )
    if struct.partition_tie_peers:
        peers = tuple(_decode_ac_ria(p) for p in struct.partition_tie_peers)
        tied = (instance, *peers)
        instance._partition_ties = tied
        for peer in peers:
            peer._partition_ties = tied
    return instance


def _register_ac_ria() -> None:
    from pyphi.models.actual_causation import AcRepertoireIrreducibilityAnalysis

    _ENCODERS[AcRepertoireIrreducibilityAnalysis] = lambda r: _encode_ac_ria(
        r, include_peers=True
    )
    _DECODERS[schema.AcRIASchema] = _decode_ac_ria


def _register_causal_link() -> None:
    from pyphi.models.actual_causation import CausalLink

    def encode(link):
        peers = link._purview_ties or ()
        extended = link._extended_purview
        return schema.CausalLinkSchema(
            ria=_encode_ac_ria(link.ria, include_peers=True),
            extended_purview=(
                None if extended is None else tuple(tuple(p) for p in extended)
            ),
            purview_tie_peers=tuple(
                _encode_ac_ria(p, include_peers=False) for p in peers
            ),
        )

    def decode(struct):
        peers = tuple(_decode_ac_ria(p) for p in struct.purview_tie_peers)
        extended = struct.extended_purview
        return CausalLink(
            ria=_decode_ac_ria(struct.ria),
            extended_purview=(
                None if extended is None else tuple(tuple(p) for p in extended)
            ),
            purview_ties=peers if peers else None,
        )

    _ENCODERS[CausalLink] = encode
    _DECODERS[schema.CausalLinkSchema] = decode


def _register_account() -> None:
    from pyphi.models.actual_causation import Account
    from pyphi.models.actual_causation import DirectedAccount

    _ENCODERS[Account] = lambda a: schema.AccountSchema(
        causal_links=tuple(to_schema(link) for link in a)
    )
    _ENCODERS[DirectedAccount] = lambda a: schema.DirectedAccountSchema(
        causal_links=tuple(to_schema(link) for link in a)
    )
    _DECODERS[schema.AccountSchema] = lambda s: Account(
        [from_schema(link) for link in s.causal_links]
    )
    _DECODERS[schema.DirectedAccountSchema] = lambda s: DirectedAccount(
        [from_schema(link) for link in s.causal_links]
    )


def _register_ac_sia() -> None:
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis

    _ENCODERS[AcSystemIrreducibilityAnalysis] = lambda s: schema.AcSIASchema(
        alpha=None if s.alpha is None else float(s.alpha),
        direction=_enc_optional_direction(s.direction),
        account=_enc_optional(s.account),
        partitioned_account=_enc_optional(s.partitioned_account),
        partition=_enc_optional(s.partition),
        before_state=_opt_tuple(s.before_state),
        after_state=_opt_tuple(s.after_state),
        size=s.size,
        node_indices=_opt_tuple(s.node_indices),
        cause_indices=_opt_tuple(s.cause_indices),
        effect_indices=_opt_tuple(s.effect_indices),
        node_labels=_enc_optional(s.node_labels),
    )
    _DECODERS[schema.AcSIASchema] = lambda s: AcSystemIrreducibilityAnalysis(
        alpha=s.alpha,
        direction=_dec_optional(s.direction),
        account=_dec_optional(s.account),
        partitioned_account=_dec_optional(s.partitioned_account),
        partition=_dec_optional(s.partition),
        before_state=_opt_tuple(s.before_state),
        after_state=_opt_tuple(s.after_state),
        size=s.size,
        node_indices=_opt_tuple(s.node_indices),
        cause_indices=_opt_tuple(s.cause_indices),
        effect_indices=_opt_tuple(s.effect_indices),
        node_labels=_dec_optional(s.node_labels),
    )


def _enc_optional_direction(direction: Any) -> Any:
    if direction is None:
        return None
    return schema.DirectionSchema(name=direction.name)


def _register_complex() -> None:
    from pyphi.models.complex import Complex

    _ENCODERS[Complex] = lambda c: schema.ComplexSchema(
        sia=to_schema(c.sia),
        substrate=to_schema(c.substrate),
        is_maximal=bool(c.is_maximal),
        excluded=tuple(to_schema(e) for e in c.excluded),
    )
    _DECODERS[schema.ComplexSchema] = lambda s: Complex(
        sia=from_schema(s.sia),
        substrate=from_schema(s.substrate),
        is_maximal=s.is_maximal,
        excluded=tuple(from_schema(e) for e in s.excluded),
    )


_REGISTERED = False


def _ensure_registered() -> None:
    """Populate the encoder/decoder registries on first use.

    Registration imports the domain modules; deferring it to the first
    ``to_schema``/``from_schema`` call keeps ``import pyphi.serialize`` free of
    domain imports (and free of import cycles).
    """
    global _REGISTERED  # noqa: PLW0603
    if _REGISTERED:
        return
    _REGISTERED = True
    _register_direction()
    _register_pyphi_float()
    _register_distance_result()
    _register_node_labels()
    _register_state_specification()
    _register_system_state_specification()
    _register_part()
    _register_null_cut()
    _register_directed_bipartition()
    _register_joint_partition()
    _register_joint_bipartition()
    _register_joint_tripartition()
    _register_directed_joint_partition()
    _register_edge_cut()
    _register_complete_edge_cut()
    _register_directed_set_partition()
    _register_ria()
    _register_mice()
    _register_distinction()
    _register_distinctions()
    _register_provenance()
    _register_excluded_candidate()
    _register_iit3_sia()
    _register_iit4_sia()
    _register_relation()
    _register_relations()
    _register_ces()
    _register_substrate()
    _register_system()
    _register_transition()
    _register_ac_ria()
    _register_causal_link()
    _register_account()
    _register_ac_sia()
    _register_complex()
