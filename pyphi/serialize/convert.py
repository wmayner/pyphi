"""Convert between PyPhi domain objects and their msgspec schema Structs.

Two registries map a domain type to its encoder and a schema Struct type to its
decoder. This replaces the per-class ``to_json`` / ``from_json`` methods. Each
serializable type adds one ``_register_<type>()`` populating both registries,
all called at import time.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from pyphi.direction import Direction

from . import arrays
from . import schema

_ENCODERS: dict[type, Callable[[Any], Any]] = {}  # domain type   -> encode
_DECODERS: dict[type, Callable[[Any], Any]] = {}  # schema Struct  -> decode


def to_schema(obj: Any) -> Any:
    encode = _ENCODERS.get(type(obj))
    if encode is None:
        raise TypeError(f"No serializer registered for {type(obj).__name__}")
    return encode(obj)


def from_schema(struct: Any) -> Any:
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


_register_edge_cut()
_register_complete_edge_cut()
_register_directed_set_partition()
_register_ria()
_register_mice()
_register_distinction()
_register_distinctions()
