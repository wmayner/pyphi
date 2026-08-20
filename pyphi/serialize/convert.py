"""Convert between PyPhi domain objects and their msgspec schema Structs.

Two registries map a domain type to its encoder and a schema Struct type to its
decoder. Each serializable type adds one ``_register_<type>()`` populating both
registries, invoked on first use via ``_ensure_registered()``.
"""

import contextvars
import math
from collections.abc import Callable
from typing import Any

import msgspec
import numpy as np

from pyphi.direction import Direction

from . import arrays
from . import frames
from . import schema

_ENCODERS: dict[type, Callable[[Any], Any]] = {}  # domain type   -> encode
_DECODERS: dict[type, Callable[[Any], Any]] = {}  # schema Struct  -> decode


def to_schema(obj: Any) -> Any:
    _ensure_registered()
    encode = _ENCODERS.get(type(obj))
    if encode is None:
        # A φ value stored as a native float serializes as-is (msgspec handles
        # it), so it needs no schema Struct.
        if type(obj) is float:
            return obj
        raise TypeError(f"No serializer registered for {type(obj).__name__}")
    return encode(obj)


def from_schema(struct: Any) -> Any:
    _ensure_registered()
    decode = _DECODERS.get(type(struct))
    if decode is None:
        # A native float decoded from a φ position round-trips unchanged.
        if type(struct) is float:
            return struct
        raise TypeError(f"No deserializer registered for {type(struct).__name__}")
    return decode(struct)


def _enc_optional(obj: Any) -> Any:
    """Encode a nested domain object that may be ``None``."""
    return to_schema(obj) if obj is not None else None


# Document label frame. dumps()/loads() establish these contexts; encoders
# and decoders resolve per-object labels against them. Outside a document
# context (a direct to_schema/from_schema call), labels stay per-object.
_ENC_FRAME: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "_ENC_FRAME", default=None
)
_DEC_FRAME: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_DEC_FRAME", default=None
)


def _enc_labels(labels: Any) -> Any:
    """Encode a ``node_labels`` attribute against the document frame.

    The first labeled object claims the frame and writes ``None`` into its
    own struct; labels equal to the frame also write ``None``; labels that
    differ are written per-object.
    """
    if labels is None:
        return None
    encoded = to_schema(labels)
    holder = _ENC_FRAME.get()
    if holder is None:
        return encoded
    if holder[0] is None:
        holder[0] = encoded
        return None
    if encoded == holder[0]:
        return None
    return encoded


def _dec_labels(stored: Any) -> Any:
    """Resolve labels: the object's own stored labels, else the frame."""
    if stored is not None:
        return from_schema(stored)
    return _DEC_FRAME.get()


def encode_document(obj: Any) -> tuple[Any, Any]:
    """Encode ``obj`` to a payload struct plus the claimed label frame."""
    holder: list = [None]
    token = _ENC_FRAME.set(holder)
    try:
        payload = to_schema(obj)
    finally:
        _ENC_FRAME.reset(token)
    return payload, holder[0]


def decode_document(payload: Any, frame: Any, node_labels: Any = None) -> Any:
    """Decode ``payload`` under a document label frame.

    ``frame`` is the document's stored ``NodeLabelsSchema`` (or ``None``);
    ``node_labels`` is a caller-supplied domain ``NodeLabels`` that
    replaces it.
    """
    resolved = node_labels
    if resolved is None and frame is not None:
        resolved = from_schema(frame)
    token = _DEC_FRAME.set(resolved)
    try:
        return from_schema(payload)
    finally:
        _DEC_FRAME.reset(token)


def _dec_optional(struct: Any) -> Any:
    """Decode a nested schema struct that may be ``None``."""
    return from_schema(struct) if struct is not None else None


def _register_direction() -> None:
    _ENCODERS[Direction] = lambda d: schema.DirectionSchema(name=d.name)
    _DECODERS[schema.DirectionSchema] = lambda s: Direction[s.name]


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
        runner_up_state=_opt_tuple(spec.runner_up_state),
        runner_up_intrinsic_information=_enc_optional(
            spec.runner_up_intrinsic_information
        ),
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
        runner_up_state=_opt_tuple(struct.runner_up_state),
        runner_up_intrinsic_information=_dec_optional(
            struct.runner_up_intrinsic_information
        ),
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
        node_labels=_enc_labels(c.node_labels),
    )
    _DECODERS[schema.EdgeCutSchema] = lambda s: EdgeCut(
        tuple(s.node_indices),
        arrays.bytes_to_array(s.cut_matrix),
        _dec_labels(s.node_labels),
    )


def _register_complete_edge_cut() -> None:
    from pyphi.models.partitions import CompleteEdgeCut

    _ENCODERS[CompleteEdgeCut] = lambda c: schema.CompleteEdgeCutSchema(
        node_indices=tuple(c.node_indices),
        node_labels=_enc_labels(c.node_labels),
    )
    _DECODERS[schema.CompleteEdgeCutSchema] = lambda s: CompleteEdgeCut(
        tuple(s.node_indices), _dec_labels(s.node_labels)
    )


def _register_directed_set_partition() -> None:
    from pyphi.models.partitions import DirectedSetPartition

    _ENCODERS[DirectedSetPartition] = lambda c: schema.DirectedSetPartitionSchema(
        node_indices=tuple(c.node_indices),
        cut_matrix=arrays.array_to_bytes(np.asarray(c._cut_matrix)),
        set_partition=tuple(tuple(part) for part in c.set_partition),
        node_labels=_enc_labels(c.node_labels),
    )
    _DECODERS[schema.DirectedSetPartitionSchema] = lambda s: DirectedSetPartition(
        node_indices=tuple(s.node_indices),
        cut_matrix=arrays.bytes_to_array(s.cut_matrix),
        set_partition=[list(part) for part in s.set_partition],
        node_labels=_dec_labels(s.node_labels),
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
        node_labels=_enc_labels(ria.node_labels),
        partition_tie_peers=tuple(
            _encode_ria(p, include_peers=False) for p in partition_peers
        ),
        state_tie_peers=tuple(_encode_ria(p, include_peers=False) for p in state_peers),
        partition_margin=_enc_optional(ria.partition_margin),
        signed_phi=_enc_optional(ria.signed_phi),
        selectivity=ria.selectivity,
        reasons=_enc_reasons(ria.reasons),
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
        node_labels=_dec_labels(struct.node_labels),
        partition_margin=_dec_optional(struct.partition_margin),
        signed_phi=_dec_optional(struct.signed_phi),
        selectivity=struct.selectivity,
        reasons=_dec_reasons(struct.reasons),
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


def _mice_struct_cls(mice: Any) -> Any:
    from pyphi.models.mice import MaximallyIrreducibleCause
    from pyphi.models.mice import MaximallyIrreducibleEffect

    if isinstance(mice, MaximallyIrreducibleCause):
        return schema.MICECauseSchema
    if isinstance(mice, MaximallyIrreducibleEffect):
        return schema.MICEEffectSchema
    return schema.MICESchema


def _encode_mice(mice: Any, struct_cls: Any, *, include_peers: bool = True) -> Any:
    # Purview ties are tri-state: None = never computed; () = computed with
    # no ties; otherwise the tied peers excluding this MICE, each encoded
    # with its own tie field suppressed (the shared tie tuple contains this
    # MICE, so recursing into peers' ties would never terminate).
    peers: tuple | None = None
    if mice._purview_ties is not None:
        peers = (
            tuple(
                _encode_mice(t, _mice_struct_cls(t), include_peers=False)
                for t in mice._purview_ties
                if t is not mice
            )
            if include_peers
            else ()
        )
    return struct_cls(
        ria=to_schema(mice.ria),
        purview_margin=_enc_optional(mice.purview_margin),
        purview_tie_peers=peers,
    )


def _decode_mice(cls: type, struct: Any) -> Any:
    instance = cls(from_schema(struct.ria))
    if struct.purview_tie_peers is None:
        instance._purview_ties = None
    else:
        peers = tuple(from_schema(p) for p in struct.purview_tie_peers)
        tied = (instance, *peers)
        instance._purview_ties = tied
        for peer in peers:
            peer._purview_ties = tied
    instance.purview_margin = _dec_optional(struct.purview_margin)
    return instance


def _register_mice() -> None:
    from pyphi.models.mice import MaximallyIrreducibleCause
    from pyphi.models.mice import MaximallyIrreducibleCauseOrEffect
    from pyphi.models.mice import MaximallyIrreducibleEffect

    _ENCODERS[MaximallyIrreducibleCauseOrEffect] = lambda m: _encode_mice(
        m, schema.MICESchema
    )
    _ENCODERS[MaximallyIrreducibleCause] = lambda m: _encode_mice(
        m, schema.MICECauseSchema
    )
    _ENCODERS[MaximallyIrreducibleEffect] = lambda m: _encode_mice(
        m, schema.MICEEffectSchema
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
        estimator=p.estimator,
    )
    _DECODERS[schema.ProvenanceSchema] = lambda s: Provenance(
        **msgspec.structs.asdict(s)
    )


def _register_macro_unit() -> None:
    from pyphi.macro.units import MacroUnit

    def _enc(u: Any) -> Any:
        return schema.MacroUnitSchema(
            constituents=tuple(
                _enc(c) if isinstance(c, MacroUnit) else int(c) for c in u.constituents
            ),
            update_grain=u.update_grain,
            mapping=tuple(u.mapping),
            background_apportionment=tuple(u.background_apportionment),
        )

    def _dec(s: Any) -> Any:
        return MacroUnit(
            constituents=tuple(
                _dec(c) if isinstance(c, schema.MacroUnitSchema) else int(c)
                for c in s.constituents
            ),
            update_grain=s.update_grain,
            mapping=tuple(s.mapping),
            background_apportionment=tuple(s.background_apportionment),
        )

    _ENCODERS[MacroUnit] = _enc
    _DECODERS[schema.MacroUnitSchema] = _dec


def _encode_optional_units(units: Any) -> Any:
    if units is None:
        return None
    return tuple(to_schema(u) for u in units)


def _decode_optional_units(units: Any) -> Any:
    if units is None:
        return None
    return tuple(from_schema(u) for u in units)


def _register_excluded_candidate() -> None:
    from pyphi.models.complex import ExcludedCandidate

    _ENCODERS[ExcludedCandidate] = lambda e: schema.ExcludedCandidateSchema(
        node_indices=tuple(e.node_indices),
        phi=float(e.phi),
        units=_encode_optional_units(e.units),
    )
    _DECODERS[schema.ExcludedCandidateSchema] = lambda s: ExcludedCandidate(
        s.node_indices, s.phi, units=_decode_optional_units(s.units)
    )


def _encode_iit3_sia(sia: Any, *, include_peers: bool) -> Any:
    peers = tuple(t for t in sia.ties if t is not sia) if include_peers else ()
    return schema.IIT3SIASchema(
        phi=_enc_optional(sia.phi),
        distinctions=_enc_optional(sia.distinctions),
        partitioned_distinctions=_enc_optional(sia.partitioned_distinctions),
        partition=_enc_optional(sia.partition),
        node_indices=_opt_tuple(sia.node_indices),
        node_labels=_enc_labels(sia.node_labels),
        current_state=_opt_tuple(sia.current_state),
        tie_peers=tuple(_encode_iit3_sia(p, include_peers=False) for p in peers),
        runner_up=_enc_runner_up(sia.runner_up),
        reasons=_enc_reasons(sia.reasons),
        config=_enc_config(sia.config),
        provenance=_enc_optional(sia.provenance),
    )


def _decode_iit3_sia(struct: Any) -> Any:
    from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis

    instance = IIT3SystemIrreducibilityAnalysis(
        phi=_dec_optional(struct.phi),
        distinctions=_dec_optional(struct.distinctions),
        partitioned_distinctions=_dec_optional(struct.partitioned_distinctions),
        partition=_dec_optional(struct.partition),
        node_indices=_opt_tuple(struct.node_indices),
        node_labels=_dec_labels(struct.node_labels),
        current_state=_opt_tuple(struct.current_state),
        runner_up=_dec_runner_up(struct.runner_up),
        reasons=_dec_reasons(struct.reasons),
        config=struct.config,
        provenance=_dec_optional(struct.provenance),
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


def _enc_runner_up(runner_up: Any) -> Any:
    if runner_up is None:
        return None
    return schema.RunnerUpSchema(
        partition=to_schema(runner_up.partition),
        phi=to_schema(runner_up.phi),
    )


def _dec_runner_up(struct: Any) -> Any:
    if struct is None:
        return None
    from pyphi.models.explanation import RunnerUp

    return RunnerUp(partition=from_schema(struct.partition), phi=from_schema(struct.phi))


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
        node_labels=_enc_labels(sia.node_labels),
        intrinsic_differentiation=_enc_intrinsic_diff(sia.intrinsic_differentiation),
        reasons=_enc_reasons(sia.reasons),
        signed_phi=_enc_optional(sia.signed_phi),
        signed_normalized_phi=_enc_optional(sia.signed_normalized_phi),
        config=_enc_config(sia.config),
        provenance=_enc_optional(sia.provenance),
        tie_peers=tuple(_encode_iit4_sia(p, include_peers=False) for p in peers),
        partition_margin=_enc_optional(sia.partition_margin),
        runner_up=_enc_runner_up(sia.runner_up),
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
        "node_labels": _dec_labels(struct.node_labels),
        "intrinsic_differentiation": _dec_intrinsic_diff(
            struct.intrinsic_differentiation
        ),
        "reasons": _dec_reasons(struct.reasons),
        "signed_phi": _dec_optional(struct.signed_phi),
        "signed_normalized_phi": _dec_optional(struct.signed_normalized_phi),
        "config": struct.config,
        "provenance": _dec_optional(struct.provenance),
        "partition_margin": _dec_optional(struct.partition_margin),
        "runner_up": _dec_runner_up(struct.runner_up),
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


def _decode_relations_ref(
    struct: Any, table: list, distinctions: Any | None = None
) -> Any:
    from pyphi.models.distinctions import ResolvedDistinctions
    from pyphi.relations import AnalyticalRelations
    from pyphi.relations import ConcreteRelations
    from pyphi.relations import NullRelations
    from pyphi.relations import Relation

    if type(struct) is schema.NullRelationsRefSchema:
        return NullRelations()
    if type(struct) is schema.AnalyticalRelationsRefSchema:
        # Reuse the already-decoded distinctions object so the wrapper type
        # and identity are shared with the structure's own distinctions.
        if distinctions is not None:
            return AnalyticalRelations(distinctions)
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
        config=_enc_config(ces.config),
        provenance=_enc_optional(ces.provenance),
    )


def _decode_ces(struct: Any, domain_cls: Any) -> Any:
    distinctions = from_schema(struct.distinctions)
    table = list(distinctions)
    relations = _decode_relations_ref(struct.relations, table, distinctions)
    return domain_cls(
        sia=from_schema(struct.sia),
        distinctions=distinctions,
        relations=relations,
        config=struct.config,
        provenance=_dec_optional(struct.provenance),
    )


def _register_ces() -> None:
    from pyphi.formalism.iit4 import NullCauseEffectStructure
    from pyphi.models.ces import CauseEffectStructure

    _ENCODERS[CauseEffectStructure] = lambda c: _encode_ces(c, schema.CESSchema)
    _ENCODERS[NullCauseEffectStructure] = lambda c: _encode_ces(c, schema.NullCESSchema)
    _DECODERS[schema.CESSchema] = lambda s: _decode_ces(s, CauseEffectStructure)
    _DECODERS[schema.NullCESSchema] = lambda s: _decode_ces(s, NullCauseEffectStructure)


def _encode_factor(f: Any) -> tuple[bytes, bool]:
    """Encode one conditional factor, storing only the on-probability slice
    of a binary factor whose off slice is its exact float complement.

    The trim is applied only after verifying ``factor[..., 0] == 1.0 −
    factor[..., 1]`` elementwise, so reconstruction on decode is exact; any
    factor failing the check (including every non-binary factor) is stored
    in full.
    """
    arr = np.asarray(f)
    if arr.shape[-1] == 2 and np.array_equal(arr[..., 0], 1.0 - arr[..., 1]):
        return arrays.array_to_bytes(np.ascontiguousarray(arr[..., 1])), True
    return arrays.array_to_bytes(arr), False


def _decode_factor(data: bytes, trimmed: bool) -> np.ndarray:
    arr = arrays.bytes_to_array(data)
    if trimmed:
        return np.stack([1.0 - arr, arr], axis=-1)
    return arr


def _register_substrate() -> None:
    from pyphi.core.tpm.factored import FactoredTPM
    from pyphi.substrate import Substrate

    def _encode_substrate(s: Substrate) -> schema.SubstrateSchema:
        encoded = [_encode_factor(f) for f in s.factored_tpm.factors]
        return schema.SubstrateSchema(
            factors=tuple(data for data, _ in encoded),
            state_space=tuple(tuple(labels) for labels in s.factored_tpm.state_space),
            cm=arrays.array_to_bytes(np.asarray(s.cm)),
            node_labels=_enc_labels(s.node_labels),
            factors_trimmed=tuple(trimmed for _, trimmed in encoded),
        )

    _ENCODERS[Substrate] = _encode_substrate

    def _decode_substrate(s: schema.SubstrateSchema) -> Substrate:
        trimmed = s.factors_trimmed or (False,) * len(s.factors)
        factored = FactoredTPM(
            factors=tuple(
                _decode_factor(f, t) for f, t in zip(s.factors, trimmed, strict=True)
            ),
            state_space=s.state_space,
        )
        return Substrate.from_factored(
            factored,
            cm=arrays.bytes_to_array(s.cm),
            node_labels=_dec_labels(s.node_labels),
        )

    _DECODERS[schema.SubstrateSchema] = _decode_substrate


def _register_system() -> None:
    from pyphi.system import System

    _ENCODERS[System] = lambda s: schema.SystemSchema(
        substrate=to_schema(s.substrate),
        state=tuple(s.state),
        node_indices=tuple(s.node_indices),
        partition=to_schema(s.partition),
        external_indices=tuple(s.external_indices),
        background_conditioning=s.background_conditioning,
        background_state=(
            tuple(s.background_state) if s.background_state is not None else None
        ),
    )
    _DECODERS[schema.SystemSchema] = lambda s: System(
        substrate=from_schema(s.substrate),
        state=tuple(s.state),
        node_indices=tuple(s.node_indices),
        partition=from_schema(s.partition),
        external_indices=tuple(s.external_indices),
        background_conditioning=s.background_conditioning,
        background_state=(
            tuple(s.background_state) if s.background_state is not None else None
        ),
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
        noise_background=t.noise_background,
    )
    _DECODERS[schema.TransitionSchema] = lambda t: Transition(
        substrate=from_schema(t.substrate),
        before_state=tuple(t.before_state),
        after_state=tuple(t.after_state),
        cause_indices=tuple(t.cause_indices),
        effect_indices=tuple(t.effect_indices),
        partition=from_schema(t.partition),
        noise_background=t.noise_background,
    )

    from pyphi.actual import TransitionSystem

    _ENCODERS[TransitionSystem] = lambda t: schema.TransitionSystemSchema(
        substrate=to_schema(t.substrate),
        before_state=tuple(t.before_state),
        after_state=tuple(t.after_state),
        cause_indices=tuple(t.cause_indices),
        effect_indices=tuple(t.effect_indices),
        direction=schema.DirectionSchema(name=t.direction.name),
        partition=to_schema(t.partition),
        noise_background=t.noise_background,
    )
    _DECODERS[schema.TransitionSystemSchema] = lambda t: TransitionSystem(
        substrate=from_schema(t.substrate),
        before_state=tuple(t.before_state),
        after_state=tuple(t.after_state),
        cause_indices=tuple(t.cause_indices),
        effect_indices=tuple(t.effect_indices),
        direction=from_schema(t.direction),
        partition=from_schema(t.partition),
        noise_background=t.noise_background,
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
        node_labels=_enc_labels(ria.node_labels),
        reasons=_enc_reasons(ria.reasons),
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
        node_labels=_dec_labels(struct.node_labels),
        reasons=_dec_reasons(struct.reasons),
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


def _encode_ac_sia(s: Any, *, include_peers: bool) -> Any:
    peers = tuple(t for t in s.ties if t is not s) if include_peers else ()
    return schema.AcSIASchema(
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
        node_labels=_enc_labels(s.node_labels),
        reasons=_enc_reasons(s.reasons),
        config=_enc_config(s.config),
        provenance=_enc_optional(s.provenance),
        tie_peers=tuple(_encode_ac_sia(p, include_peers=False) for p in peers),
    )


def _decode_ac_sia(struct: Any) -> Any:
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis

    instance = AcSystemIrreducibilityAnalysis(
        alpha=struct.alpha,
        direction=_dec_optional(struct.direction),
        account=_dec_optional(struct.account),
        partitioned_account=_dec_optional(struct.partitioned_account),
        partition=_dec_optional(struct.partition),
        before_state=_opt_tuple(struct.before_state),
        after_state=_opt_tuple(struct.after_state),
        size=struct.size,
        node_indices=_opt_tuple(struct.node_indices),
        cause_indices=_opt_tuple(struct.cause_indices),
        effect_indices=_opt_tuple(struct.effect_indices),
        node_labels=_dec_labels(struct.node_labels),
        reasons=_dec_reasons(struct.reasons),
        config=struct.config,
        provenance=_dec_optional(struct.provenance),
    )
    if struct.tie_peers:
        peers = tuple(_decode_ac_sia(p) for p in struct.tie_peers)
        tied = (instance, *peers)
        instance._ties = tied
        for peer in peers:
            peer._ties = tied
    return instance


def _register_ac_sia() -> None:
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis

    _ENCODERS[AcSystemIrreducibilityAnalysis] = lambda s: _encode_ac_sia(
        s, include_peers=True
    )
    _DECODERS[schema.AcSIASchema] = _decode_ac_sia


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
        units=_encode_optional_units(c.units),
        node_indices=tuple(c.node_indices),
    )
    _DECODERS[schema.ComplexSchema] = lambda s: Complex(
        sia=from_schema(s.sia),
        substrate=from_schema(s.substrate),
        is_maximal=s.is_maximal,
        excluded=tuple(from_schema(e) for e in s.excluded),
        units=_decode_optional_units(s.units),
        node_indices=s.node_indices,
    )


def _register_analysis() -> None:
    from pyphi.analyze import Analysis

    _ENCODERS[Analysis] = lambda a: schema.AnalysisSchema(
        system=to_schema(a.system),
        sia=to_schema(a.sia),
        ces=to_schema(a.ces),
    )
    _DECODERS[schema.AnalysisSchema] = lambda s: Analysis(
        system=from_schema(s.system),
        sia=from_schema(s.sia),
        ces=from_schema(s.ces),
    )


def _register_coverage_report() -> None:
    from pyphi.estimate import CoverageReport

    _ENCODERS[CoverageReport] = lambda c: schema.CoverageReportSchema(
        counts=arrays.array_to_bytes(np.asarray(c.counts)),
        n_units=c.n_units,
    )
    _DECODERS[schema.CoverageReportSchema] = lambda s: CoverageReport(
        counts=arrays.bytes_to_array(s.counts),
        n_units=s.n_units,
    )


def _register_substrate_posterior() -> None:
    from pyphi.estimate import SubstratePosterior

    _ENCODERS[SubstratePosterior] = lambda p: schema.SubstratePosteriorSchema(
        alpha_on=arrays.array_to_bytes(np.asarray(p.alpha_on)),
        alpha_off=arrays.array_to_bytes(np.asarray(p.alpha_off)),
        regime=p.regime,
        prior=float(p.prior),
        coverage=to_schema(p.coverage),
        node_labels=_opt_tuple(p.node_labels),
        provenance=to_schema(p.provenance),
    )
    _DECODERS[schema.SubstratePosteriorSchema] = lambda s: SubstratePosterior(
        alpha_on=arrays.bytes_to_array(s.alpha_on),
        alpha_off=arrays.bytes_to_array(s.alpha_off),
        regime=s.regime,
        prior=s.prior,
        coverage=from_schema(s.coverage),
        node_labels=_opt_tuple(s.node_labels),
        provenance=from_schema(s.provenance),
    )


def _register_phi_posterior() -> None:
    from pyphi.estimate import PhiPosterior

    _ENCODERS[PhiPosterior] = lambda p: schema.PhiPosteriorSchema(
        samples=arrays.array_to_bytes(np.asarray(p.samples)),
        complex_samples=tuple(tuple(c) for c in p.complex_samples),
        state=tuple(p.state),
        subset=_opt_tuple(p.subset),
        seed=p.seed,
        regime=p.regime,
        coverage=to_schema(p.coverage),
        provenance=to_schema(p.provenance),
        screen_margin=p.screen_margin,
        screened=p.screened,
        reference_margins=(
            None if p.reference_margins is None else dict(p.reference_margins)
        ),
    )
    _DECODERS[schema.PhiPosteriorSchema] = lambda s: PhiPosterior(
        samples=arrays.bytes_to_array(s.samples),
        complex_samples=tuple(tuple(c) for c in s.complex_samples),
        state=tuple(s.state),
        subset=_opt_tuple(s.subset),
        seed=s.seed,
        regime=s.regime,
        coverage=from_schema(s.coverage),
        provenance=from_schema(s.provenance),
        screen_margin=s.screen_margin,
        screened=s.screened,
        reference_margins=(
            None if s.reference_margins is None else dict(s.reference_margins)
        ),
    )


def _register_sweep_result() -> None:
    from pyphi.sweep import SweepResult

    _ENCODERS[SweepResult] = lambda r: schema.SweepResultSchema(
        df=frames.dataframe_to_schema(r.df),
        results=tuple(
            obj if isinstance(obj, float) else to_schema(obj) for obj in r.results
        ),
        skipped=tuple(
            (label, formalism, tuple(subset), tuple(state))
            for label, formalism, subset, state in r.skipped
        ),
    )

    def _decode_sweep_result(s: schema.SweepResultSchema) -> Any:
        return SweepResult(
            df=frames.schema_to_dataframe(s.df),
            results=[
                obj if isinstance(obj, float) else from_schema(obj) for obj in s.results
            ],
            skipped=[
                (label, formalism, tuple(subset), tuple(state))
                for label, formalism, subset, state in s.skipped
            ],
        )

    _DECODERS[schema.SweepResultSchema] = _decode_sweep_result


def _register_campaign() -> None:
    from pyphi.campaign import CampaignTask
    from pyphi.campaign import CampaignTaskOutput
    from pyphi.campaign import CellOutput

    _ENCODERS[CampaignTask] = lambda t: schema.CampaignTaskSchema(
        task_id=t.task_id,
        kind=t.kind,
        compute=t.compute,
        compute_ref=t.compute_ref,
        config_overrides=dict(t.config_overrides),
        cells=tuple(
            (label, formalism, tuple(subset), tuple(state))
            for label, formalism, subset, state in t.cells
        ),
        skip_uncomputable=t.skip_uncomputable,
    )

    def _decode_campaign_task(s: schema.CampaignTaskSchema) -> Any:
        return CampaignTask(
            task_id=s.task_id,
            kind=s.kind,
            compute=s.compute,
            compute_ref=s.compute_ref,
            config_overrides=dict(s.config_overrides),
            cells=tuple(
                (label, formalism, tuple(subset), tuple(state))
                for label, formalism, subset, state in s.cells
            ),
            skip_uncomputable=s.skip_uncomputable,
        )

    _DECODERS[schema.CampaignTaskSchema] = _decode_campaign_task

    _ENCODERS[CellOutput] = lambda e: schema.CellOutputSchema(
        status=e.status,
        result=None if e.result is None else to_schema(e.result),
        traceback=e.traceback,
        aux=None if e.aux is None else dict(e.aux),
    )

    def _decode_cell_output(s: schema.CellOutputSchema) -> Any:
        return CellOutput(
            status=s.status,
            result=None if s.result is None else from_schema(s.result),
            traceback=s.traceback,
            aux=None if s.aux is None else dict(s.aux),
        )

    _DECODERS[schema.CellOutputSchema] = _decode_cell_output

    _ENCODERS[CampaignTaskOutput] = lambda o: schema.CampaignTaskOutputSchema(
        task_id=o.task_id,
        pyphi_version=o.pyphi_version,
        entries=tuple(to_schema(e) for e in o.entries),
        metrics=o.metrics,
    )

    def _decode_campaign_task_output(s: schema.CampaignTaskOutputSchema) -> Any:
        return CampaignTaskOutput(
            task_id=s.task_id,
            pyphi_version=s.pyphi_version,
            entries=tuple(from_schema(e) for e in s.entries),
            metrics=s.metrics,
        )

    _DECODERS[schema.CampaignTaskOutputSchema] = _decode_campaign_task_output

    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope

    _ENCODERS[AxisScope] = lambda a: schema.AxisScopeSchema(
        explicit=a.explicit,
        min_order=a.min_order,
        max_order=a.max_order,
        containing=a.containing,
        within=a.within,
    )

    def _decode_axis_scope(s: schema.AxisScopeSchema) -> Any:
        return AxisScope(
            explicit=None if s.explicit is None else tuple(tuple(e) for e in s.explicit),
            min_order=s.min_order,
            max_order=s.max_order,
            containing=None if s.containing is None else tuple(s.containing),
            within=None if s.within is None else tuple(s.within),
        )

    _DECODERS[schema.AxisScopeSchema] = _decode_axis_scope

    _ENCODERS[CESScope] = lambda c: schema.CESScopeSchema(
        mechanisms=to_schema(c.mechanisms),
        cause_purviews=to_schema(c.cause_purviews),
        effect_purviews=to_schema(c.effect_purviews),
        max_purview_order_by_mechanism_order=c.max_purview_order_by_mechanism_order,
    )

    def _decode_ces_scope(s: schema.CESScopeSchema) -> Any:
        return CESScope(
            mechanisms=from_schema(s.mechanisms),
            cause_purviews=from_schema(s.cause_purviews),
            effect_purviews=from_schema(s.effect_purviews),
            max_purview_order_by_mechanism_order=(
                None
                if s.max_purview_order_by_mechanism_order is None
                else tuple((m, p) for m, p in s.max_purview_order_by_mechanism_order)
            ),
        )

    _DECODERS[schema.CESScopeSchema] = _decode_ces_scope

    from pyphi.campaign import CESShardTask
    from pyphi.campaign import SIAShardTask
    from pyphi.campaign.shards import ShardSpec

    _ENCODERS[ShardSpec] = lambda s: schema.ShardSpecSchema(
        payload_kind=s.payload_kind,
        mechanisms=tuple(tuple(m) for m in s.mechanisms),
        mechanism=None if s.mechanism is None else tuple(s.mechanism),
        direction=s.direction,
        purviews=tuple(tuple(p) for p in s.purviews),
        purview=None if s.purview is None else tuple(s.purview),
        stride=s.stride,
        units=s.units,
        memory_bytes=s.memory_bytes,
    )

    def _decode_shard_spec(s: schema.ShardSpecSchema) -> Any:
        return ShardSpec(
            payload_kind=s.payload_kind,
            mechanisms=tuple(tuple(m) for m in s.mechanisms),
            mechanism=None if s.mechanism is None else tuple(s.mechanism),
            direction=s.direction,
            purviews=tuple(tuple(p) for p in s.purviews),
            purview=None if s.purview is None else tuple(s.purview),
            stride=None if s.stride is None else (s.stride[0], s.stride[1]),
            units=s.units,
            memory_bytes=s.memory_bytes,
        )

    _DECODERS[schema.ShardSpecSchema] = _decode_shard_spec

    _ENCODERS[CESShardTask] = lambda t: schema.CESShardTaskSchema(
        task_id=t.task_id,
        kind=t.kind,
        substrate_label=t.substrate_label,
        state=tuple(t.state),
        subset=None if t.subset is None else tuple(t.subset),
        scope=to_schema(t.scope),
        config_overrides=dict(t.config_overrides),
        formalism=t.formalism,
        spec=to_schema(t.spec),
        ordering=t.ordering,
    )

    def _decode_ces_shard_task(s: schema.CESShardTaskSchema) -> Any:
        return CESShardTask(
            task_id=s.task_id,
            kind=s.kind,
            substrate_label=s.substrate_label,
            state=tuple(s.state),
            subset=None if s.subset is None else tuple(s.subset),
            scope=from_schema(s.scope),
            config_overrides=dict(s.config_overrides),
            formalism=s.formalism,
            spec=from_schema(s.spec),
            ordering=s.ordering,
        )

    _DECODERS[schema.CESShardTaskSchema] = _decode_ces_shard_task

    _ENCODERS[SIAShardTask] = lambda t: schema.SIAShardTaskSchema(
        task_id=t.task_id,
        kind=t.kind,
        substrate_label=t.substrate_label,
        state=tuple(t.state),
        subset=None if t.subset is None else tuple(t.subset),
        config_overrides=dict(t.config_overrides),
        formalism=t.formalism,
        stride=(t.stride[0], t.stride[1]),
    )

    def _decode_sia_shard_task(s: schema.SIAShardTaskSchema) -> Any:
        return SIAShardTask(
            task_id=s.task_id,
            kind=s.kind,
            substrate_label=s.substrate_label,
            state=tuple(s.state),
            subset=None if s.subset is None else tuple(s.subset),
            config_overrides=dict(s.config_overrides),
            formalism=s.formalism,
            stride=(s.stride[0], s.stride[1]),
        )

    _DECODERS[schema.SIAShardTaskSchema] = _decode_sia_shard_task


def _register_optimization_result() -> None:
    from pyphi.optimize import OptimizationResult

    def _encode_optimization_result(r: Any) -> Any:
        best_objective = float(r.best_objective)
        return schema.OptimizationResultSchema(
            best_params=arrays.array_to_bytes(np.asarray(r.best_params, dtype=float)),
            best_objective=None if math.isnan(best_objective) else best_objective,
            best_substrate=to_schema(r.best_substrate),
            best_sia=_enc_optional(r.best_sia),
            trajectory=frames.dataframe_to_schema(r.trajectory),
            bounds=tuple((float(lo), float(hi)) for lo, hi in r.bounds),
            seed=int(r.seed),
            direction=r.direction,
            objective_name=r.objective_name,
            settings=dict(r.settings),
            config_snapshot=dict(r.config_snapshot),
            n_evaluations=int(r.n_evaluations),
            n_unreachable=int(r.n_unreachable),
        )

    _ENCODERS[OptimizationResult] = _encode_optimization_result

    def _decode_optimization_result(s: schema.OptimizationResultSchema) -> Any:
        return OptimizationResult(
            best_params=arrays.bytes_to_array(s.best_params),
            best_objective=math.nan if s.best_objective is None else s.best_objective,
            best_substrate=from_schema(s.best_substrate),
            best_sia=_dec_optional(s.best_sia),
            trajectory=frames.schema_to_dataframe(s.trajectory),
            bounds=[(lo, hi) for lo, hi in s.bounds],
            seed=s.seed,
            direction=s.direction,
            objective_name=s.objective_name,
            settings=s.settings,
            config_snapshot=s.config_snapshot,
            n_evaluations=s.n_evaluations,
            n_unreachable=s.n_unreachable,
        )

    _DECODERS[schema.OptimizationResultSchema] = _decode_optimization_result


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
    _register_macro_unit()
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
    _register_analysis()
    _register_coverage_report()
    _register_substrate_posterior()
    _register_phi_posterior()
    _register_sweep_result()
    _register_campaign()
    _register_optimization_result()
