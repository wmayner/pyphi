# formalism/iit4/__init__.py
"""IIT 4.0 system-level analysis: SIA, distinctions, relations, Φ-structure.

Implements the algorithms from Albantakis et al. 2023 (and the 2026 extension
when configured via the ``IIT_4_0_2026`` formalism). Concrete formalism
classes wrapping these algorithms live in :mod:`pyphi.formalism.iit4.formalism`.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from enum import auto
from enum import unique
from typing import Any
from typing import ClassVar

from pyphi import conf
from pyphi import connectivity
from pyphi import metrics
from pyphi import resolve_ties
from pyphi import utils
from pyphi import validate
from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.conf.snapshot import ConfigSnapshot
from pyphi.core import repertoire_algebra as repertoire
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.formalism import iit3
from pyphi.labels import NodeLabels
from pyphi.metrics.distribution import DistanceResult
from pyphi.metrics.protocols import CompositeMeasure
from pyphi.metrics.protocols import DistributionMeasure
from pyphi.metrics.protocols import StateAwareMeasure
from pyphi.metrics.protocols import StatefulDistributionMeasure
from pyphi.metrics.protocols import satisfies_composite_measure
from pyphi.models import cmp
from pyphi.models import fmt
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.distinctions import Distinctions
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import EdgeCut
from pyphi.models.partitions import NullCut
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.models.state_specification import StateSpecification
from pyphi.models.state_specification import SystemStateSpecification
from pyphi.parallel import MapReduce
from pyphi.partition import system_partitions
from pyphi.relations import ConcreteRelations
from pyphi.relations import Relations
from pyphi.relations import relations as compute_relations
from pyphi.system import System

_SERIALIZING_AS_TIE_PEER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sia_serializing_as_tie_peer", default=False
)

##############################################################################
# Information
##############################################################################


# TODO(4.0) refactor
def system_intrinsic_information(
    system: System,
    *,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    directions: Iterable[Direction] | None = None,
) -> SystemStateSpecification:
    """Return the cause/effect states specified by the system.

    ``specification_measure`` is a Protocol-typed measure callable used to
    compute intrinsic information; passed explicitly by the active
    formalism rather than read from config.

    NOTE: State ties are arbitrarily broken (for now).
    """
    directions = fallback(directions, Direction.both())
    # Ensure directions is not None before converting to tuple
    if directions is None:
        directions = Direction.both()
    directions = tuple(directions)
    # TODO move to Direction
    # TODO have validation methods return the validated value
    validate.directions(directions)
    # TODO(ties) deal with ties here
    ii = {
        direction: system.intrinsic_information(
            direction,
            mechanism=system.node_indices,
            purview=system.node_indices,
            specification_measure=specification_measure,
        )
        for direction in directions
    }
    # Get the state specifications
    # SystemStateSpecification's constructor should handle None values
    # if that's the expected behavior when a direction is not present
    return SystemStateSpecification(
        cause=ii.get(Direction.CAUSE),  # pyright: ignore[reportArgumentType] - Constructor handles Optional
        effect=ii.get(Direction.EFFECT),  # pyright: ignore[reportArgumentType] - Constructor handles Optional
    )


##############################################################################
# Integration
##############################################################################


@dataclass
class SystemIrreducibilityAnalysis(cmp.OrderableByPhi):
    """System-level integrated information.

    ``phi`` is the non-negative integrated-information value defined by
    Eqs. 19-20 (the ``|·|+`` operator applied to the raw integration).
    ``signed_phi`` is the raw value before clamping; when negative, it
    flags "preventative cause" structure that the clamp hides. Construction
    accepts the signed value as ``phi`` and ``__post_init__`` writes the
    clamped value to ``phi`` while preserving the raw value on
    ``signed_phi``. ``normalized_phi`` and ``signed_normalized_phi``
    follow the same pattern.
    """

    phi: float | DistanceResult
    partition: DirectedBipartition | DirectedBipartition | NullCut
    normalized_phi: float = 0
    cause: RepertoireIrreducibilityAnalysis | None = None
    effect: RepertoireIrreducibilityAnalysis | None = None
    system_state: SystemStateSpecification | None = None
    current_state: tuple[int, ...] | None = None
    node_indices: tuple[int, ...] | None = None
    node_labels: NodeLabels | None = None
    intrinsic_differentiation: dict | None = None
    reasons: list | None = None
    signed_phi: float | DistanceResult | None = None
    signed_normalized_phi: float | DistanceResult | None = None
    config: ConfigSnapshot | None = None

    def __post_init__(self):
        if self.config is None:
            from pyphi.conf import config as _global

            self.config = _global.snapshot()
        # Snapshot the raw signed values *before* clamping.
        if self.signed_phi is None:
            self.signed_phi = self.phi
        if self.signed_normalized_phi is None:
            self.signed_normalized_phi = self.normalized_phi
        # Eqs 19-20: clamp negative integration to zero via ``|·|+``.
        clamped_phi = utils.positive_part(self.signed_phi)
        clamped_normalized = utils.positive_part(self.signed_normalized_phi)
        if not isinstance(self.phi, DistanceResult):
            self.phi = PyPhiFloat(clamped_phi)
        else:
            # Clamp the numeric component while preserving the
            # DistanceResult's metadata.
            self.phi = type(self.phi)(clamped_phi, **self.phi._public_aux_data())
        if not isinstance(self.normalized_phi, DistanceResult):
            self.normalized_phi = PyPhiFloat(clamped_normalized)
        else:
            self.normalized_phi = type(self.normalized_phi)(
                clamped_normalized, **self.normalized_phi._public_aux_data()
            )
        if not isinstance(self.signed_phi, DistanceResult):
            self.signed_phi = PyPhiFloat(self.signed_phi)
        if not isinstance(self.signed_normalized_phi, DistanceResult):
            self.signed_normalized_phi = PyPhiFloat(self.signed_normalized_phi)
        if self.intrinsic_differentiation is None:
            self.intrinsic_differentiation = {
                Direction.CAUSE: PyPhiFloat(0),
                Direction.EFFECT: PyPhiFloat(0),
            }

    _sia_attributes: ClassVar[list[str]] = [
        "phi",
        "partition",
        "normalized_phi",
        "signed_phi",
        "signed_normalized_phi",
        "cause",
        "effect",
        "system_state",
        "current_state",
        "node_indices",
        "intrinsic_differentiation",
    ]

    def order_by(self):
        return self.phi

    @property
    def ties(self):
        try:
            return self._ties
        except AttributeError:
            self._ties = [self]
            return self._ties

    def set_ties(self, ties):
        self._ties = ties

    def resolve_system_state(self) -> None:
        """Update system_state to reflect the specified states resolved by the MIP.

        When the system has tied specified states, the MIP resolves the tie by
        selecting the state most vulnerable to the winning partition. This
        back-propagates that resolution into system_state so that downstream
        consumers (e.g., congruence filtering in ces) see the correct
        specified states.
        """
        if self.system_state is None:
            return
        new_cause = self.system_state.cause
        new_effect = self.system_state.effect
        if self.cause is not None and self.cause.specified_state is not None:
            new_cause = self.cause.specified_state
        if self.effect is not None and self.effect.specified_state is not None:
            new_effect = self.effect.specified_state
        if (
            new_cause is not self.system_state.cause
            or new_effect is not self.system_state.effect
        ):
            self.system_state = replace(
                self.system_state, cause=new_cause, effect=new_effect
            )

    def __eq__(self, other):
        return cmp.general_eq(self, other, self._sia_attributes)

    def __bool__(self):
        """Whether |big_phi > 0|."""
        return utils.is_positive(self.phi)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.partition,
            )
        )

    def _repr_columns(self):
        if self.node_labels is not None and self.node_indices is not None:
            # coerce_to_labels returns tuple[str | int, ...], need to convert to str
            system_label = ",".join(
                str(label)
                for label in self.node_labels.coerce_to_labels(self.node_indices)
            )
        elif self.node_indices is not None:
            system_label = ",".join(str(i) for i in self.node_indices)
        else:
            system_label = None

        columns = [
            ("System", system_label),
            (
                "Current state",
                (
                    fmt.state(self.current_state)
                    if self.current_state is not None
                    else None
                ),
            ),
            (f"           {fmt.SMALL_PHI}_s", self.phi),
            (f"Normalized {fmt.SMALL_PHI}_s", self.normalized_phi),
            (
                "Int. diff. CAUSE",
                (
                    self.intrinsic_differentiation[Direction.CAUSE]
                    if self.intrinsic_differentiation
                    else None
                ),
            ),
            (
                "Int. diff. EFFECT",
                (
                    self.intrinsic_differentiation[Direction.EFFECT]
                    if self.intrinsic_differentiation
                    else None
                ),
            ),
        ]

        # Add system_state columns if it exists
        if self.system_state is not None:
            columns.extend(self.system_state._repr_columns())

        columns.extend([("#(tied MIPs)", len(self.ties) - 1), ("Partition", "")])
        if self.reasons:
            columns.append(("Reasons", ", ".join(self.reasons)))
        return columns

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_2)
        body = fmt.center(body)
        column_extent = body.split("\n")[2].index(":")
        if self.partition:
            body += "\n" + fmt.indent(str(self.partition), column_extent + 2)
        return fmt.box(body)

    def to_json(self):
        dct = self.__dict__.copy()
        dct.pop("_ties", None)
        if _SERIALIZING_AS_TIE_PEER.get():
            return dct
        peers = tuple(t for t in self.ties if t is not self)
        if peers:
            from pyphi.jsonify import jsonify

            token = _SERIALIZING_AS_TIE_PEER.set(True)
            try:
                dct["_tie_peers"] = [jsonify(p.to_json()) for p in peers]
            finally:
                _SERIALIZING_AS_TIE_PEER.reset(token)
        return dct

    @classmethod
    def from_json(cls, dct):
        peers_raw: Any = dct.pop("_tie_peers", ())
        peers: tuple[SystemIrreducibilityAnalysis, ...] = tuple(
            cls(**dict(p)) for p in peers_raw
        )
        instance = cls(**dct)
        if peers:
            tied: list[SystemIrreducibilityAnalysis] = [instance, *peers]
            instance._ties = tied
            for peer in peers:
                peer._ties = tied
        return instance


class NullSystemIrreducibilityAnalysis(SystemIrreducibilityAnalysis):
    def __init__(self, node_indices=None, node_labels=None, **kwargs):
        from pyphi.models import NullCut

        # NullCut requires indices, use empty tuple if not provided
        indices = node_indices if node_indices is not None else ()
        super().__init__(
            phi=0,
            partition=NullCut(indices, node_labels),
            cause=None,
            effect=None,
            node_indices=node_indices,
            node_labels=node_labels,
            **kwargs,
        )

    @classmethod
    def from_json(cls, dct):
        """Deserialize from JSON.

        The JSON dict contains all attributes including phi, partition, etc.
        We can construct directly using the parent class since all attributes
        are already present in the dictionary.
        """
        # Use parent class constructor directly with all attributes
        obj = object.__new__(cls)
        SystemIrreducibilityAnalysis.__init__(obj, **dct)
        return obj

    def _repr_columns(self):
        columns = []
        # Handle node_labels and node_indices being None
        if self.node_labels is not None and self.node_indices is not None:
            # coerce_to_labels returns tuple[str | int, ...], need to convert to str
            system_label = ",".join(
                str(label)
                for label in self.node_labels.coerce_to_labels(self.node_indices)
            )
            columns.append(("System", system_label))
        elif self.node_indices is not None:
            system_label = ",".join(str(i) for i in self.node_indices)
            columns.append(("System", system_label))

        columns.append((f"           {fmt.BIG_PHI}", self.phi))
        if self.system_state is not None:
            columns.append(self.system_state._repr_columns())
        if self.reasons:
            columns.append(("Reasons", ", ".join([r.name for r in self.reasons])))
        return columns


def normalization_factor(partition: DirectedBipartition | EdgeCut) -> float:
    if hasattr(partition, "normalization_factor"):
        return partition.normalization_factor()  # pyright: ignore[reportAttributeAccessIssue]
    # For EdgeCut, we need to check hasattr before accessing attributes
    if hasattr(partition, "from_nodes") and hasattr(partition, "to_nodes"):
        return 1 / (len(partition.from_nodes) * len(partition.to_nodes))  # pyright: ignore[reportAttributeAccessIssue]
    # Default fallback
    return 1.0


def _integration_value_for_state(
    direction: Direction,
    system: System,
    cut_system: System,
    partition: DirectedBipartition,
    specified: StateSpecification,
    repertoire_distance: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
) -> RepertoireIrreducibilityAnalysis:
    """Compute the integration value for a single specified state."""
    mechanism = purview = system.node_indices
    if satisfies_composite_measure(repertoire_distance):
        partitioned_repertoire = cut_system.forward_repertoire(
            direction,
            mechanism,
            purview,
            specified.state,
        ).squeeze()[specified.state]
    else:
        partitioned_repertoire = cut_system.repertoire(
            direction, system.node_indices, system.node_indices
        )
    from pyphi.formalism.queries import evaluate_partition

    return evaluate_partition(
        system,
        direction,
        system.node_indices,
        system.node_indices,
        partition,  # pyright: ignore[reportArgumentType] - DirectedBipartition passed to JointBipartition param in IIT 4.0
        partitioned_repertoire=partitioned_repertoire,
        repertoire_distance=repertoire_distance,
        state=specified,
    )


def integration_value(
    direction: Direction,
    system: System,
    partition: DirectedBipartition,
    system_state: SystemStateSpecification,
    *,
    system_measure: CompositeMeasure,
) -> RepertoireIrreducibilityAnalysis:
    """Compute the integration value for a partition along a direction.

    Evaluates against the spec stored at ``system_state[direction]``;
    tied specified states are handled at the orchestration layer (see
    :func:`sia`) by enumerating them and calling this function per
    candidate. ``system_measure`` is a Protocol-typed composite measure
    passed explicitly by the caller (no config fallback).
    """
    cut_system = system.apply_cut(partition)
    specified = system_state[direction]
    return _integration_value_for_state(
        direction,
        system,
        cut_system,
        partition,
        specified,
        system_measure,  # pyright: ignore[reportArgumentType]
    )


def intrinsic_differentiation_value(
    direction: Direction,
    system: System,
) -> float:
    mechanism = purview = system.node_indices

    unpartitioned_repertoire = repertoire.forward_repertoire(
        system,
        direction,
        mechanism,  # pyright: ignore[reportArgumentType]
        purview,  # pyright: ignore[reportArgumentType]
    )

    return metrics.distribution.intrinsic_differentiation(
        unpartitioned_repertoire,
        state=system.proper_state,
    )


def evaluate_partition(
    partition: DirectedBipartition,
    system: System,
    system_state: SystemStateSpecification,
    *,
    system_measure: CompositeMeasure,
    directions: Iterable[Direction] | None = None,
) -> SystemIrreducibilityAnalysis:
    """Evaluate a system-level partition and return the resulting SIA.

    ``system_measure`` is a Protocol-typed composite measure used at the
    system level; passed explicitly by the caller (no config fallback).
    Partition integration uses ``system_measure.partition_measure`` if
    set (otherwise ``system_measure`` itself), and the ``ii(s)`` cap
    (Eq. 23) is applied when ``system_measure.applies_ii_cap`` is True.
    """
    directions = fallback(directions, Direction.both())
    if directions is None:
        directions = Direction.both()
    directions = tuple(directions)
    validate.directions(directions)

    # Eqs. 19-20: partition integration uses the composite measure's
    # ``partition_measure`` (GID for II; self for GID). The ii(s) cap
    # (Eq. 23) is applied separately below.
    partition_distance: CompositeMeasure = (
        system_measure.partition_measure or system_measure
    )

    integration = {
        direction: integration_value(
            direction,
            system,
            partition,
            system_state,
            system_measure=partition_distance,
        )
        for direction in directions
    }

    intrinsic_differentiation = {
        direction: intrinsic_differentiation_value(
            direction,
            system,
        )
        for direction in directions
    }

    # Take the min over directions on the *signed* phi so the resulting
    # SIA's ``signed_phi`` metadata captures the raw preventative-cause
    # value when present. The canonical (clamped) ``phi`` is derived in
    # ``SystemIrreducibilityAnalysis.__post_init__`` via the |·|+ operator.
    # ``min`` and ``positive_part`` commute, so the clamped result is the
    # same as taking the min of clamped values.
    phi = min(integration[direction].signed_phi for direction in directions)

    # Eq. 23: φ_s(s) = min{φ_c(s), φ_e(s), ii(s)}
    # where ii(s) = min_d{min(i_diff_d, i_spec_d)}.
    # Clamp the cap components via the |·|+ operator (Eqs. 19-20); the
    # result still flows through ``signed_phi`` so the clamp at SIA
    # construction yields the right canonical value.
    if system_measure.applies_ii_cap:
        for direction in directions:
            i_spec = utils.positive_part(system_state[direction].intrinsic_information)
            i_diff = utils.positive_part(intrinsic_differentiation[direction])
            phi = min(phi, i_spec, i_diff)

    norm = normalization_factor(partition)
    normalized_phi = phi * norm

    result = SystemIrreducibilityAnalysis(
        phi=phi,
        normalized_phi=normalized_phi,
        cause=integration.get(Direction.CAUSE),
        effect=integration.get(Direction.EFFECT),
        partition=partition,
        system_state=system_state,
        current_state=system.proper_state,
        node_indices=system.node_indices,
        node_labels=system.node_labels,
        intrinsic_differentiation=intrinsic_differentiation,
    )
    return result


@unique
class ShortCircuitConditions(Enum):
    NO_VALID_PARTITIONS = auto()
    NO_CAUSE = auto()
    NO_EFFECT = auto()
    NO_SYSTEM = auto()
    NO_STRONG_CONNECTIVITY = auto()
    MONAD_WITH_NO_SELFLOOP = auto()
    MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI = auto()


def _has_no_cause_or_effect(system_state):
    reasons = []
    for direction, reason in zip(
        Direction.both(),
        [ShortCircuitConditions.NO_CAUSE, ShortCircuitConditions.NO_EFFECT],
        strict=False,
    ):
        if system_state[direction].intrinsic_information <= 0:
            reasons.append(reason)
    return reasons


def sia(
    system: System,
    *,
    system_measure: CompositeMeasure,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    directions: Iterable[Direction] | None = None,
    partition_scheme: str | None = None,
    partitions: Iterable | None = None,
    system_state: SystemStateSpecification | None = None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system.

    ``system_measure`` and ``specification_measure`` are Protocol-typed
    measure callables passed explicitly by the active formalism (no
    config fallback). ``system_measure`` drives system-level partition
    integration (and the ``ii(s)`` cap, if ``INTRINSIC_INFORMATION``);
    ``specification_measure`` drives the intrinsic-information
    computation of the system state.
    """
    partition_scheme = fallback(
        partition_scheme, config.formalism.iit.system_partition_scheme
    )

    # TODO(4.0): trivial reducibility

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the system is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null SIA.
    def _null_sia(**kwargs):
        return NullSystemIrreducibilityAnalysis(
            system_state=system_state,
            node_indices=system.node_indices,
            node_labels=system.node_labels,
            **kwargs,
        )

    if not system:
        # system is empty
        return _null_sia(reasons=[ShortCircuitConditions.NO_SYSTEM])

    if not connectivity.is_strong(system.cm, system.node_indices):
        # system is not strongly connected
        return _null_sia(reasons=[ShortCircuitConditions.NO_STRONG_CONNECTIVITY])

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(system.partition_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not system.cm[system.node_indices][system.node_indices]:
            return _null_sia(reasons=[ShortCircuitConditions.MONAD_WITH_NO_SELFLOOP])
        # Even if the node has a self-loop, we may still define phi to be zero.
        if not config.formalism.iit.single_micro_nodes_with_selfloops_have_phi:
            return _null_sia(
                reasons=[
                    ShortCircuitConditions.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI
                ]
            )
    # =========================================================================

    if partitions is None:
        filter_func = None
        if partitions == "GENERAL":

            def is_disconnecting_partition(partition):
                # Special case for length 1 systems so complete partition is included
                return (
                    not connectivity.is_strong(system.apply_cut(partition).proper_cm)
                ) or len(system) == 1

            filter_func = is_disconnecting_partition

        partitions = system_partitions(
            system.node_indices,
            node_labels=system.node_labels,
            partition_scheme=partition_scheme,
            filter_func=filter_func,
        )

    if system_state is None:
        system_state = system_intrinsic_information(
            system,
            specification_measure=specification_measure,
            directions=directions,
        )

    if config.formalism.iit.shortcircuit_sia:
        shortcircuit_reasons = _has_no_cause_or_effect(system_state)
        if shortcircuit_reasons:
            return _null_sia(reasons=shortcircuit_reasons)

    default_sia = _null_sia(reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS])
    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_partition_evaluation), **kwargs
    )

    # Per Albantakis et al. 2023 S1: when cause/effect specified states
    # tie at maximum intrinsic information, the canonical winner is the
    # joint ``(cause_spec, effect_spec)`` pair whose system phi
    # ``φ_s = min(φ_c, φ_e)`` is greatest. The cascade enumerates the
    # Cartesian product of tied specs and selects via
    # :func:`resolve_ties.resolve_state_tie`. ``partitions`` is
    # materialized so each per-pair MIP search can iterate it
    # independently.
    if not isinstance(partitions, (list, tuple)):
        partitions = list(partitions)

    cause_specs = _spec_candidates(system_state.cause)
    effect_specs = _spec_candidates(system_state.effect)

    if len(cause_specs) <= 1 and len(effect_specs) <= 1:
        mip_sia = _find_mip_for_fixed_state(
            system=system,
            system_state=system_state,
            partitions=partitions,
            system_measure=system_measure,
            directions=directions,
            parallel_kwargs=parallel_kwargs,
            default_sia=default_sia,
        )
    else:
        per_pair_sias: dict[tuple, SystemIrreducibilityAnalysis] = {}
        for c in cause_specs:
            for e in effect_specs:
                forced_state = _build_untied_system_state(c, e)
                key = (
                    c.state if c is not None else None,
                    e.state if e is not None else None,
                )
                per_pair_sias[key] = _find_mip_for_fixed_state(
                    system=system,
                    system_state=forced_state,
                    partitions=partitions,
                    system_measure=system_measure,
                    directions=directions,
                    parallel_kwargs=parallel_kwargs,
                    default_sia=default_sia,
                )

        # Apply the per-state max-min cascade at the Integration level.
        # The Composition step of the cascade requires a ``big_phi`` value
        # on each per-pair SIA (CES Φ), which the SIA does not carry; the
        # Integration budget keeps that key out of the resolver's read
        # path.
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Integration")
        outcome = resolve_ties.resolve_state_tie(per_pair_sias, context=ctx)
        chosen_key = outcome.resolved
        if chosen_key is None:
            assert outcome.tied_set, "cascade outcome has neither winner nor ties"
            chosen_key = outcome.tied_set[0]
        mip_sia = per_pair_sias[chosen_key]

        _restore_tie_metadata(
            mip_sia,
            original_cause=system_state.cause,
            original_effect=system_state.effect,
        )
        mip_sia.set_ties(tuple(per_pair_sias.values()))

    if config.infrastructure.clear_system_caches_after_computing_sia:
        system.clear_caches()

    return mip_sia


_sia = sia


def _spec_candidates(state_spec: StateSpecification | None) -> tuple:
    """Return the tied set of state specs, or ``(state_spec,)`` if untied.

    Empty tuple when ``state_spec`` is ``None`` (degenerate / null state).
    """
    if state_spec is None:
        return ()
    return state_spec.ties if state_spec.ties else (state_spec,)


def _build_untied_system_state(
    cause: StateSpecification | None,
    effect: StateSpecification | None,
) -> SystemStateSpecification:
    """Build a ``SystemStateSpecification`` whose ``cause`` and ``effect``
    fields are the given specs with empty ``ties``. Used by the
    system-state cascade to force a single (cause, effect) pair through
    one MIP search.
    """
    cause_untied = replace(cause, _ties=()) if cause is not None else None
    effect_untied = replace(effect, _ties=()) if effect is not None else None
    return SystemStateSpecification(
        cause=cause_untied,  # pyright: ignore[reportArgumentType]
        effect=effect_untied,  # pyright: ignore[reportArgumentType]
    )


def _restore_tie_metadata(
    sia_result: SystemIrreducibilityAnalysis,
    *,
    original_cause: StateSpecification | None,
    original_effect: StateSpecification | None,
) -> None:
    """Set ``sia_result.system_state.cause.ties`` and ``.effect.ties`` to
    the tied sets from ``original_cause`` / ``original_effect`` while
    keeping the chosen state values. Surfaces the full tied set to
    downstream consumers that inspect ``sia.system_state.cause.ties``.
    """
    if sia_result.system_state is None:
        return
    chosen_cause = sia_result.system_state.cause
    chosen_effect = sia_result.system_state.effect
    new_cause = chosen_cause
    new_effect = chosen_effect
    if chosen_cause is not None and original_cause is not None and original_cause.ties:
        new_cause = replace(chosen_cause, _ties=original_cause.ties)
    if (
        chosen_effect is not None
        and original_effect is not None
        and original_effect.ties
    ):
        new_effect = replace(chosen_effect, _ties=original_effect.ties)
    sia_result.system_state = replace(
        sia_result.system_state, cause=new_cause, effect=new_effect
    )


def _find_mip_for_fixed_state(
    *,
    system: System,
    system_state: SystemStateSpecification,
    partitions: Iterable,
    system_measure: CompositeMeasure,
    directions: Iterable[Direction] | None,
    parallel_kwargs: dict,
    default_sia: SystemIrreducibilityAnalysis,
) -> SystemIrreducibilityAnalysis:
    """Find the MIP for a given (fixed) system state.

    Runs the partition MapReduce with the supplied ``system_state``,
    selects the MIP via :func:`resolve_ties.sias`, and back-propagates
    the chosen state onto each tied MIP's ``system_state``.
    """
    sias = MapReduce(
        evaluate_partition,
        partitions,
        map_kwargs={
            "system": system,
            "system_state": system_state,
            "system_measure": system_measure,
            "directions": directions,
        },
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating partitions",
        **parallel_kwargs,
    ).run()

    candidates = list(sias) if sias is not None else []
    if not candidates:
        candidates = [default_sia]
    ties = tuple(resolve_ties.sias(candidates))
    mip_sia = ties[0]
    for tied_mip in ties:
        tied_mip.resolve_system_state()
        tied_mip.set_ties(ties)
    return mip_sia


##############################################################################
# Composition
##############################################################################


class NullCauseEffectStructure(CauseEffectStructure):
    def __init__(self, **kwargs):
        super().__init__(
            sia=NullSystemIrreducibilityAnalysis(),
            distinctions=ResolvedDistinctions([]),
            relations=ConcreteRelations([]),
            **kwargs,
        )


def ces(
    system: System,
    *,
    system_measure: CompositeMeasure,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    sia: SystemIrreducibilityAnalysis | None = None,
    distinctions: Distinctions | None = None,
    relations: Relations | None = None,
    sia_kwargs: dict | None = None,
    ces_kwargs: dict | None = None,
    relations_kwargs: dict | None = None,
) -> CauseEffectStructure:
    """Analyze the irreducible cause-effect structure of a system (Eq. 57).

    ``system_measure`` and ``specification_measure`` are Protocol-typed
    measure callables passed explicitly by the active formalism (no
    config fallback).
    """
    sia_kwargs = sia_kwargs or {}
    ces_kwargs = ces_kwargs or {}
    relations_kwargs = relations_kwargs or {}

    # Analyze irreducibility if not provided
    if sia is None:
        sia = _sia(
            system,
            system_measure=system_measure,
            specification_measure=specification_measure,
            **sia_kwargs,
        )

    # Compute distinctions if not provided
    if distinctions is None:
        # CES building dispatches find_mice through the active formalism;
        # the iit3 helper is reused because the outer mechanism x purview
        # iteration is the same in both formalisms.
        distinctions = iit3.ces(system, **ces_kwargs)  # type: ignore[arg-type]
    # Resolve tied specified states against a SIA system_state. When the
    # SIA is null (degenerate substrate, e.g., not strongly connected)
    # we still want a usable bag of distinctions for diagnostics, so
    # fall back to the system's intrinsic-information state.
    if sia.system_state is not None:
        resolution_state = sia.system_state
    else:
        resolution_state = system_intrinsic_information(
            system, specification_measure=specification_measure
        )
    resolved_distinctions = distinctions.resolve_congruence(resolution_state)

    # Compute relations if not provided
    if relations is None:
        relations = compute_relations(resolved_distinctions, **relations_kwargs)

    return CauseEffectStructure(
        sia=sia,
        distinctions=resolved_distinctions,
        relations=relations,
    )
