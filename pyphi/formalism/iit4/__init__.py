# formalism/iit4/__init__.py
"""IIT 4.0 system-level analysis: SIA, distinctions, relations, Φ-structure.

Implements the algorithms from Albantakis et al. 2023 (and the 2026 extension
when configured via the ``IIT_4_0_2026`` formalism). Concrete formalism
classes wrapping these algorithms live in :mod:`pyphi.formalism.iit4.formalism`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
from enum import auto
from enum import unique
from typing import ClassVar

from pyphi import conf
from pyphi import connectivity
from pyphi import metrics
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
from pyphi.models import cmp
from pyphi.models import fmt
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.cuts import GeneralKCut
from pyphi.models.cuts import NullCut
from pyphi.models.cuts import SystemPartition
from pyphi.models.distinctions import Distinctions
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.models.state_specification import StateSpecification
from pyphi.models.state_specification import SystemStateSpecification
from pyphi.parallel import MapReduce
from pyphi.partition import system_partitions
from pyphi.relations import ConcreteRelations
from pyphi.relations import Relations
from pyphi.relations import relations as compute_relations
from pyphi.substrate import reachable_systems
from pyphi.system import System
from pyphi.warnings import warn_about_tie_serialization

##############################################################################
# Information
##############################################################################


# TODO(4.0) refactor
def system_intrinsic_information(
    system: System,
    repertoire_distance: str | None = None,
    directions: Iterable[Direction] | None = None,
) -> SystemStateSpecification:
    """Return the cause/effect states specified by the system.

    NOTE: Uses ``config.formalism.iit.specification_measure``.
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
    repertoire_distance = fallback(
        repertoire_distance,
        config.formalism.iit.specification_measure,  # pyright: ignore[reportAttributeAccessIssue]
    )
    # TODO(ties) deal with ties here
    ii = {
        direction: system.intrinsic_information(
            direction,
            mechanism=system.node_indices,
            purview=system.node_indices,
            repertoire_distance=repertoire_distance,
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

    ``phi`` is the paper-faithful, non-negative integrated information value
    (Eqs. 19-20 with the ``|·|+`` operator applied). ``signed_phi`` is the
    raw value before clamping — when negative, it indicates "preventative
    cause" structure that would otherwise be invisible. Constructors pass
    the *signed* value as ``phi``; ``__post_init__`` clamps it and stores
    the raw value in ``signed_phi``. ``normalized_phi`` and
    ``signed_normalized_phi`` follow the same pattern.
    """

    phi: float | DistanceResult
    partition: SystemPartition | SystemPartition | NullCut
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
        # Apply the |·|+ operator to surface the paper-faithful value.
        clamped_phi = utils.positive_part(self.signed_phi)
        clamped_normalized = utils.positive_part(self.signed_normalized_phi)
        if not isinstance(self.phi, DistanceResult):
            self.phi = PyPhiFloat(clamped_phi)
        else:
            # Preserve metadata-bearing DistanceResult while clamping the
            # numeric value. PyPhi's metric machinery never produces a
            # DistanceResult with negative signed phi today, but the
            # contract is explicit.
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
        consumers (e.g., congruence filtering in phi_structure) see the correct
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
        warn_about_tie_serialization(self.__class__.__name__, serialize=True)
        dct = self.__dict__.copy()
        # TODO(ties) implement serialization of ties
        # Remove ties because of circular references (if present)
        dct.pop("_ties", None)
        return dct


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


def normalization_factor(partition: SystemPartition | GeneralKCut) -> float:
    if hasattr(partition, "normalization_factor"):
        return partition.normalization_factor()  # pyright: ignore[reportAttributeAccessIssue]
    # For GeneralKCut, we need to check hasattr before accessing attributes
    if hasattr(partition, "from_nodes") and hasattr(partition, "to_nodes"):
        return 1 / (len(partition.from_nodes) * len(partition.to_nodes))  # pyright: ignore[reportAttributeAccessIssue]
    # Default fallback
    return 1.0


def _integration_value_for_state(
    direction: Direction,
    system: System,
    cut_system: System,
    partition: SystemPartition,
    specified: StateSpecification,
    repertoire_distance: str,
) -> RepertoireIrreducibilityAnalysis:
    """Compute the integration value for a single specified state."""
    mechanism = purview = system.node_indices
    if repertoire_distance in [
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
    ]:
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
        partition,  # pyright: ignore[reportArgumentType] - SystemPartition passed to Bipartition param in IIT 4.0
        partitioned_repertoire=partitioned_repertoire,
        repertoire_distance=repertoire_distance,
        state=specified,
    )


def integration_value(
    direction: Direction,
    system: System,
    partition: SystemPartition,
    system_state: SystemStateSpecification,
    repertoire_distance: str | None = None,
) -> RepertoireIrreducibilityAnalysis:
    repertoire_distance = fallback(
        repertoire_distance, config.formalism.iit.system_phi_measure
    )
    cut_system = system.apply_cut(partition)
    specified = system_state[direction]
    tied_specs = specified.ties if specified.ties else (specified,)
    # When there are tied specified states, evaluate all of them and take the
    # minimum integration (the "cruelest cut"): among equally-specified states,
    # the partition should be evaluated against the one it hurts most.
    best_ria = None
    for spec in tied_specs:
        ria = _integration_value_for_state(
            direction,
            system,
            cut_system,
            partition,
            spec,
            repertoire_distance,  # pyright: ignore[reportArgumentType]
        )
        # Cruelest-cut convention (PyPhi-specific, not paper-mandated):
        # among tied specified states, prefer the one with the lowest
        # *signed* phi — i.e., the most preventative state pin. We
        # compare on ``signed_phi`` (raw) rather than ``phi`` (clamped)
        # so that all-negative tied pins resolve to the most-preventative
        # one rather than first-encountered.
        if best_ria is None or ria.signed_phi < best_ria.signed_phi:
            best_ria = ria
    return best_ria  # pyright: ignore[reportReturnType]


def intrinsic_differentiation_value(
    direction: Direction,
    system: System,
    partition: SystemPartition,  # noqa: ARG001
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
    partition: SystemPartition,
    system: System,
    system_state: SystemStateSpecification,
    repertoire_distance: str | None = None,
    directions: Iterable[Direction] | None = None,
) -> SystemIrreducibilityAnalysis:
    directions = fallback(directions, Direction.both())
    if directions is None:
        directions = Direction.both()
    directions = tuple(directions)
    validate.directions(directions)

    # Eqs. 19-20: system-level partition integration uses GID only.
    # The ii(s) cap (Eq. 23) is applied separately below.
    effective_distance = fallback(
        repertoire_distance, config.formalism.iit.system_phi_measure
    )
    partition_distance = (
        "GENERALIZED_INTRINSIC_DIFFERENCE"
        if effective_distance == "INTRINSIC_INFORMATION"
        else effective_distance
    )

    integration = {
        direction: integration_value(
            direction,
            system,
            partition,
            system_state,
            repertoire_distance=partition_distance,
        )
        for direction in directions
    }

    intrinsic_differentiation = {
        direction: intrinsic_differentiation_value(
            direction,
            system,
            partition,
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
    if effective_distance == "INTRINSIC_INFORMATION":
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


def sia_minimization_key(sia):
    """Return a sorting key that minimizes the normalized phi value.

    Ties are broken by maximizing the phi value."""
    return (sia.normalized_phi, -sia.phi)


def sia(
    system: System,
    repertoire_distance: str | None = None,
    directions: Iterable[Direction] | None = None,
    partition_scheme: str | None = None,
    partitions: Iterable | None = None,
    system_state: SystemStateSpecification | None = None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
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
    if len(system.cut_indices) == 1:
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
        system_state = system_intrinsic_information(system, directions=directions)

    if config.formalism.iit.shortcircuit_sia:
        shortcircuit_reasons = _has_no_cause_or_effect(system_state)
        if shortcircuit_reasons:
            return _null_sia(reasons=shortcircuit_reasons)

    default_sia = _null_sia(reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS])

    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_cut_evaluation), **kwargs
    )
    sias = MapReduce(
        evaluate_partition,
        partitions,
        map_kwargs={
            "system": system,
            "system_state": system_state,
            "repertoire_distance": repertoire_distance,
            "directions": directions,
        },
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating partitions",
        **parallel_kwargs,
    ).run()

    # Find MIP in one pass, keeping track of ties
    # TODO(ties) refactor into resolve_ties module
    mip_sia = default_sia
    mip_key = (float("inf"), float("-inf"))
    ties = [default_sia]
    if sias is None:
        sias = []
    for candidate_mip_sia in sias:
        candidate_key = sia_minimization_key(candidate_mip_sia)
        if candidate_key < mip_key:
            mip_sia = candidate_mip_sia
            mip_key = candidate_key
            ties = [mip_sia]
        elif candidate_key == mip_key:
            ties.append(candidate_mip_sia)
    for tied_mip in ties:
        tied_mip.resolve_system_state()
        tied_mip.set_ties(ties)

    if config.infrastructure.clear_system_caches_after_computing_sia:
        system.clear_caches()

    return mip_sia


_sia = sia


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


def phi_structure(
    system: System,
    sia: SystemIrreducibilityAnalysis | None = None,
    distinctions: Distinctions | None = None,
    relations: Relations | None = None,
    sia_kwargs: dict | None = None,
    ces_kwargs: dict | None = None,
    relations_kwargs: dict | None = None,
) -> CauseEffectStructure:
    """Analyze the irreducible cause-effect structure of a system."""
    sia_kwargs = sia_kwargs or {}
    ces_kwargs = ces_kwargs or {}
    relations_kwargs = relations_kwargs or {}

    # Analyze irreducibility if not provided
    if sia is None:
        sia = _sia(system, **sia_kwargs)

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
        resolution_state = system_intrinsic_information(system)
    resolved_distinctions = distinctions.resolve_congruence(resolution_state)

    # Compute relations if not provided
    if relations is None:
        relations = compute_relations(resolved_distinctions, **relations_kwargs)

    return CauseEffectStructure(
        sia=sia,
        distinctions=resolved_distinctions,
        relations=relations,
    )


def all_complexes(substrate, state, parallel_kwargs=None, **kwargs):
    """Return SIAs for all systems of the substrate.

    Dispatches through :class:`pyphi.parallel.MapReduce` using the
    ``parallel_complex_evaluation`` config; pass ``parallel_kwargs`` to
    override per-call.
    """
    pkwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_complex_evaluation),
        **(parallel_kwargs or {}),
    )
    systems = list(reachable_systems(substrate, substrate.node_indices, state))
    return MapReduce(
        sia,
        systems,
        map_kwargs={"progress": False, **kwargs},
        desc="Evaluating complexes",
        **pkwargs,
    ).run()


def irreducible_complexes(substrate, state, complexes=None, **kwargs):
    """Yield SIAs for irreducible systems of the substrate."""
    if complexes is None:
        complexes = all_complexes(substrate, state, **kwargs)
    yield from filter(None, complexes)


def maximal_complex(substrate, state, complexes=None, **kwargs):
    return max(
        irreducible_complexes(substrate, state, complexes=complexes, **kwargs),
        default=NullCauseEffectStructure(),
    )
