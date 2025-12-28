# new_big_phi/__init__.py
"""Implements the IIT 4.0 formalism for system-level analysis."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from enum import auto
from enum import unique

from .. import compute
from .. import conf
from .. import connectivity
from .. import utils
from .. import validate
from ..compute.network import reachable_subsystems
from ..conf import config
from ..conf import fallback
from ..data_structures import PyPhiFloat
from ..direction import Direction
from ..labels import NodeLabels
from ..models import cmp
from ..models import fmt
from ..models.cuts import Cut
from ..models.cuts import GeneralKCut
from ..models.cuts import SystemPartition
from ..models.mechanism import RepertoireIrreducibilityAnalysis
from ..models.subsystem import CauseEffectStructure
from ..models.subsystem import SystemStateSpecification
from ..parallel import MapReduce
from ..partition import system_partitions
from ..relations import ConcreteRelations
from ..relations import Relations
from ..relations import relations as compute_relations
from ..subsystem import Subsystem
from ..warnings import warn_about_tie_serialization

##############################################################################
# Information
##############################################################################


# TODO(4.0) refactor
def system_intrinsic_information(
    subsystem: Subsystem,
    repertoire_distance: str | None = None,
    directions: Iterable[Direction] | None = None,
) -> SystemStateSpecification:
    """Return the cause/effect states specified by the system.

    NOTE: Uses ``config.REPERTOIRE_DISTANCE_INFORMATION``.
    NOTE: State ties are arbitrarily broken (for now).
    """
    directions = fallback(directions, Direction.both())
    directions = tuple(directions)
    # TODO move to Direction
    # TODO have validation methods return the validated value
    validate.directions(directions)
    repertoire_distance = fallback(
        repertoire_distance, config.REPERTOIRE_DISTANCE_INFORMATION
    )
    # TODO(ties) deal with ties here
    ii = {
        direction: subsystem.intrinsic_information(
            direction,
            mechanism=subsystem.node_indices,
            purview=subsystem.node_indices,
            repertoire_distance=repertoire_distance,
        )
        for direction in directions
    }
    return SystemStateSpecification(
        cause=ii.get(Direction.CAUSE), effect=ii.get(Direction.EFFECT)
    )


##############################################################################
# Integration
##############################################################################


@dataclass
class SystemIrreducibilityAnalysis(cmp.OrderableByPhi):
    phi: float
    partition: Cut | SystemPartition
    normalized_phi: float = 0
    cause: RepertoireIrreducibilityAnalysis | None = None
    effect: RepertoireIrreducibilityAnalysis | None = None
    system_state: SystemStateSpecification | None = None
    current_state: tuple[int] | None = None
    node_indices: tuple[int] | None = None
    node_labels: NodeLabels | None = None
    reasons: list | None = None

    def __post_init__(self):
        self.phi = PyPhiFloat(self.phi)
        self.normalized_phi = PyPhiFloat(self.normalized_phi)

    _sia_attributes = [
        "phi",
        "partition",
        "normalized_phi",
        "cause",
        "effect",
        "system_state",
        "current_state",
        "node_indices",
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
        if self.node_labels is not None:
            subsystem_label = ",".join(
                self.node_labels.coerce_to_labels(self.node_indices)
            )
        elif self.node_indices is not None:
            subsystem_label = ",".join(map(str, self.node_indices))
        else:
            subsystem_label = None
        columns = (
            [
                ("Subsystem", subsystem_label),
                ("Current state", fmt.state(self.current_state)),
                (f"           {fmt.SMALL_PHI}_s", self.phi),
                (f"Normalized {fmt.SMALL_PHI}_s", self.normalized_phi),
            ]
            + self.system_state._repr_columns()
            + [("#(tied MIPs)", len(self.ties) - 1), ("Partition", "")]
        )
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
        # Remove ties because of circular references
        del dct["_ties"]
        return dct


class NullSystemIrreducibilityAnalysis(SystemIrreducibilityAnalysis):
    def __init__(self, **kwargs):
        super().__init__(
            phi=0,
            partition=None,
            cause=None,
            effect=None,
            **kwargs,
        )

    def _repr_columns(self):
        columns = [
            (
                "Subsystem",
                ",".join(self.node_labels.coerce_to_labels(self.node_indices)),
            ),
            (f"           {fmt.BIG_PHI}", self.phi),
        ]
        if self.system_state is not None:
            columns.append(self.system_state._repr_columns())
        if self.reasons:
            columns.append(("Reasons", ", ".join([r.name for r in self.reasons])))
        return columns


def normalization_factor(partition: Cut | GeneralKCut) -> float:
    if hasattr(partition, "normalization_factor"):
        return partition.normalization_factor()
    return 1 / (len(partition.from_nodes) * len(partition.to_nodes))


def integration_value(
    direction: Direction,
    subsystem: Subsystem,
    partition: Cut,
    system_state: SystemStateSpecification,
    repertoire_distance: str | None = None,
) -> float:
    repertoire_distance = fallback(repertoire_distance, config.REPERTOIRE_DISTANCE)
    cut_subsystem = subsystem.apply_cut(partition)
    # TODO(4.0) deal with proliferation of special cases for GID
    if repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE":
        partitioned_repertoire = cut_subsystem.forward_repertoire(
            direction,
            subsystem.node_indices,
            subsystem.node_indices,
            system_state[direction].state,
        ).squeeze()[system_state[direction].state]
    else:
        partitioned_repertoire = cut_subsystem.repertoire(
            direction, subsystem.node_indices, subsystem.node_indices
        )
    return subsystem.evaluate_partition(
        direction,
        subsystem.node_indices,
        subsystem.node_indices,
        partition,
        partitioned_repertoire=partitioned_repertoire,
        repertoire_distance=repertoire_distance,
        state=system_state[direction],
    )


def evaluate_partition(
    partition: Cut,
    subsystem: Subsystem,
    system_state: SystemStateSpecification,
    repertoire_distance: str = None,
    directions: Iterable[Direction] | None = None,
) -> SystemIrreducibilityAnalysis:
    directions = fallback(directions, Direction.both())
    directions = tuple(directions)
    validate.directions(directions)
    integration = {
        direction: integration_value(
            direction,
            subsystem,
            partition,
            system_state,
            repertoire_distance=repertoire_distance,
        )
        for direction in directions
    }
    phi = min(integration[direction].phi for direction in directions)
    norm = normalization_factor(partition)
    normalized_phi = phi * norm

    return SystemIrreducibilityAnalysis(
        phi=phi,
        normalized_phi=normalized_phi,
        cause=integration.get(Direction.CAUSE),
        effect=integration.get(Direction.EFFECT),
        partition=partition,
        system_state=system_state,
        current_state=subsystem.proper_state,
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )


@unique
class ShortCircuitConditions(Enum):
    NO_VALID_PARTITIONS = auto()
    NO_CAUSE = auto()
    NO_EFFECT = auto()
    NO_SUBSYSTEM = auto()
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
    subsystem: Subsystem,
    repertoire_distance: str | None = None,
    directions: Iterable[Direction] | None = None,
    partition_scheme: str | None = None,
    partitions: Iterable | None = None,
    system_state: SystemStateSpecification | None = None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    partition_scheme = fallback(partition_scheme, config.SYSTEM_PARTITION_TYPE)

    # TODO(4.0): trivial reducibility

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null SIA.
    def _null_sia(**kwargs):
        return NullSystemIrreducibilityAnalysis(
            system_state=system_state,
            node_indices=subsystem.node_indices,
            node_labels=subsystem.node_labels,
            **kwargs,
        )

    if not subsystem:
        # subsystem is empty
        return _null_sia(reasons=[ShortCircuitConditions.NO_SUBSYSTEM])

    if not connectivity.is_strong(subsystem.cm, subsystem.node_indices):
        # subsystem is not strongly connected
        return _null_sia(reasons=[ShortCircuitConditions.NO_STRONG_CONNECTIVITY])

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(subsystem.cut_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not subsystem.cm[subsystem.node_indices][subsystem.node_indices]:
            return _null_sia(reasons=[ShortCircuitConditions.MONAD_WITH_NO_SELFLOOP])
        # Even if the node has a self-loop, we may still define phi to be zero.
        if not config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI:
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
                # Special case for length 1 subsystems so complete partition is included
                return (
                    not connectivity.is_strong(subsystem.apply_cut(partition).proper_cm)
                ) or len(subsystem) == 1

            filter_func = is_disconnecting_partition

        partitions = system_partitions(
            subsystem.node_indices,
            node_labels=subsystem.node_labels,
            partition_scheme=partition_scheme,
            filter_func=filter_func,
        )

    if system_state is None:
        system_state = system_intrinsic_information(subsystem, directions=directions)

    if config.SHORTCIRCUIT_SIA:
        shortcircuit_reasons = _has_no_cause_or_effect(system_state)
        if shortcircuit_reasons:
            return _null_sia(reasons=shortcircuit_reasons)

    default_sia = _null_sia(reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS])

    parallel_kwargs = conf.parallel_kwargs(config.PARALLEL_CUT_EVALUATION, **kwargs)
    sias = MapReduce(
        evaluate_partition,
        partitions,
        map_kwargs=dict(
            subsystem=subsystem,
            system_state=system_state,
            repertoire_distance=repertoire_distance,
            directions=directions,
        ),
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating partitions",
        **parallel_kwargs,
    ).run()

    # Find MIP in one pass, keeping track of ties
    # TODO(ties) refactor into resolve_ties module
    mip_sia = default_sia
    mip_key = (float("inf"), float("-inf"))
    ties = [default_sia]
    for candidate_mip_sia in sias:
        candidate_key = sia_minimization_key(candidate_mip_sia)
        if candidate_key < mip_key:
            mip_sia = candidate_mip_sia
            mip_key = candidate_key
            ties = [mip_sia]
        elif candidate_key == mip_key:
            ties.append(candidate_mip_sia)
    for tied_mip in ties:
        tied_mip.set_ties(ties)

    if config.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA:
        subsystem.clear_caches()

    return mip_sia


_sia = sia


##############################################################################
# Composition
##############################################################################


class PhiStructure(cmp.Orderable):
    _SIA_INHERITED_ATTRIBUTES = ["phi", "partition", "system_state"]

    def __init__(
        self,
        sia: SystemIrreducibilityAnalysis,
        distinctions: CauseEffectStructure,
        relations: Relations,
    ):
        self._sia = sia
        self._distinctions = distinctions
        self._relations = relations

    @property
    def sia(self):
        return self._sia

    @property
    def distinctions(self):
        return self._distinctions

    @property
    def relations(self):
        return self._relations

    @property
    def components(self):
        for distinction in self.distinctions:
            yield distinction
        for relation in self.relations:
            yield relation

    def __getattr__(self, attr):
        if attr in self._SIA_INHERITED_ATTRIBUTES:
            return getattr(self.sia, attr)
        return super().__getattribute__(attr)

    def order_by(self):
        return self.phi

    def __hash__(self):
        return hash((self.distinctions, self.relations))

    def __bool__(self):
        return bool(self.sia)

    def __eq__(self, other):
        return (
            self.sia == other.sia
            and self.distinctions == other.distinctions
            and self.relations == other.relations
        )

    def _repr_columns(self):
        return [
            ("Φ", self.big_phi),
            ("#(distinctions)", len(self.distinctions)),
            ("Σ φ_d", self.sum_phi_distinctions),
            ("#(relations)", len(self.relations)),
            ("Σ φ_r", self.sum_phi_relations),
        ]

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_1)
        body += "\n" + str(self.sia)
        return fmt.box(fmt.center(body))

    @property
    def sum_phi_relations(self):
        return self.relations.sum_phi()

    @property
    def sum_phi_distinctions(self):
        try:
            return self._sum_phi_distinctions
        except AttributeError:
            # TODO delegate sum to distinctions
            self._sum_phi_distinctions = self.distinctions.sum_phi()
            return self._sum_phi_distinctions

    @property
    def big_phi(self):
        try:
            return self._big_phi
        except AttributeError:
            self._big_phi = self.sum_phi_distinctions + self.sum_phi_relations
            return self._big_phi

    def to_json(self):
        return dict(
            sia=self.sia, distinctions=self.distinctions, relations=self.relations
        )


class NullPhiStructure(PhiStructure):
    def __init__(self, **kwargs):
        super().__init__(
            sia=NullSystemIrreducibilityAnalysis(),
            distinctions=CauseEffectStructure([]),
            relations=ConcreteRelations([]),
            **kwargs,
        )


def phi_structure(
    subsystem: Subsystem,
    sia: SystemIrreducibilityAnalysis = None,
    distinctions: CauseEffectStructure = None,
    relations: Relations = None,
    sia_kwargs: dict = None,
    ces_kwargs: dict = None,
    relations_kwargs: dict = None,
) -> PhiStructure:
    """Analyze the irreducible cause-effect structure of a system."""
    sia_kwargs = sia_kwargs or dict()
    ces_kwargs = ces_kwargs or dict()
    relations_kwargs = relations_kwargs or dict()

    # Analyze irreducibility if not provided
    if sia is None:
        sia = _sia(subsystem, **sia_kwargs)

    # Compute distinctions if not provided
    if distinctions is None:
        distinctions = compute.ces(subsystem, **ces_kwargs)
    # Filter out incongruent distinctions
    if sia.system_state is not None:
        distinctions = distinctions.resolve_congruence(sia.system_state)

    # Compute relations if not provided
    if relations is None:
        relations = compute_relations(distinctions, **relations_kwargs)

    return PhiStructure(
        sia=sia,
        distinctions=distinctions,
        relations=relations,
    )


def all_complexes(network, state, **kwargs):
    """Yield SIAs for all subsystems of the network."""
    # TODO(4.0) parallelize
    for subsystem in reachable_subsystems(network, network.node_indices, state):
        yield sia(subsystem, **kwargs)


def irreducible_complexes(network, state, **kwargs):
    """Yield SIAs for irreducible subsystems of the network."""
    # TODO(4.0) parallelize
    yield from filter(None, all_complexes(network, state, **kwargs))


def maximal_complex(network, state, **kwargs):
    # TODO(4.0) parallelize
    return max(
        irreducible_complexes(network, state, **kwargs), default=NullPhiStructure()
    )
