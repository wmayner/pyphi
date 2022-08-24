# new_big_phi/__init__.py

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto, unique
from textwrap import indent
from typing import Dict, Iterable, Optional, Union

from numpy.typing import ArrayLike
from toolz import concat

from .. import Direction, Subsystem, compute, config, connectivity
from ..conf import fallback
from ..labels import NodeLabels
from ..metrics.distribution import repertoire_distance as _repertoire_distance
from ..models import cmp, fmt
from ..models.cuts import Cut, GeneralKCut, SystemPartition
from ..models.subsystem import CauseEffectStructure
from ..partition import system_partitions
from ..relations import Relations
from ..relations import relations as compute_relations
from ..utils import is_positive

DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD = 2 ** 4
DEFAULT_PARTITION_CHUNKSIZE = 2 ** 2 * DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD


##############################################################################
# Information
##############################################################################


@dataclass
class SystemState:
    cause: tuple
    effect: tuple
    intrinsic_information: Dict[Direction, float]

    def __getitem__(self, direction: Direction) -> tuple:
        if direction == Direction.CAUSE:
            return self.cause
        elif direction == Direction.EFFECT:
            return self.effect
        raise KeyError("Invalid direction")

    def _repr_columns(self, prefix=""):
        return list(
            concat(
                [
                    [
                        (
                            f"{prefix}{direction}",
                            str(self[direction]),
                        ),
                        (
                            f"{prefix}II_{str(direction)[:1].lower()}",
                            self.intrinsic_information[direction],
                        ),
                    ]
                    for direction in Direction.both()
                ]
            )
        )

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header("System state", body, under_char=fmt.HEADER_BAR_3)
        return fmt.box(fmt.center(body))


def find_system_state(
    subsystem: Subsystem,
    repertoire_distance: Optional[str] = None,
    system_state: Optional[SystemState] = None,
) -> SystemState:
    """Return the cause/effect states specified by the system.

    NOTE: Uses ``config.REPERTOIRE_DISTANCE_INFORMATION``.
    NOTE: State ties are arbitrarily broken (for now).
    """
    repertoire_distance = fallback(
        repertoire_distance, config.REPERTOIRE_DISTANCE_INFORMATION
    )

    if system_state is None:
        cause_states = None
        effect_states = None
    else:
        cause_states = [system_state[Direction.CAUSE]]
        effect_states = [system_state[Direction.EFFECT]]

    cause_states, ii_cause = subsystem.find_maximal_state_under_complete_partition(
        Direction.CAUSE,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
        repertoire_distance=repertoire_distance,
        return_information=True,
        states=cause_states,
    )
    effect_states, ii_effect = subsystem.find_maximal_state_under_complete_partition(
        Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
        repertoire_distance=repertoire_distance,
        return_information=True,
        states=effect_states,
    )
    return SystemState(
        # NOTE: tie-breaking happens here
        cause=cause_states[0],
        effect=effect_states[0],
        intrinsic_information={Direction.CAUSE: ii_cause, Direction.EFFECT: ii_effect},
    )


##############################################################################
# Integration
##############################################################################


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    phi: float
    partition: Union[Cut, SystemPartition]
    normalized_phi: float = 0
    phi_cause: float = 0
    phi_effect: float = 0
    repertoire_cause: Optional[ArrayLike] = None
    partitioned_repertoire_cause: Optional[ArrayLike] = None
    repertoire_effect: Optional[ArrayLike] = None
    partitioned_repertoire_effect: Optional[ArrayLike] = None
    atomic_integration: Optional[Dict[Direction, float]] = None
    system_state: Optional[SystemState] = None
    current_state: Optional[tuple[int]] = None
    node_indices: Optional[tuple[int]] = None
    node_labels: Optional[NodeLabels] = None
    reasons: Optional[list] = None

    _sia_attributes = [
        "phi",
        "normalized_phi",
        "partition",
        "repertoire",
        "partitioned_repertoire",
        "system_state",
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
        return is_positive(self.phi)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.partition,
            )
        )

    def _repr_columns(self):
        columns = (
            [
                (
                    "Subsystem",
                    ",".join(self.node_labels.coerce_to_labels(self.node_indices)),
                ),
                ("Current state", ",".join(map(str, self.current_state))),
                (f"           {fmt.BIG_PHI}", self.phi),
                (f"Normalized {fmt.BIG_PHI}", self.normalized_phi),
            ]
            + self.system_state._repr_columns()
            + [("# tied MIPs", len(self.ties)), ("Partition", "")]
        )
        if self.reasons:
            columns.append(("Reasons", ", ".join(self.reasons)))
        return columns

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_2)
        body = fmt.center(body)
        column_extent = body.split("\n")[2].index(":")
        body += "\n" + indent(str(self.partition), " " * (column_extent + 2))
        return fmt.box(body)


class NullSystemIrreducibilityAnalysis(SystemIrreducibilityAnalysis):
    def __init__(self, **kwargs):
        super().__init__(
            phi=0,
            partition=None,
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
        if self.reasons:
            columns.append(("Reasons", ", ".join([r.name for r in self.reasons])))
        return columns


def normalization_factor(partition: Union[Cut, GeneralKCut]) -> float:
    if hasattr(partition, "normalization_factor"):
        return partition.normalization_factor()
    return 1 / (len(partition.from_nodes) * len(partition.to_nodes))


def integration_value(
    direction: Direction,
    subsystem: Subsystem,
    cut_subsystem: Subsystem,
    system_state: SystemState,
    repertoire_distance: str = None,
) -> float:
    unpartitioned_repertoire = subsystem.repertoire(
        direction, subsystem.node_indices, subsystem.node_indices
    )
    partitioned_repertoire = cut_subsystem.repertoire(
        direction, subsystem.node_indices, subsystem.node_indices
    )
    return (
        _repertoire_distance(
            unpartitioned_repertoire,
            partitioned_repertoire,
            repertoire_distance=repertoire_distance,
            state=system_state[direction],
        ),
        unpartitioned_repertoire,
        partitioned_repertoire,
    )


def evaluate_partition(
    partition: Cut,
    subsystem: Subsystem,
    system_state: SystemState,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
) -> SystemIrreducibilityAnalysis:
    directions = fallback(directions, Direction.both())

    cut_subsystem = subsystem.apply_cut(partition)

    integration = {
        direction: integration_value(
            direction,
            subsystem,
            cut_subsystem,
            system_state,
            repertoire_distance=repertoire_distance,
        )
        for direction in Direction.both()
    }
    phi_c, repertoire_c, partitioned_repertoire_c = integration[Direction.CAUSE]
    phi_e, repertoire_e, partitioned_repertoire_e = integration[Direction.EFFECT]

    phi = min(integration[direction][0] for direction in directions)
    norm = normalization_factor(partition)
    normalized_phi = phi * norm

    return SystemIrreducibilityAnalysis(
        phi=phi,
        normalized_phi=normalized_phi,
        phi_cause=phi_c,
        phi_effect=phi_e,
        partition=partition,
        repertoire_cause=repertoire_c,
        partitioned_repertoire_cause=partitioned_repertoire_c,
        repertoire_effect=repertoire_e,
        partitioned_repertoire_effect=partitioned_repertoire_e,
        system_state=system_state,
        current_state=subsystem.proper_state,
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )


@unique
class ShortCircuitConditions(Enum):
    NO_VALID_PARTITIONS = auto()


def sia_minimization_key(sia):
    """Return a sorting key that minimizes the normalized phi value.

    Ties are broken by maximizing the phi value."""
    return (sia.normalized_phi, -sia.phi)


def find_mip(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
    partitions: str = None,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)
    partitions = fallback(partitions, config.SYSTEM_PARTITION_TYPE)
    # TODO: trivial reducibility

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
        partition_scheme=partitions,
        filter_func=filter_func,
    )

    system_state = find_system_state(subsystem)

    default_sia = NullSystemIrreducibilityAnalysis(
        reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS],
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )
    sias = compute.parallel.map(
        evaluate_partition,
        partitions,
        subsystem=subsystem,
        system_state=system_state,
        repertoire_distance=repertoire_distance,
        directions=directions,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        shortcircuit_value=0.0,
        parallel=parallel,
        progress=progress,
        desc="Evaluating partitions",
    )
    # Find MIP in one pass, keeping track of ties
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
    return mip_sia


##############################################################################
# Composition
##############################################################################


def congruent_subset(distinctions, direction, state):
    """Find the subset(s) of distinctions congruent with the given system state(s)."""
    # Map states to congruent subsets of distinctions
    tied_states = defaultdict(set)
    for distinction in distinctions:
        for state in tied_states:
            if distinction.mice(direction).is_congruent(state):
                tied_states[state].add(distinction)
    # TODO HERE finish


@dataclass
class PhiStructure(cmp.Orderable):
    sia: SystemIrreducibilityAnalysis
    distinctions: CauseEffectStructure
    relations: Relations

    _SIA_INHERITED_ATTRIBUTES = ["phi", "partition", "system_state"]

    def __getattr__(self, attr):
        if attr in self._SIA_INHERITED_ATTRIBUTES:
            return getattr(self.sia, attr)
        return super().__getattr__(attr)

    @property
    def big_phi(self):
        try:
            return self._big_phi
        except AttributeError:
            self._big_phi = sum(self.distinctions.phis) + self.relations.sum_phis()


def phi_structure(subsystem: Subsystem) -> PhiStructure:
    """Analyze the irreducible cause-effect structure of a system."""
    mip = find_mip(subsystem)
    distinctions = compute.ces(subsystem)
    relations = compute_relations(subsystem, distinctions)
    return PhiStructure(
        sia=mip,
        distinctions=distinctions,
        relations=relations,
    )
