# new_big_phi/__init__.py

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from textwrap import indent
from typing import Dict, Generator, Iterable, Optional, Union

from numpy.typing import ArrayLike
from toolz import concat

from .. import Direction, Subsystem, compute, config, utils
from ..conf import ConfigurationError, fallback
from ..labels import NodeLabels
from ..metrics.distribution import repertoire_distance
from ..models import cmp, fmt
from ..models.cuts import Cut, SystemPartition
from ..models.subsystem import CauseEffectStructure
from ..partition import directed_bipartition
from ..registry import Registry
from ..relations import Relations
from ..relations import relations as compute_relations
from ..utils import is_positive

DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD = 2 ** 4
DEFAULT_PARTITION_CHUNKSIZE = 2 ** 2 * DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD


##############################################################################
# System state and intrinsic information
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
# System irreducible analysis
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
        # Break ties using negative phi (i.e. reverse order)
        return self.phi

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
            + [("Partition", "")]
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


##############################################################################
# Partition schemes and integration
##############################################################################


class PartitionSchemeRegistry(Registry):
    """Storage for partition schemes registered with PyPhi.

    Users can define custom partitions:

    Examples:
        >>> @pyphi.new_big_phi.partition_types.register('NONE')  # doctest: +SKIP
        ... def no_partitions(mechanism, purview):
        ...    return []

    And use them by setting ``config.IIT_4_SYSTEM_PARTITION_TYPE = 'NONE'``
    """

    desc = "IIT 4.0 system partition schemes"


partition_schemes = PartitionSchemeRegistry()


def system_partitions(
    node_indices: tuple[int], node_labels: NodeLabels
) -> Generator[SystemPartition, None, None]:
    """Generate system partitions."""
    return partition_schemes[config.IIT_4_SYSTEM_PARTITION_TYPE](
        node_indices, node_labels
    )


class SystemPartition:
    """Abstract base class representing a partition of the system."""

    def normalization_factor(self):
        raise NotImplementedError

    def evaluate(
        self, subsystem: Subsystem, system_state: SystemState, **kwargs
    ) -> tuple[float, ArrayLike, ArrayLike]:
        raise NotImplementedError


@dataclass
class HorizontalSystemPartition(SystemPartition):
    """A 'horizontal' system partition."""

    direction: Direction
    purview: tuple[int]
    unpartitioned_mechanism: tuple[int]
    partitioned_mechanism: tuple[int]
    node_labels: Optional[NodeLabels] = None

    def normalization_factor(self):
        return 1 / len(self.purview)

    def evaluate(
        self, subsystem: Subsystem, system_state: SystemState, **kwargs
    ) -> tuple[float, ArrayLike, ArrayLike]:
        valid_distances = ["IIT_4.0_SMALL_PHI", "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE"]
        if config.REPERTOIRE_DISTANCE not in valid_distances:
            raise ValueError(
                f"Must set config.REPERTOIRE_DISTANCE to one of {valid_distances}; "
                f"got {config.REPERTOIRE_DISTANCE}"
            )
        purview_state = utils.state_of(
            # Get purview indices relative to subsystem indices
            [subsystem.node_indices.index(n) for n in self.purview],
            system_state[self.direction],
        )
        unpartitioned_repertoire = subsystem.repertoire(
            self.direction, self.unpartitioned_mechanism, self.purview
        )
        partitioned_repertoire = subsystem.repertoire(
            self.direction,
            self.partitioned_mechanism,
            self.purview,
        )
        phi = repertoire_distance(
            unpartitioned_repertoire,
            partitioned_repertoire,
            direction=self.direction,
            state=purview_state,
            **kwargs,
        )
        normalized_phi = phi * self.normalization_factor()
        return (phi, normalized_phi, unpartitioned_repertoire, partitioned_repertoire)

    def __repr__(self):
        purview = "".join(self.node_labels.coerce_to_labels(self.purview))
        unpartitioned_mechanism = "".join(
            self.node_labels.coerce_to_labels(self.unpartitioned_mechanism)
        )
        partitioned_mechanism = (
            "∅"
            if not self.partitioned_mechanism
            else "".join(self.node_labels.coerce_to_labels(self.partitioned_mechanism))
        )
        return f"π({purview}|{unpartitioned_mechanism}) || π({purview}|{partitioned_mechanism}) [{self.direction}]"


# TODO
class CompletePartition(HorizontalSystemPartition):
    """The partition that severs all connections."""

    pass


#     def __init__(self, direction, subsystem, **kwargs):
#         super().__init__(
#             direction,
#             purview=subsystem.node_indices,
#             unpartitioned_mechanism=subsystem.node_indices,

#             # NOTE: Order is important here
#             Part(mechanism=(), purview=tuple(node_indices)),
#             Part(mechanism=tuple(node_indices), purview=()),
#             **kwargs,
#         )

#     def normalization_factor(self):
#         raise NotImplementedError

#     def evaluate(
#         self, subsystem: Subsystem, system_state: SystemState, **kwargs
#     ) -> tuple[float, ArrayLike, ArrayLike]:
#         raise NotImplementedError

#     def __str__(self):
#         return (
#             super(HybridHorizontalSystemPartition, self).__str__()
#             + f" ({self.direction})"
#         )


# TODO
# class AtomicPartition(HorizontalSystemPartition):
#     """The partition that severs all connections between elements (not
#     self-loops).
#     """

#     def __init__(self, direction, node_indices, **kwargs):
#         self.direction = direction
#         super().__init__(
#             direction,
#             *[Part(mechanism=(n,), purview=(n,)) for n in node_indices],
#             **kwargs,
#         )

#     def __str__(self):
#         return (
#             super(HybridHorizontalSystemPartition, self).__str__()
#             + f" ({self.direction})"
#         )


def _cause_normalization_horizontal(partition):
    if not partition.partitioned_mechanism:
        return len(partition.purview)
    return float("inf")


def _effect_normalization_horizontal(partition):
    return len(partition.purview) * (
        len(partition.unpartitioned_mechanism) - len(partition.partitioned_mechanism)
    )


_horizontal_normalizations = {
    Direction.CAUSE: _cause_normalization_horizontal,
    Direction.EFFECT: _effect_normalization_horizontal,
}


def normalization_factor_horizontal(partition, directions=None):
    directions = fallback(directions, Direction.both())
    return 1 / min(
        _horizontal_normalizations[direction](partition) for direction in directions
    )


# TODO use enum?
_EMPTY_SET = "0"
_PART_ONE = "1"
_PART_TWO = "2"
_FULL_SYSTEM = "3"


def code_number_to_part(number, node_indices, part1, part2=None):
    return {
        _EMPTY_SET: (),
        _PART_ONE: part1,
        _PART_TWO: part2,
        _FULL_SYSTEM: node_indices,
    }[number]


def sia_partitions_horizontal(
    node_indices: Iterable,
    node_labels: Optional[NodeLabels] = None,
    directions=None,
    code=None,
) -> Generator[SystemPartition, None, None]:
    """Yield 'horizontal-type' system partitions."""
    code = fallback(code, config.HORIZONTAL_PARTITION_CODE)
    if code[0] != code[2]:
        raise ConfigurationError(
            "Invalid horizontal partition code: purview of unpartitioned and "
            "partitioned repertoires don't match"
        )

    if directions is None:
        directions = Direction.both()

    # Special case for single-element systems
    if len(node_indices) == 1:
        # Complete partition
        for direction in Direction.both():
            yield CompletePartition(direction, node_indices, node_labels=node_labels)
        return

    if _PART_ONE not in code and _PART_TWO not in code:
        for direction in directions:
            yield HorizontalSystemPartition(
                direction=direction,
                purview=node_indices,
                unpartitioned_mechanism=node_indices,
                partitioned_mechanism=(),
            )
        return
    for (part1, part2), direction in product(
        directed_bipartition(node_indices, nontrivial=True), directions
    ):
        if _PART_ONE in code and _PART_TWO in code:
            purview = code_number_to_part(code[0], node_indices, part1, part2=part2)
            unpartitioned_mechanism = code_number_to_part(
                code[1], node_indices, part1, part2=part2
            )
            partitioned_mechanism = code_number_to_part(
                code[3], node_indices, part1, part2=part2
            )
            yield HorizontalSystemPartition(
                direction,
                purview=purview,
                unpartitioned_mechanism=unpartitioned_mechanism,
                partitioned_mechanism=partitioned_mechanism,
                node_labels=node_labels,
            )
        else:
            purview = code_number_to_part(code[0], node_indices, part1)
            unpartitioned_mechanism = code_number_to_part(code[1], node_indices, part1)
            partitioned_mechanism = code_number_to_part(code[2], node_indices, part1)
            yield HorizontalSystemPartition(
                direction,
                purview=purview,
                unpartitioned_mechanism=unpartitioned_mechanism,
                partitioned_mechanism=partitioned_mechanism,
                node_labels=node_labels,
            )


##############################################################################
# Hybrid horizontal
##############################################################################


# def _sia_partitions_hybrid_horizontal_excluding_complete(
#     node_indices: Iterable, node_labels: Optional[NodeLabels] = None
# ) -> Generator[SystemPartition, None, None]:
#     """Yield all system partitions."""
#     # Special case for single-element systems
#     if len(node_indices) == 1:
#         # Complete partition
#         for direction in Direction.both():
#             yield CompletePartition(direction, node_indices, node_labels=node_labels)
#         return

#     for part, direction in product(
#         utils.powerset(
#             node_indices,
#             nonempty=True,
#             max_size=(len(node_indices) - 1),
#         ),
#         Direction.both(),
#     ):
#         # Compare π(part|system) vs π(part|part)
#         # Code 1311
#         yield HybridHorizontalSystemPartition(
#             direction,
#             Part(mechanism=part, purview=part),
#             node_labels=node_labels,
#         )


# def _sia_partitions_hybrid_horizontal_including_complete(
#     node_indices: Iterable, node_labels: Optional[NodeLabels] = None
# ) -> Generator[SystemPartition, None, None]:
#     """Yield all system partitions (including the complete partition)."""
#     for part, direction in product(
#         utils.powerset(
#             node_indices,
#             nonempty=True,
#         ),
#         Direction.both(),
#     ):
#         if len(part) == len(node_indices):
#             # Complete partition
#             yield CompletePartition(direction, node_indices, node_labels=node_labels)
#         else:
#             # Compare π(part|system) vs π(part|part)
#             yield HybridHorizontalSystemPartition(
#                 direction,
#                 Part(mechanism=part, purview=part),
#                 node_labels=node_labels,
#             )


# def sia_partitions_hybrid_horizontal(
#     node_indices: Iterable,
#     node_labels: Optional[NodeLabels] = None,
#     include_complete: bool = False,
# ) -> Generator[SystemPartition, None, None]:
#     if include_complete:
#         yield from _sia_partitions_hybrid_horizontal_including_complete(
#             node_indices, node_labels=node_labels
#         )
#     else:
#         yield from _sia_partitions_hybrid_horizontal_excluding_complete(
#             node_indices, node_labels=node_labels
#         )


# def normalization_factor_hybrid_horizontal(
#     subsystem: Subsystem, partition: SystemPartition
# ) -> float:
#     """Normalize the phi value according to the partition."""
#     if isinstance(partition, CompletePartition):
#         return 1 / len(subsystem)
#     part = partition.purview
#     return 1 / (len(part) * (len(subsystem) - len(part)))


# # def integration_value(
# #     partition: HybridHorizontalSystemPartition,
# #     subsystem: Subsystem,
# #     system_state: SystemState,
# # ) -> tuple[float, ArrayLike, ArrayLike]:
# #     # TODO(4.0) configure repertoire distance
# #     valid_distances = ["IIT_4.0_SMALL_PHI", "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE"]
# #     if config.REPERTOIRE_DISTANCE not in valid_distances:
# #         raise ValueError(
# #             f"Must set config.REPERTOIRE_DISTANCE to one of {valid_distances}; "
# #             f"got {config.REPERTOIRE_DISTANCE}"
# #         )
# #     purview_state = utils.state_of(
# #         # Get purview indices relative to subsystem indices
# #         [subsystem.node_indices.index(n) for n in partition.purview],
# #         system_state[partition.direction],
# #     )
# #     # Compare π(part|system) vs π(part|part)
# #     phi, partitioned_repertoire, repertoire = subsystem.evaluate_partition(
# #         direction=partition.direction,
# #         mechanism=subsystem.node_indices,
# #         purview=partition.purview,
# #         partition=partition,
# #         state=purview_state,
# #         return_unpartitioned_repertoire=True,
# #     )
# #     return phi, partitioned_repertoire, repertoire


# def evaluate_partition_hybrid_horizontal(
#     partition: HybridHorizontalSystemPartition,
#     subsystem: Subsystem,
#     system_state: SystemState,
#     atomic_integration: Optional[Dict[Direction, float]] = None,
# ) -> SystemIrreducibilityAnalysisHybridHorizontal:
#     phi, partitioned_repertoire, repertoire = integration_value(
#         partition, subsystem, system_state
#     )
#     normalized_phi = phi * normalization_factor_hybrid_horizontal(subsystem, partition)
#     return SystemIrreducibilityAnalysisHybridHorizontal(
#         phi=phi,
#         normalized_phi=normalized_phi,
#         partition=partition,
#         repertoire=repertoire,
#         partitioned_repertoire=partitioned_repertoire,
#         system_state=system_state,
#         node_indices=subsystem.node_indices,
#         node_labels=subsystem.node_labels,
#         atomic_integration=atomic_integration,
#     )


# def atomic_integration_value(
#     direction: Direction, subsystem: Subsystem, system_state: SystemState
# ) -> float:
#     """Return the integration value for the atomic partition."""
#     phi, _, _ = integration_value(
#         partition=AtomicPartition(
#             direction, subsystem.node_indices, node_labels=subsystem.node_labels
#         ),
#         subsystem=subsystem,
#         system_state=system_state,
#     )
#     return phi


# def find_mip_hybrid_horizontal(
#     subsystem: Subsystem,
#     parallel: Optional[bool] = None,
#     progress: Optional[bool] = None,
#     check_trivial_reducibility: Optional[bool] = True,
#     chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
#     sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
#     include_complete: bool = False,
# ) -> SystemIrreducibilityAnalysisHybridHorizontal:
#     """Find the minimum information partition of a system."""
#     parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
#     progress = fallback(progress, config.PROGRESS_BARS)

#     # TODO(4.0) implement
#     # if check_trivial_reducibility and is_trivially_reducible(phi_structure):
#     #     return NullSystemIrreducibilityAnalysis()

#     system_state = find_system_state(subsystem)

#     # Compute atomic integration with the atomic partition
#     atomic_integration = {
#         direction: atomic_integration_value(direction, subsystem, system_state)
#         for direction in Direction.both()
#     }

#     partitions = sia_partitions_hybrid_horizontal(
#         node_indices=subsystem.node_indices,
#         node_labels=subsystem.node_labels,
#         include_complete=include_complete,
#     )

#     return compute.parallel.map_reduce(
#         evaluate_partition_hybrid_horizontal,
#         min,
#         partitions,
#         system_state=system_state,
#         subsystem=subsystem,
#         atomic_integration=atomic_integration,
#         chunksize=chunksize,
#         sequential_threshold=sequential_threshold,
#         shortcircuit_value=0.0,
#         parallel=parallel,
#         progress=progress,
#         desc="Evaluating partitions",
#     )


##############################################################################


def find_mip(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    check_trivial_reducibility: Optional[bool] = True,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    version: str = "hybrid_horizontal",
) -> SystemIrreducibilityAnalysis:
    if version == "hybrid_horizontal":
        return find_mip_hybrid_horizontal(
            subsystem=subsystem,
            parallel=parallel,
            progress=progress,
            check_trivial_reducibility=check_trivial_reducibility,
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
        )
    elif version == "vertical":
        return find_mip_vertical(
            subsystem=subsystem,
            parallel=parallel,
            progress=progress,
            check_trivial_reducibility=check_trivial_reducibility,
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
        )
    raise ValueError(f"Unknown version: {version}")


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
    phi: float
    partition: SystemPartition
    distinctions: CauseEffectStructure
    relations: Relations
    cause_state: ArrayLike
    effect_state: ArrayLike


def phi_structure(subsystem: Subsystem) -> PhiStructure:
    """Analyze the irreducible cause-effect structure of a system."""
    mip = find_mip(subsystem)

    cause_state = subsystem.find_maximal_state_under_complete_partition(
        Direction.CAUSE,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )
    effect_state = subsystem.find_maximal_state_under_complete_partition(
        Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )

    distinctions = compute.ces(subsystem)
    relations = compute_relations(subsystem)

    return PhiStructure(
        phi=mip.phi,
        partition=mip.partition,
        distinctions=distinctions,
        relations=relations,
        cause_state=cause_state,
        effect_state=effect_state,
    )
