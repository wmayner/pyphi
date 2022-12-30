# new_big_phi/min_max_horizontal.py

from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple, Union

from numpy.typing import ArrayLike

from .. import Direction, Subsystem, config, utils
from ..compute.parallel import MapReduce
from ..conf import fallback
from ..labels import NodeLabels
from ..metrics.distribution import repertoire_distance as _repertoire_distance
from ..models import cmp, fmt
from ..models.cuts import Cut, SystemPartition
from ..new_big_phi import (
    DEFAULT_PARTITION_CHUNKSIZE,
    DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    NullSystemIrreducibilityAnalysis,
    ShortCircuitConditions,
    SystemStateSpecification,
    system_intrinsic_information,
)
from ..registry import Registry

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
    node_indices: Tuple[int], node_labels: NodeLabels
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
        self, subsystem: Subsystem, system_state: SystemStateSpecification, **kwargs
    ) -> Tuple[float, ArrayLike, ArrayLike]:
        raise NotImplementedError


@dataclass
class HorizontalSystemPartition(SystemPartition):
    """A 'horizontal' system partition."""

    direction: Direction
    purview: Tuple[int]
    unpartitioned_mechanism: Tuple[int]
    partitioned_mechanism: Tuple[int]
    node_labels: Optional[NodeLabels] = None

    def normalization_factor(self):
        return 1 / len(self.purview)

    def evaluate(
        self, subsystem: Subsystem, system_state: SystemStateSpecification, **kwargs
    ) -> Tuple[float, ArrayLike, ArrayLike]:
        valid_distances = ["IIT_4.0_SMALL_PHI", "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE"]
        if config.REPERTOIRE_DISTANCE not in valid_distances:
            raise ValueError(
                f"Must set config.REPERTOIRE_DISTANCE to one of {valid_distances}; "
                f"got {config.REPERTOIRE_DISTANCE}"
            )
        purview_state = utils.state_of_subsystem_nodes(
            subsystem, self.purview, system_state[self.direction]
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


HORIZONTAL_PARTITION_CODE = "1210"


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    phi: float
    normalized_phi: float
    partition: Union[Cut, SystemPartition]
    maximal_purview: Tuple[int]
    repertoire: ArrayLike
    partitioned_repertoire: ArrayLike
    system_state: Optional[Tuple[int]] = None
    node_indices: Optional[Tuple[int]] = None
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
        # TODO deal with exclusion maximization here when we get to that
        return (self.normalized_phi, self.phi)

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

    def __repr__(self):
        body = "\n".join(
            fmt.align_columns(
                [
                    (
                        "Subsystem",
                        ",".join(self.node_labels.coerce_to_labels(self.node_indices)),
                    ),
                    ("Partition", str(self.partition)),
                    (
                        "Maximal purview",
                        "".join(
                            self.node_labels.coerce_to_labels(self.maximal_purview)
                        ),
                    ),
                    (f"           {fmt.BIG_PHI}", self.phi),
                    (f"Normalized {fmt.BIG_PHI}", self.normalized_phi),
                    (str(Direction.CAUSE), str(self.system_state[Direction.CAUSE])),
                    ("II_c", self.system_state.intrinsic_information[Direction.CAUSE]),
                    # (
                    #     f"Atomic {fmt.BIG_PHI}_c",
                    #     self.atomic_integration[Direction.CAUSE],
                    # ),
                    (str(Direction.EFFECT), str(self.system_state[Direction.EFFECT])),
                    ("II_e", self.system_state.intrinsic_information[Direction.EFFECT]),
                    # (
                    #     f"Atomic {fmt.BIG_PHI}_e",
                    #     self.atomic_integration[Direction.EFFECT],
                    # ),
                ]
            )
        )
        body = fmt.header(
            "System irreducibility analysis", body, under_char=fmt.HEADER_BAR_2
        )
        return fmt.box(fmt.center(body))


def evaluate_purview(purview, partition, subsystem, system_state, **kwargs):
    purview_state = utils.state_of(
        # Get purview indices relative to subsystem indices
        [subsystem.node_indices.index(n) for n in purview],
        system_state[partition.direction],
    )
    unpartitioned_repertoire = subsystem.repertoire(
        partition.direction,
        partition.unpartitioned_mechanism,
        purview,
    )
    partitioned_repertoire = subsystem.repertoire(
        partition.direction,
        partition.partitioned_mechanism,
        purview,
    )
    phi = _repertoire_distance(
        unpartitioned_repertoire,
        partitioned_repertoire,
        direction=partition.direction,
        state=purview_state,
        **kwargs,
    )
    normalized_phi = phi * normalization_factor_horizontal(partition)
    return SystemIrreducibilityAnalysis(
        phi=phi,
        normalized_phi=normalized_phi,
        partition=partition,
        maximal_purview=purview,
        repertoire=unpartitioned_repertoire,
        partitioned_repertoire=partitioned_repertoire,
        system_state=system_state,
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )


def evaluate_partition(
    partition: HorizontalSystemPartition,
    subsystem: Subsystem,
    system_state: SystemStateSpecification,
    repertoire_distance: str = None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    valid_distances = [
        "IIT_4.0_SMALL_PHI",
        "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
        "APMI",
    ]
    if config.REPERTOIRE_DISTANCE not in valid_distances:
        raise ValueError(
            f"Must set config.REPERTOIRE_DISTANCE to one of {valid_distances}; "
            f"got {config.REPERTOIRE_DISTANCE}"
        )
    # TODO parallel args?
    return evaluate_purview(
        partition.purview,
        partition=partition,
        subsystem=subsystem,
        system_state=system_state,
        repertoire_distance=repertoire_distance,
        **kwargs,
    )


def sia(
    subsystem: Subsystem,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
    code: str = HORIZONTAL_PARTITION_CODE,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    kwargs = {
        "parallel": config.PARALLEL_CUT_EVALUATION,
        "progress": config.PROGRESS_BARS,
        "chunksize": DEFAULT_PARTITION_CHUNKSIZE,
        "sequential_threshold": DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
        **kwargs,
    }

    # TODO: trivial reducibility

    system_state = system_intrinsic_information(subsystem)

    # Find MIP
    partitions = sia_partitions_horizontal(
        subsystem.node_indices,
        node_labels=subsystem.node_labels,
        code=code,
        directions=directions,
    )

    default_sia = NullSystemIrreducibilityAnalysis(
        reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS],
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )

    min_sia = MapReduce(
        evaluate_partition,
        partitions,
        map_kwargs=dict(
            subsystem=subsystem,
            system_state=system_state,
            repertoire_distance=repertoire_distance,
            directions=directions,
        ),
        reduce_func=min,
        reduce_kwargs=dict(default=default_sia),
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating partitions",
        **kwargs,
    ).run()

    # Find maximal purview for the minimum information partition
    purviews = utils.powerset(min_sia.partition.purview, nonempty=True)
    return MapReduce(
        evaluate_purview,
        purviews,
        reduce_func=max,
        map_kwargs=dict(
            partition=min_sia.partition,
            subsystem=subsystem,
            system_state=system_state,
            repertoire_distance=repertoire_distance,
        ),
        desc="Evaluating purviews",
        **kwargs,
    ).run()
