# new_big_phi.py

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Dict, Generator, Iterable, Optional

from numpy.typing import ArrayLike

from pyphi.labels import NodeLabels
from pyphi.models.subsystem import CauseEffectStructure

from . import Direction, Subsystem, compute, config, metrics, utils
from .conf import fallback
from .models import cmp, fmt
from .models.cuts import CompleteSystemPartition, KPartition, Part, SystemPartition
from .partition import system_partition_types, complete_partition

# TODO change SystemPartition
from .relations import Relations
from .relations import relations as compute_relations

DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD = 256
DEFAULT_PARTITION_CHUNKSIZE = 4 * DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    phi: float
    normalized_phi: float
    partition: SystemPartition
    repertoire: ArrayLike
    partitioned_repertoire: ArrayLike
    system_state: Optional[tuple[int]] = None
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
        return self.normalized_phi

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


class NullSystemIrreducibilityAnalysis(SystemIrreducibilityAnalysis):
    def __init__(self):
        super().__init__(
            partition=None,
            phi=0.0,
            reasons=[],
        )


##############################################################################
# Vertical
##############################################################################


class SystemIrreducibilityAnalysisVertical(SystemIrreducibilityAnalysis):
    def __repr__(self):
        body = str(self.partition)
        body = (
            "\n".join(
                fmt.align_columns(
                    [
                        (fmt.BIG_PHI, self.phi),
                    ]
                )
            )
            + "\n"
            + body
        )
        body = fmt.header(
            "System irreducibility analysis", body, under_char=fmt.HEADER_BAR_2
        )
        return fmt.box(fmt.center(body))


def normalization_factor_vertical(
    subsystem: Subsystem, partition: SystemPartition
) -> float:
    """Normalize the phi value according to the partition."""
    smallest_part_size = min(len(partition.from_nodes), len(partition.to_nodes))
    return len(subsystem) / smallest_part_size


def integration_value_vertical(
    subsystem: Subsystem,
    partition: SystemPartition,
    unpartitioned_repertoire: ArrayLike,
    partitioned_repertoire: ArrayLike,
) -> float:
    """Return the (normalized) integration value of a partition and associated repertoires."""
    kld = metrics.distribution.kld(unpartitioned_repertoire, partitioned_repertoire)
    norm = normalization_factor_vertical(subsystem, partition)
    return kld * norm


def evaluate_partition_vertical(
    partition: SystemPartition, subsystem: Subsystem
) -> SystemIrreducibilityAnalysisVertical:
    mechanism = partition.from_nodes
    purview = partition.to_nodes
    unpartitioned_system_repertoire = subsystem.repertoire(
        partition.direction, mechanism, purview
    )
    partitioned_system_repertoire = subsystem.unconstrained_repertoire(
        partition.direction, purview
    )
    phi = integration_value_vertical(
        subsystem,
        partition,
        unpartitioned_system_repertoire,
        partitioned_system_repertoire,
    )
    # TODO(4.0) configure repertoire distance
    # TODO separate normalization from integration
    return SystemIrreducibilityAnalysisVertical(
        phi=phi,
        partition=partition,
        repertoire=unpartitioned_system_repertoire,
        partitioned_repertoire=partitioned_system_repertoire,
    )


def sia_partitions_vertical(
    node_indices: Iterable, node_labels: Optional[NodeLabels] = None
) -> Generator[SystemPartition, None, None]:
    """Yield all system partitions."""
    # TODO(4.0) consolidate 3.0 and 4.0 cuts
    scheme = config.SYSTEM_PARTITION_TYPE
    valid = ["TEMPORAL_DIRECTED_BI", "TEMPORAL_DIRECTED_BI_CUT_ONE"]
    if scheme not in valid:
        raise ValueError(
            "IIT 4.0 calculations must use one of the following system"
            f"partition schemes: {valid}; got {scheme}"
        )
    # Special case for single-element systems
    if len(node_indices) == 1:
        yield CompleteSystemPartition()
    else:
        yield from system_partition_types[config.SYSTEM_PARTITION_TYPE](
            node_indices, node_labels=node_labels
        )


def find_mip_vertical(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    check_trivial_reducibility: Optional[bool] = True,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
) -> SystemIrreducibilityAnalysisVertical:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)

    if len(subsystem) == 1:
        raise ValueError("Phi of single-node system is not defined yet")

    # TODO(4.0) implement
    # if check_trivial_reducibility and is_trivially_reducible(phi_structure):
    #     return NullSystemIrreducibilityAnalysis()

    partitions = sia_partitions_vertical(subsystem.node_indices, subsystem.node_labels)

    return compute.parallel.map_reduce(
        evaluate_partition_vertical,
        min,
        partitions,
        subsystem=subsystem,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        shortcircuit_value=0.0,
        parallel=parallel,
        progress=progress,
        desc="Evaluating partitions",
    )


##############################################################################
# Hybrid horizontal
##############################################################################


class SystemIrreducibilityAnalysisHybridHorizontal(SystemIrreducibilityAnalysis):
    def __repr__(self):
        body = ""
        for line in reversed(
            [
                f"Subsystem: {','.join(self.node_labels.coerce_to_labels(self.node_indices))}",
                f"           {fmt.BIG_PHI}: {self.phi}",
                f"Normalized {fmt.BIG_PHI}: {self.normalized_phi}",
                f"Partition: {self.partition}",
                f" Cause state: {self.system_state[Direction.CAUSE]}",
                f"Effect state: {self.system_state[Direction.EFFECT]}",
            ]
        ):
            body = fmt.header(line, body)
        body = fmt.header(
            "System irreducibility analysis", body, under_char=fmt.HEADER_BAR_2
        )
        return fmt.box(fmt.center(body))


class HybridHorizontalSystemPartition(KPartition):
    """Represents a "hybrid horizontal" system partition where one part is cut
    away from the system in a particular direction.
    """

    def __init__(self, direction, *args, **kwargs):
        self.direction = direction
        super().__init__(*args, **kwargs)

    def __repr__(self):
        part = self[0].purview
        if self.node_labels:
            part = self.node_labels.coerce_to_labels(part)
        return f"{','.join(map(str, part))} ({self.direction})"

    def __str__(self):
        return repr(self)


def sia_partitions_hybrid_horizontal(
    node_indices: Iterable, node_labels: Optional[NodeLabels] = None
) -> Generator[SystemPartition, None, None]:
    """Yield all system partitions."""
    # Special case for single-element systems
    if len(node_indices) == 1:
        # Complete partition
        part = tuple(node_indices)
        for direction in Direction.both():
            yield HybridHorizontalSystemPartition(
                direction,
                # NOTE: Order is important here
                Part(mechanism=(), purview=part),
                Part(mechanism=part, purview=()),
                node_labels=node_labels,
            )
        return

    for part, direction in product(
        utils.powerset(
            node_indices,
            nonempty=True,
            max_size=(len(node_indices) - 1),
        ),
        Direction.both(),
    ):
        # Compare π(part|system) vs π(part|part)
        yield HybridHorizontalSystemPartition(
            direction,
            Part(mechanism=part, purview=part),
            node_labels=node_labels,
        )


def normalization_factor_hybrid_horizontal(
    subsystem: Subsystem, partition: SystemPartition
) -> float:
    """Normalize the phi value according to the partition."""
    part = partition[0].purview
    if len(part) == len(subsystem):
        return 1 / len(subsystem) ** 2
    return 1 / (len(part) * (len(subsystem) - len(part)))


def evaluate_partition_hybrid_horizontal(
    partition: HybridHorizontalSystemPartition,
    subsystem: Subsystem,
    system_state: Dict[Direction, tuple],
) -> SystemIrreducibilityAnalysisHybridHorizontal:
    # TODO(4.0) configure repertoire distance
    if not config.REPERTOIRE_DISTANCE == "IIT_4.0_SMALL_PHI":
        raise ValueError('Must set config.REPERTOIRE_DISTANCE = "IIT_4.0_SMALL_PHI"')
    purview = partition[0].purview
    purview_state = utils.state_of(
        # Get purview indices relative to subsystem indices
        [subsystem.node_indices.index(n) for n in purview],
        system_state[partition.direction],
    )
    # Compare π(part|system) vs π(part|part)
    phi, partitioned_repertoire, repertoire = subsystem.evaluate_partition(
        direction=partition.direction,
        mechanism=subsystem.node_indices,
        purview=purview,
        partition=partition,
        state=purview_state,
        return_unpartitioned_repertoire=True,
    )
    normalized_phi = phi * normalization_factor_hybrid_horizontal(subsystem, partition)
    return SystemIrreducibilityAnalysisHybridHorizontal(
        phi=phi,
        normalized_phi=normalized_phi,
        partition=partition,
        repertoire=repertoire,
        partitioned_repertoire=partitioned_repertoire,
        system_state=system_state,
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )


def find_mip_hybrid_horizontal(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    check_trivial_reducibility: Optional[bool] = True,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
) -> SystemIrreducibilityAnalysisHybridHorizontal:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)

    # TODO(4.0) implement
    # if check_trivial_reducibility and is_trivially_reducible(phi_structure):
    #     return NullSystemIrreducibilityAnalysis()

    # NOTE: tie-breaking happens here
    system_state = {
        direction: subsystem.find_maximal_state_under_complete_partition(
            direction,
            mechanism=subsystem.node_indices,
            purview=subsystem.node_indices,
        )[0]
        for direction in Direction.both()
    }

    partitions = sia_partitions_hybrid_horizontal(
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )

    return compute.parallel.map_reduce(
        evaluate_partition_hybrid_horizontal,
        min,
        partitions,
        system_state=system_state,
        subsystem=subsystem,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        shortcircuit_value=0.0,
        parallel=parallel,
        progress=progress,
        desc="Evaluating partitions",
    )


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
