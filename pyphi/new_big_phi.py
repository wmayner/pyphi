# new_big_phi.py

from collections import defaultdict
from dataclasses import dataclass
from typing import Generator, Iterable, Optional

from numpy.typing import ArrayLike

from pyphi.labels import NodeLabels
from pyphi.models.subsystem import CauseEffectStructure

from . import Direction, Subsystem, compute, config, metrics, utils
from .conf import fallback
from .models import cmp, fmt
from .models.cuts import CompleteSystemPartition, SystemPartition
from .partition import system_partition_types

# TODO change SystemPartition
from .relations import Relations
from .relations import relations as compute_relations


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    partition: SystemPartition
    phi: float
    reasons: Optional[list] = None

    _sia_attributes = ["phi", "partition"]

    def order_by(self):
        return self.phi

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
        # body = ""
        # for line in reversed(
        #     [
        #         f" Cause state: {self.cause_state}",
        #         f"Effect state: {self.effect_state}",
        #         f" Effect only: {self.effect_only}",
        #     ]
        # ):
        #     body = fmt.header(line, body)
        body = str(self.partition)
        body = (
            "\n".join(
                fmt.align_columns(
                    [
                        (fmt.BIG_PHI, self.phi),
                        # ("Norm", self.norm),
                        # (f"Normalized {fmt.BIG_PHI}", self.normalized_phi),
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


class NullSystemIrreducibilityAnalysis(SystemIrreducibilityAnalysis):
    def __init__(self):
        super().__init__(
            partition=None,
            phi=0.0,
            reasons=[],
        )


def normalization_factor(subsystem: Subsystem, partition: SystemPartition) -> float:
    """Normalize the phi value according to the partition."""
    smallest_part_size = min(len(partition.from_nodes), len(partition.to_nodes))
    return len(subsystem) / smallest_part_size


def integration_value(
    subsystem: Subsystem,
    partition: SystemPartition,
    unpartitioned_repertoire: ArrayLike,
    partitioned_repertoire: ArrayLike,
) -> float:
    """Return the (normalized) integration value of a partition and associated repertoires."""
    kld = metrics.distribution.kld(unpartitioned_repertoire, partitioned_repertoire)
    norm = normalization_factor(subsystem, partition)
    return kld * norm


def evaluate_partition(
    partition: SystemPartition, subsystem: Subsystem
) -> SystemIrreducibilityAnalysis:
    mechanism = partition.from_nodes
    purview = partition.to_nodes
    unpartitioned_system_repertoire = subsystem.repertoire(
        partition.direction, mechanism, purview
    )
    partitioned_system_repertoire = subsystem.unconstrained_repertoire(
        partition.direction, purview
    )
    phi = integration_value(
        subsystem,
        partition,
        unpartitioned_system_repertoire,
        partitioned_system_repertoire,
    )
    # TODO(4.0) configure repertoire distance
    return SystemIrreducibilityAnalysis(
        partition=partition,
        phi=phi,
    )


def sia_partitions(
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


DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD = 256
DEFAULT_PARTITION_CHUNKSIZE = 4 * DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD


def find_mip(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    check_trivial_reducibility: Optional[bool] = True,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)

    # TODO(4.0) implement
    # if check_trivial_reducibility and is_trivially_reducible(phi_structure):
    #     return NullSystemIrreducibilityAnalysis()

    partitions = sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels)

    return compute.parallel.map_reduce(
        evaluate_partition,
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
