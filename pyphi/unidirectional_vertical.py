# unidirectional_vertical.py

from dataclasses import dataclass
from typing import Dict, Optional
from numpy.typing import ArrayLike

from . import compute
from .conf import config, fallback
from .direction import Direction
from .metrics.distribution import repertoire_distance as _repertoire_distance
from .models import cmp, fmt
from .models.cuts import Cut
from .partition import system_directed_bipartitions
from .subsystem import Subsystem
from .utils import is_positive
from .labels import NodeLabels

DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD = 256
DEFAULT_PARTITION_CHUNKSIZE = 4 * DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD


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


def find_system_state(
    subsystem: Subsystem,
) -> SystemState:
    # NOTE: Uses config.REPERTOIRE_DISTANCE_INFORMATION
    cause_states, ii_cause = subsystem.find_maximal_state_under_complete_partition(
        Direction.CAUSE,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
        return_information=True,
    )
    effect_states, ii_effect = subsystem.find_maximal_state_under_complete_partition(
        Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
        return_information=True,
    )
    return SystemState(
        # NOTE: tie-breaking happens here
        cause=cause_states[0],
        effect=effect_states[0],
        intrinsic_information={Direction.CAUSE: ii_cause, Direction.EFFECT: ii_effect},
    )


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    phi: float
    normalized_phi: float
    phi_cause: float
    phi_effect: float
    partition: Cut
    repertoire_cause: ArrayLike
    partitioned_repertoire_cause: ArrayLike
    repertoire_effect: ArrayLike
    partitioned_repertoire_effect: ArrayLike
    atomic_integration: Optional[Dict[Direction, float]] = None
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
        # TODO deal with exclusion maximization here when we get to that
        return (self.normalized_phi, self.phi)

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

    def __repr__(self):
        body = "\n".join(
            fmt.align_columns(
                [
                    (
                        "Subsystem",
                        ",".join(self.node_labels.coerce_to_labels(self.node_indices)),
                    ),
                    ("Partition", str(self.partition)),
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


def null_sia() -> SystemIrreducibilityAnalysis:
    return SystemIrreducibilityAnalysis(
        phi=0.0,
        normalized_phi=0.0,
        norm=0.0,
        phi_cause=0.0,
        phi_effect=0.0,
        cause_state=None,
        effect_state=None,
        partition=None,
    )


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


def normalization_factor(partition: Cut) -> float:
    return 1 / (len(partition.from_nodes) * len(partition.to_nodes))


def evaluate_partition(
    partition: Cut,
    subsystem: Subsystem,
    system_state: SystemState,
    repertoire_distance: str = None,
) -> SystemIrreducibilityAnalysis:
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
    norm = normalization_factor(partition)
    phi = min(phi_c, phi_e)
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
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )


def find_mip(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    progress: Optional[bool] = None,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    # include_complete: bool = False,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)
    # TODO: trivial reducibility

    system_state = find_system_state(subsystem)

    # Compute atomic integration with the atomic partition
    # atomic_integration = {
    #     direction: atomic_integration_value(direction, subsystem, system_state)
    #     for direction in Direction.both()
    # }

    partitions = system_directed_bipartitions(subsystem.node_indices)

    return compute.parallel.map_reduce(
        evaluate_partition,
        min,
        partitions,
        subsystem=subsystem,
        system_state=system_state,
        # atomic_integration=atomic_integration,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        shortcircuit_value=0.0,
        parallel=parallel,
        progress=progress,
        desc="Evaluating partitions",
    )
