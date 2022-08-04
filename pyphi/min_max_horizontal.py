from dataclasses import dataclass
from typing import Iterable, Optional, Union

from numpy.typing import ArrayLike

from . import Direction, Subsystem, compute, config, utils
from .conf import fallback
from .labels import NodeLabels
from .metrics.distribution import repertoire_distance as _repertoire_distance
from .models import cmp, fmt
from .models.cuts import Cut, SystemPartition
from .new_big_phi import (
    DEFAULT_PARTITION_CHUNKSIZE,
    DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    HorizontalSystemPartition,
    SystemState,
    find_system_state,
    sia_partitions_horizontal,
    normalization_factor_horizontal,
)

HORIZONTAL_PARTITION_CODE = "1210"


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    phi: float
    normalized_phi: float
    partition: Union[Cut, SystemPartition]
    maximal_purview: tuple[int]
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
    system_state: SystemState,
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


def find_mip(
    subsystem: Subsystem,
    parallel: Optional[bool] = None,
    chunksize: int = DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold: int = DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    progress: Optional[bool] = None,
    repertoire_distance: str = None,
    directions: Optional[Iterable[Direction]] = None,
    code: str = HORIZONTAL_PARTITION_CODE,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)
    # TODO: trivial reducibility

    system_state = find_system_state(subsystem)

    # Find MIP
    partitions = sia_partitions_horizontal(
        subsystem.node_indices,
        node_labels=subsystem.node_labels,
        code=code,
        directions=directions,
    )
    mip = compute.parallel.map_reduce(
        evaluate_partition,
        min,
        partitions,
        subsystem=subsystem,
        system_state=system_state,
        repertoire_distance=repertoire_distance,
        # atomic_integration=atomic_integration,
        parallel=parallel,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        shortcircuit_value=0.0,
        progress=progress,
        desc="Evaluating partitions",
    ).partition

    # Find maximal purview for the minimum information partition
    purviews = utils.powerset(mip.purview, nonempty=True)
    return compute.parallel.map_reduce(
        evaluate_purview,
        max,
        purviews,
        partition=mip,
        subsystem=subsystem,
        system_state=system_state,
        repertoire_distance=repertoire_distance,
        parallel=parallel,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        progress=progress,
        desc="Evaluating purviews",
        **kwargs,
    )
