# new_big_phi/vertical_cuts.py

from dataclasses import dataclass

from .. import Direction, combinatorics, config, utils
from ..compute.parallel import MapReduce
from ..models import cmp, fmt
from ..new_big_phi import (
    DEFAULT_PARTITION_CHUNKSIZE,
    DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    NullSystemIrreducibilityAnalysis,
    ShortCircuitConditions,
)
from ..partition import CompletePartition, KPartition, Part, complete_partition


def normalization_factor(subsystem, partition):
    if isinstance(partition, CompletePartition):
        return len(subsystem) ** 2
    return sum(
        [
            # NOTE: Assumes mechanism and purview are the same in each part
            len(part.mechanism) * (len(subsystem) - len(part.mechanism))
            for part in partition
        ]
    ) * len(subsystem)


def system_set_partitions(collection, node_labels=None):
    collection = tuple(collection)
    partitions = combinatorics.set_partitions(collection)
    for partition in partitions:
        if len(partition) == 1:
            yield complete_partition(collection, collection)
        else:
            yield KPartition(
                *[Part(tuple(part), tuple(part)) for part in partition],
                node_labels=node_labels,
            )


def evaluate_system_partition(
    partition,
    subsystem,
    sys_effect_state,
    sys_cause_state,
    effect_only=False,
):
    effect_info, _ = subsystem.evaluate_partition(
        Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
        partition=partition,
        state=sys_effect_state,
    )
    if effect_only:
        phi = effect_info
    else:
        cause_info, _ = subsystem.evaluate_partition(
            Direction.CAUSE,
            mechanism=subsystem.node_indices,
            purview=subsystem.node_indices,
            partition=partition,
            state=sys_cause_state,
        )
        phi = min(effect_info, cause_info)

    norm = normalization_factor(subsystem, partition)
    normalized_phi = phi / norm
    return SystemIrreducibilityAnalysis(
        phi=phi,
        norm=norm,
        normalized_phi=normalized_phi,
        cause_state=sys_cause_state,
        effect_state=sys_effect_state,
        partition=partition,
    )


@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    phi: float
    norm: float
    normalized_phi: float
    cause_state: tuple
    effect_state: tuple
    partition: KPartition
    effect_only: bool = False

    _sia_attributes = ["phi", "cause_state", "effect_state", "partition"]

    def order_by(self):
        return self.normalized_phi

    def __eq__(self, other):
        return cmp.general_eq(self, other, self._sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has |big_phi > 0|."""
        return utils.is_positive(self.phi)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.cause_state,
                self.effect_state,
                self.partition,
            )
        )

    def __repr__(self):
        body = ""
        for line in reversed(
            [
                f" Cause state: {self.cause_state}",
                f"Effect state: {self.effect_state}",
                f" Effect only: {self.effect_only}",
            ]
        ):
            body = fmt.header(line, body)
        body = "\n" + fmt.header(str(self.partition), body, under_char=" ")
        body = (
            "\n".join(
                fmt.align_columns(
                    [
                        (fmt.BIG_PHI, self.phi),
                        ("Norm", self.norm),
                        (f"Normalized {fmt.BIG_PHI}", self.normalized_phi),
                        ("Partition", ""),
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


def sia(subsystem, effect_only=False, **kwargs):
    kwargs = {
        "parallel": config.PARALLEL_CUT_EVALUATION,
        "progress": config.PROGRESS_BARS,
        "chunksize": DEFAULT_PARTITION_CHUNKSIZE,
        "sequential_threshold": DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
        **kwargs,
    }

    # NOTE: Tie-breaking happens here when we access the first element.
    sys_effect_state = subsystem.intrinsic_information(
        Direction.EFFECT,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]
    sys_cause_state = subsystem.intrinsic_information(
        Direction.CAUSE,
        mechanism=subsystem.node_indices,
        purview=subsystem.node_indices,
    )[0]

    default_sia = NullSystemIrreducibilityAnalysis(
        reasons=[ShortCircuitConditions.NO_VALID_PARTITIONS],
        node_indices=subsystem.node_indices,
        node_labels=subsystem.node_labels,
    )
    return MapReduce(
        evaluate_system_partition,
        system_set_partitions(
            subsystem.node_indices, node_labels=subsystem.node_labels
        ),
        map_kwargs=dict(
            subsystem=subsystem,
            sys_cause_state=sys_cause_state,
            sys_effect_state=sys_effect_state,
            effect_only=effect_only,
        ),
        reduce_func=min,
        reduce_kwargs=dict(default=default_sia),
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating partitions",
        **kwargs,
    )
