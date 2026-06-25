import numpy as np
import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.models.partitions import CompleteEdgeCut
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import DirectedJointPartition
from pyphi.models.partitions import DirectedSetPartition
from pyphi.models.partitions import EdgeCut
from pyphi.models.partitions import JointBipartition
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import JointTripartition
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import Part

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def _joint():
    return JointPartition(Part((0,), (1,)), Part((1,), (0,)))


PARTITIONS = [
    Part((0, 1), (2,)),
    NullCut((0, 1, 2)),
    DirectedBipartition(Direction.CAUSE, (0,), (1,)),
    _joint(),
    JointBipartition(Part((0,), (1,)), Part((1,), (0,))),
    JointTripartition(Part((0,), (1,)), Part((1,), (2,)), Part((2,), (0,))),
    DirectedJointPartition(Direction.EFFECT, _joint()),
    EdgeCut((0, 1), np.array([[0, 1], [1, 0]])),
    CompleteEdgeCut((0, 1, 2)),
    DirectedSetPartition((0, 1), np.array([[0, 1], [1, 0]]), [[0], [1]]),
]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("obj", PARTITIONS, ids=lambda o: type(o).__name__)
def test_partition_round_trips(obj, fmt):
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is type(obj)
