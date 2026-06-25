import numpy as np
import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.models.distinction import Distinction
from pyphi.models.distinctions import Distinctions
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.distinctions import UnresolvedDistinctions
from pyphi.models.mice import MaximallyIrreducibleCause
from pyphi.models.mice import MaximallyIrreducibleEffect
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_ria(direction, node):
    return RepertoireIrreducibilityAnalysis(
        phi=0.3,
        direction=direction,
        mechanism=(node,),
        purview=(1,),
        partition=JointPartition(Part((node,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
    )


def make_distinction(node=0):
    return Distinction(
        mechanism=(node,),
        cause=MaximallyIrreducibleCause(make_ria(Direction.CAUSE, node)),
        effect=MaximallyIrreducibleEffect(make_ria(Direction.EFFECT, node)),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_distinction_round_trips(fmt):
    obj = make_distinction()
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert restored.cause.parent is restored
    assert restored.effect.parent is restored


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    "cls", [Distinctions, UnresolvedDistinctions, ResolvedDistinctions]
)
def test_distinctions_round_trip(cls, fmt):
    obj = cls([make_distinction(0), make_distinction(1)])
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is cls
