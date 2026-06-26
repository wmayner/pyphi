import numpy as np
import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.models.mice import MaximallyIrreducibleCause
from pyphi.models.mice import MaximallyIrreducibleEffect
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_ria(direction=Direction.CAUSE, phi=0.3):
    return RepertoireIrreducibilityAnalysis(
        phi=phi,
        direction=direction,
        mechanism=(0,),
        purview=(1,),
        partition=JointPartition(Part((0,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_round_trips(fmt):
    obj = make_ria()
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert np.array_equal(restored.repertoire, obj.repertoire)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_partition_tie_peers(fmt):
    a = make_ria(phi=0.3)
    b = make_ria(phi=0.3)
    tied = (a, b)
    a._partition_ties = tied
    b._partition_ties = tied
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored._partition_ties if t is not restored]
    assert len(peers) == 1
    assert restored in peers[0]._partition_ties


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_state_tie_peers(fmt):
    a = make_ria(phi=0.3)
    b = make_ria(phi=0.3)
    tied = (a, b)
    a._state_ties = tied
    b._state_ties = tied
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored._state_ties if t is not restored]
    assert len(peers) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_mic_round_trips(fmt):
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is MaximallyIrreducibleCause


@pytest.mark.parametrize("fmt", FORMATS)
def test_mie_round_trips(fmt):
    obj = MaximallyIrreducibleEffect(make_ria(Direction.EFFECT))
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is MaximallyIrreducibleEffect
