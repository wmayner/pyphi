import numpy as np
import pytest

from pyphi import serialize
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.labels import NodeLabels
from pyphi.measures.distribution import DistanceResult
from pyphi.models.state_specification import StateSpecification
from pyphi.models.state_specification import SystemStateSpecification

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_state_spec(direction=Direction.CAUSE):
    return StateSpecification(
        direction=direction,
        purview=(0, 1),
        state=(1, 0),
        intrinsic_information=PyPhiFloat(0.25),
        repertoire=np.array([[0.1, 0.4], [0.3, 0.2]]),
        unconstrained_repertoire=np.array([[0.25, 0.25], [0.25, 0.25]]),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_pyphi_float_round_trips(fmt):
    obj = PyPhiFloat(0.375)
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is PyPhiFloat


@pytest.mark.parametrize("fmt", FORMATS)
def test_distance_result_round_trips(fmt):
    obj = DistanceResult(0.5, method="EMD")
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is DistanceResult
    assert restored.method == "EMD"


@pytest.mark.parametrize("fmt", FORMATS)
def test_node_labels_round_trips(fmt):
    obj = NodeLabels(("A", "B", "C"), (0, 1, 2))
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_state_specification_round_trips(fmt):
    obj = make_state_spec()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_state_specification_preserves_tie_peers(fmt):
    a = make_state_spec()
    b = make_state_spec()
    a.set_ties((a, b))
    b.set_ties((a, b))
    restored = round_trip(a, fmt)
    assert restored == a
    # The tie peer is reconstructed and mutual.
    peers = [t for t in restored.ties if t is not restored]
    assert len(peers) == 1
    assert restored in peers[0].ties


@pytest.mark.parametrize("fmt", FORMATS)
def test_system_state_specification_round_trips(fmt):
    obj = SystemStateSpecification(
        cause=make_state_spec(Direction.CAUSE),
        effect=make_state_spec(Direction.EFFECT),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
