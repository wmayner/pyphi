import pytest

from pyphi import serialize
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis
from pyphi.models.complex import ExcludedCandidate
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import NullCut
from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis
from pyphi.provenance import Provenance

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_provenance_round_trips(fmt):
    obj = Provenance.capture()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_excluded_candidate_round_trips(fmt):
    obj = ExcludedCandidate((0, 1), 0.5)
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_round_trips(fmt):
    obj = IIT3SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
        node_indices=(0, 1),
        current_state=(1, 0),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is IIT3SystemIrreducibilityAnalysis


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_preserves_tie_peers(fmt):
    def make():
        return IIT3SystemIrreducibilityAnalysis(
            phi=0.5,
            partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
            node_indices=(0, 1),
            current_state=(1, 0),
        )

    a, b = make(), make()
    a.set_ties([a, b])
    b.set_ties([a, b])
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored.ties if t is not restored]
    assert len(peers) == 1


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_round_trips(fmt):
    obj = SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=NullCut((0, 1)),
        normalized_phi=0.25,
        current_state=(1, 0),
        node_indices=(0, 1),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is SystemIrreducibilityAnalysis
    # config degrades to a plain dict, matching the prior serializer.
    assert isinstance(restored.config, dict)


@pytest.mark.parametrize("fmt", FORMATS)
def test_null_iit4_sia_round_trips(fmt):
    obj = NullSystemIrreducibilityAnalysis(node_indices=(0, 1))
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is NullSystemIrreducibilityAnalysis


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_preserves_intrinsic_differentiation(fmt):
    obj = SystemIrreducibilityAnalysis(phi=0.5, partition=NullCut((0, 1)))
    obj.intrinsic_differentiation = {
        Direction.CAUSE: serialize.loads(
            serialize.dumps(obj.phi, format=fmt), format=fmt
        ),
        Direction.EFFECT: obj.normalized_phi,
    }
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert set(restored.intrinsic_differentiation) == {
        Direction.CAUSE,
        Direction.EFFECT,
    }


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_preserves_tie_peers(fmt):
    def make():
        return SystemIrreducibilityAnalysis(phi=0.5, partition=NullCut((0, 1)))

    a, b = make(), make()
    a.set_ties([a, b])
    b.set_ties([a, b])
    restored = round_trip(a, fmt)
    assert restored == a
    peers = [t for t in restored.ties if t is not restored]
    assert len(peers) == 1
