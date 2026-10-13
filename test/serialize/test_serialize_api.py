import msgspec
import pytest

from pyphi import serialize
from pyphi.direction import Direction


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_direction_round_trips(fmt):
    data = serialize.dumps(Direction.CAUSE, format=fmt)
    assert isinstance(data, bytes)
    assert serialize.loads(data, format=fmt) == Direction.CAUSE


def test_unknown_type_tag_raises():
    bad = serialize.dumps(Direction.CAUSE, format="json").replace(
        b'"direction"', b'"nonsense"'
    )
    with pytest.raises(ValueError, match="newer version of PyPhi") as raised:
        serialize.loads(bad, format="json")
    assert isinstance(raised.value.__cause__, msgspec.ValidationError)


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_complexes_and_sia_lists_round_trip(fmt):
    import pyphi

    with pyphi.config.override(**pyphi.iit4_2026):
        system = pyphi.examples.iit4_2023_fig1a_system()
        found = system.substrate.complexes(system.state)
        sias = system.substrate.all_sias(system.state)
    restored = pyphi.serialize.loads(
        pyphi.serialize.dumps(found, format=fmt), format=fmt
    )
    assert isinstance(restored, tuple)
    assert restored == found
    restored = pyphi.serialize.loads(pyphi.serialize.dumps(sias, format=fmt), format=fmt)
    assert isinstance(restored, list)
    assert restored == sias
