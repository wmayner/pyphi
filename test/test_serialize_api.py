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
    with pytest.raises(msgspec.ValidationError):
        serialize.loads(bad, format="json")
