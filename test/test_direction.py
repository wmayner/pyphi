import pytest

from pyphi import Direction


def test_direction_order():
    mechanism = (0,)
    purview = (1, 2)
    assert Direction.CAUSE.order(mechanism, purview) == (purview, mechanism)
    assert Direction.EFFECT.order(mechanism, purview) == (mechanism, purview)

    with pytest.raises(ValueError):
        Direction.BIDIRECTIONAL.order(mechanism, purview)


@pytest.mark.parametrize(
    "direction",
    [Direction.CAUSE, Direction.EFFECT, Direction.BIDIRECTIONAL],
)
def test_direction_serialize_round_trip(direction):
    from pyphi import serialize

    assert serialize.loads(serialize.dumps(direction)) == direction


def test_direction_str():
    assert str(Direction.CAUSE) == "CAUSE"
    assert str(Direction.EFFECT) == "EFFECT"
    assert str(Direction.BIDIRECTIONAL) == "BIDIRECTIONAL"
