from pyphi import Direction


def test_fmt_mice(s):
    mice = s.find_mice(Direction.CAUSE, (2,))
    repr(mice)
    str(mice)


def test_partition_arrow_points_along_severed_connections():
    """The concise arrow must point from ``from_nodes`` to ``to_nodes`` —
    the direction of the severed connections (``removed_edges`` and the
    severed-connections grid) — for both causal directions. The causal
    direction is annotated textually instead of by reversing the arrow."""
    from pyphi.models import DirectedBipartition

    for direction in (Direction.CAUSE, Direction.EFFECT):
        cut = DirectedBipartition(direction, (0,), (1,))
        concise = cut._concise()
        assert "[0] ━━/ /━━▶ [1]" in concise
        assert f"({direction.name.lower()})" in concise
        assert cut.removed_edges() == frozenset({(0, 1)})
