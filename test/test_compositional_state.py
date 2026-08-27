"""Tests for pyphi.compositional_state."""

import pytest

from pyphi import Direction
from pyphi.compositional_state import CompositionalState


@pytest.fixture
def conflicted():
    # Purview (0,) is claimed in the same direction by two mechanisms;
    # mechanism (2,) claims a different purview with no competition.
    cs = CompositionalState()
    cs.node_labels = None
    cs.data = {
        (0,): {Direction.CAUSE: [(0,), (1,)]},
        (1,): {Direction.CAUSE: [(2,)]},
    }
    return cs


def test_mechanism_has_conflicts_is_per_mechanism(conflicted):
    assert conflicted.has_conflicts(mechanism=(0,))
    assert conflicted.has_conflicts(mechanism=(1,))
    assert not conflicted.has_conflicts(mechanism=(2,))


def test_number_of_conflicts_consistent(conflicted):
    assert conflicted.number_of_conflicts((0,)) > 0
    assert conflicted.number_of_conflicts((2,)) == 0


def test_empty_compositional_state_is_usable():
    """No-argument construction must give a fully working empty state."""
    from pyphi.compositional_state import CompositionalState

    empty = CompositionalState()
    assert empty.mechanisms() == set()
    assert not empty.has_conflicts()
    repr(empty)


def test_conflicts_with_one_direction_purview():
    """A purview claimed in only one direction is no conflict for the
    other direction, not a KeyError."""
    from pyphi.compositional_state import CompositionalState
    from pyphi.direction import Direction

    state = CompositionalState()
    state.data[(1,)] = {Direction.CAUSE: {(0,)}}
    assert not state.conflicts_with((0,), (1,), (1,))
