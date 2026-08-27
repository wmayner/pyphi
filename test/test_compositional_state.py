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


def test_resolve_conflicts_recomputes_counts_against_the_mutated_copy():
    # Ranking candidates by conflict counts computed on the unmutated
    # ``self`` (rather than the in-progress ``resolved`` copy) discards more
    # mechanisms than necessary: after the first removal changes what is
    # actually most conflicted, a stale count keeps ranking a mechanism
    # whose conflicts have already been resolved above one that still has
    # a live conflict, so an extra mechanism gets discarded needlessly.
    from pyphi.compositional_state import CompositionalState
    from pyphi.direction import Direction

    cs = CompositionalState()
    cs.data = {
        (0,): {Direction.CAUSE: {(2,), (5,)}},
        (1,): {Direction.CAUSE: {(1,), (4,), (5,)}},
        (2,): {Direction.CAUSE: {(1,), (5,)}},
        (3,): {Direction.CAUSE: {(2,), (4,)}},
    }
    resolved = cs.resolve_conflicts()
    assert not resolved.has_conflicts()
    # A stale (self-keyed) ranking discards down to a single mechanism,
    # {(2,)}; the live (resolved-keyed) ranking keeps (1,) too.
    assert resolved.mechanisms() == {(1,), (2,)}
