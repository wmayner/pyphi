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
