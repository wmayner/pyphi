"""Default-path integration tests for the analytical relation backend.

Deliberately unpinned: these tests must observe the shipping default."""

import pytest

from pyphi import examples
from pyphi.relations import AnalyticalRelations
from pyphi.relations import ConcreteRelations


def test_default_ces_relations_are_analytical():
    ces = examples.xor_system().ces()
    assert isinstance(ces.relations, AnalyticalRelations)


def test_default_summary_matches_concrete_enumeration():
    analytical = examples.xor_system().ces().relations
    concrete = analytical.materialize()
    assert isinstance(concrete, ConcreteRelations)
    assert len(concrete) == analytical.num_relations()
    assert analytical.sum_phi() == pytest.approx(concrete.sum_phi())
