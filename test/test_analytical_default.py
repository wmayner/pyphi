"""Default-path integration tests for the analytical relation backend.

Deliberately unpinned: these tests must observe the shipping default."""

import pytest

from pyphi import examples
from pyphi.relations import AnalyticalFoldRelations
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


def test_analytical_relations_value_equality():
    a = examples.xor_system().ces().relations
    b = examples.xor_system().ces().relations
    assert isinstance(a, AnalyticalRelations)
    assert a is not b
    assert a == b
    assert hash(a) == hash(b)


def test_analytical_relations_not_equal_to_concrete():
    analytical = examples.xor_system().ces().relations
    concrete = analytical.materialize()
    assert analytical != concrete
    assert concrete != analytical


def test_analytical_fold_relations_value_equality():
    ces = examples.xor_system().ces()
    d1, d2 = ces.distinctions[0], ces.distinctions[1]
    fold1 = ces.fold([d1]).relations
    fold2 = ces.fold([d2]).relations
    assert isinstance(fold1, AnalyticalFoldRelations)

    # different seeds over the same parent: unequal
    assert fold1 != fold2

    # each equals a fresh reconstruction with the same seeds
    assert fold1 == ces.fold([d1]).relations
    assert hash(fold1) == hash(ces.fold([d1]).relations)

    # a fold summary is never equal to the plain summary of the same
    # distinctions: it describes a different (incident-only) relation set
    assert fold1 != ces.relations
    assert ces.relations != fold1


def test_analytical_fold_relations_seed_order_is_normalized():
    ces = examples.xor_system().ces()
    d1, d2 = ces.distinctions[0], ces.distinctions[1]
    parent = ces.distinctions

    forward = AnalyticalFoldRelations(parent, [d1, d2])
    reversed_ = AnalyticalFoldRelations(parent, [d2, d1])

    assert forward == reversed_
    assert hash(forward) == hash(reversed_)
