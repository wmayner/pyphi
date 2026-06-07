"""Tests for Φ-folds and apportioned relation sums."""

import pytest

from pyphi import examples
from pyphi.relations import AnalyticalRelations
from pyphi.relations import NullRelations


@pytest.fixture(scope="module")
def xor_ces():
    return examples.xor_system().ces()


def test_concrete_apportioned_sum_phi_matches_manual(xor_ces):
    relations = xor_ces.relations
    expected = sum(r.phi / len(r) for r in relations)
    assert relations.apportioned_sum_phi() == pytest.approx(expected)


def test_concrete_apportioned_sum_phi_at_most_sum_phi(xor_ces):
    relations = xor_ces.relations
    # dividing each term by |r| >= 1 cannot increase the total
    assert relations.apportioned_sum_phi() <= relations.sum_phi() + 1e-12


def test_null_relations_apportioned_sum_phi_is_zero():
    assert NullRelations().apportioned_sum_phi() == 0.0


def test_analytical_apportioned_matches_concrete(xor_ces):
    analytical = AnalyticalRelations(xor_ces.distinctions)
    assert analytical.apportioned_sum_phi() == pytest.approx(
        xor_ces.relations.apportioned_sum_phi()
    )
