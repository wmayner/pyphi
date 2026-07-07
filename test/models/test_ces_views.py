"""Tests for structure views: induced substructures and relation closure."""

import pytest

from pyphi import examples
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.ces import PhiFold
from pyphi.models.ces import StructureView
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.relations import AnalyticalRelations
from pyphi.relations import ConcreteRelations
from pyphi.relations import concrete_relations


@pytest.fixture(scope="module")
def xor_ces():
    return examples.xor_system().ces()


@pytest.fixture(scope="module")
def xor_ces_analytical(xor_ces):
    return CauseEffectStructure(
        sia=xor_ces.sia,
        distinctions=xor_ces.distinctions,
        relations=AnalyticalRelations(xor_ces.distinctions),
    )


def test_relation_closed_flags(xor_ces):
    assert xor_ces.relation_closed is True
    fold = xor_ces.fold([xor_ces.distinctions[0]])
    assert fold.relation_closed is False
    induced = xor_ces.induce([xor_ces.distinctions[0]])
    assert induced.relation_closed is True


def test_view_hierarchy(xor_ces):
    fold = xor_ces.fold([xor_ces.distinctions[0]])
    induced = xor_ces.induce([xor_ces.distinctions[0]])
    assert isinstance(fold, StructureView)
    assert isinstance(induced, StructureView)
    assert fold.parent is xor_ces
    assert induced.parent is xor_ces


def test_induce_relations_are_the_contained_ones(xor_ces):
    members = list(xor_ces.distinctions)[:3]
    induced = xor_ces.induce(members)
    member_set = set(members)
    expected = {r for r in xor_ces.relations if member_set.issuperset(r)}
    assert set(induced.relations) == expected


def test_induce_matches_fresh_computation(xor_ces):
    # relation locality: filtering the parent's relations equals computing
    # relations over the subset from scratch
    members = ResolvedDistinctions(list(xor_ces.distinctions)[:3])
    induced = xor_ces.induce(members)
    fresh = ConcreteRelations(concrete_relations(members))
    assert frozenset(induced.relations) == frozenset(fresh)


def test_induce_accepts_mechanism_tuples(xor_ces):
    by_mech = xor_ces.induce([(0, 1)])
    assert [d.mechanism for d in by_mech.distinctions] == [(0, 1)]


def test_induce_unknown_mechanism_raises(xor_ces):
    with pytest.raises(ValueError, match="not in this cause-effect structure"):
        xor_ces.induce([(9,)])


def test_induce_composes(xor_ces):
    members = list(xor_ces.distinctions)
    inner = xor_ces.induce(members[:3]).induce(members[:2])
    direct = xor_ces.induce(members[:2])
    assert set(inner.distinctions) == set(direct.distinctions)
    assert frozenset(inner.relations) == frozenset(direct.relations)


def test_induce_all_is_whole_structure(xor_ces):
    induced = xor_ces.induce(list(xor_ces.distinctions))
    assert set(induced.distinctions) == set(xor_ces.distinctions)
    assert frozenset(induced.relations) == frozenset(xor_ces.relations)
    assert induced.big_phi == pytest.approx(xor_ces.big_phi)


def test_induce_analytical_aggregates_match_concrete(xor_ces, xor_ces_analytical):
    mechanisms = [d.mechanism for d in list(xor_ces.distinctions)[:3]]
    concrete = xor_ces.induce(mechanisms)
    analytical = xor_ces_analytical.induce(mechanisms)
    assert analytical.relations.sum_phi() == pytest.approx(concrete.relations.sum_phi())
    assert analytical.relations.num_relations() == concrete.relations.num_relations()


def test_fold_still_works_after_refactor(xor_ces):
    # regression guard on the _resolve_members extraction
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    assert isinstance(fold, PhiFold)
    assert {r for r in xor_ces.relations if seed in r} == set(fold.relations)
