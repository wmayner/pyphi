"""Tests for Φ-folds and apportioned relation sums."""

import pytest

from pyphi import examples
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.ces import PhiFold
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


def test_fold_is_phi_fold_with_parent(xor_ces):
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    assert isinstance(fold, PhiFold)
    assert isinstance(fold, CauseEffectStructure)
    assert fold.parent is xor_ces
    assert [d.mechanism for d in fold.distinctions] == [seed.mechanism]


def test_fold_accepts_mechanism_tuples(xor_ces):
    by_mech = xor_ces.fold([(0, 1)])
    by_obj = xor_ces.fold([d for d in xor_ces.distinctions if d.mechanism == (0, 1)])
    assert [d.mechanism for d in by_mech.distinctions] == [(0, 1)]
    assert by_mech.relations.sum_phi() == pytest.approx(by_obj.relations.sum_phi())


def test_fold_unknown_mechanism_raises(xor_ces):
    with pytest.raises(ValueError, match="not in this cause-effect structure"):
        xor_ces.fold([(9,)])


def test_fold_relations_are_exactly_the_incident_ones(xor_ces):
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    expected = {r for r in xor_ces.relations if seed in r}
    assert set(fold.relations) == expected
    assert all(seed in r for r in fold.relations)


def test_big_phi_contribution_matches_manual(xor_ces):
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    expected = seed.phi + sum(r.phi / len(r) for r in xor_ces.relations if seed in r)
    assert fold.big_phi_contribution == pytest.approx(expected)


def test_distinction_folds_tile_big_phi(xor_ces):
    total = sum(fold.big_phi_contribution for fold in xor_ces.distinction_folds())
    assert total == pytest.approx(xor_ces.big_phi)


def test_fold_big_phi_is_universal_not_contribution(xor_ces):
    seed = xor_ces.distinctions[0]
    fold = xor_ces.fold([seed])
    full = seed.phi + sum(r.phi for r in xor_ces.relations if seed in r)
    assert fold.big_phi == pytest.approx(full)
    assert fold.big_phi >= fold.big_phi_contribution


def test_fold_relations_less_ces_raises():
    from pyphi.models.distinctions import ResolvedDistinctions

    bare = CauseEffectStructure(
        sia=None, distinctions=ResolvedDistinctions(()), relations=NullRelations()
    )
    with pytest.raises(ValueError, match="requires relations"):
        bare.fold([])


@pytest.fixture(scope="module")
def xor_ces_analytical(xor_ces):
    # same distinctions/sia, but analytical relations
    return CauseEffectStructure(
        sia=xor_ces.sia,
        distinctions=xor_ces.distinctions,
        relations=AnalyticalRelations(xor_ces.distinctions),
    )


def test_analytical_fold_sum_matches_concrete_fold(xor_ces, xor_ces_analytical):
    for distinction in xor_ces.distinctions:
        mechanism = distinction.mechanism
        concrete_fold = xor_ces.fold([mechanism])
        analytical_fold = xor_ces_analytical.fold([mechanism])
        assert analytical_fold.relations.sum_phi() == pytest.approx(
            concrete_fold.relations.sum_phi()
        )
        assert analytical_fold.relations.num_relations() == (
            concrete_fold.relations.num_relations()
        )
        assert analytical_fold.relations.apportioned_sum_phi() == pytest.approx(
            concrete_fold.relations.apportioned_sum_phi()
        )


def test_analytical_fold_tiles_big_phi(xor_ces_analytical):
    total = sum(
        fold.big_phi_contribution for fold in xor_ces_analytical.distinction_folds()
    )
    assert total == pytest.approx(xor_ces_analytical.big_phi)
