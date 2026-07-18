"""Tests for structure views: induced substructures and relation closure."""

import pytest

from pyphi import examples
from pyphi.conf import config
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.ces import PhiFold
from pyphi.models.ces import StructureView
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.relations import AnalyticalRelations
from pyphi.relations import ConcreteRelations
from pyphi.relations import concrete_relations


@pytest.fixture(scope="module")
def xor_ces():
    """Concrete relations: most tests here enumerate/index relations to check
    induced- and folded-view membership, which the analytical backend does
    not support."""
    with config.override(relation_computation="CONCRETE"):
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


@pytest.fixture(scope="module")
def grid3_ces():
    """Concrete relations: see ``xor_ces`` above."""
    with config.override(relation_computation="CONCRETE"):
        return examples.grid3_system().ces()


def test_meet_with_itself_is_whole(xor_ces):
    from pyphi.models.ces import InducedSubstructure

    met = xor_ces.meet(xor_ces)
    assert isinstance(met, InducedSubstructure)
    assert set(met.distinctions) == set(xor_ces.distinctions)
    assert met.big_phi == pytest.approx(xor_ces.big_phi)


def test_meet_of_induced_views(xor_ces):
    ds = list(xor_ces.distinctions)
    left = xor_ces.induce(ds[:3])
    right = xor_ces.induce(ds[1:])
    met = left.meet(right)
    assert set(met.distinctions) == set(ds[1:3])
    # R commutes with intersection of distinction sets
    expected = frozenset(left.relations) & frozenset(right.relations)
    assert frozenset(met.relations) == expected


def test_meet_is_commutative_on_aggregates(xor_ces):
    ds = list(xor_ces.distinctions)
    left, right = xor_ces.induce(ds[:3]), xor_ces.induce(ds[1:])
    a, b = left.meet(right), right.meet(left)
    assert set(a.distinctions) == set(b.distinctions)
    assert a.relations.sum_phi() == pytest.approx(b.relations.sum_phi())


def test_meet_is_associative_on_aggregates(grid3_ces):
    # Meet is intersection of distinction sets restricted to incident
    # relations, so it inherits associativity from set intersection. Three
    # induced views with nonempty pairwise and triple overlaps exercise it.
    ds = list(grid3_ces.distinctions)
    left = grid3_ces.induce(ds[:5])
    middle = grid3_ces.induce(ds[2:])
    right = grid3_ces.induce(ds[1:4] + ds[6:])
    grouped_left = left.meet(middle).meet(right)
    grouped_right = left.meet(middle.meet(right))
    assert set(grouped_left.distinctions) == set(grouped_right.distinctions)
    assert grouped_left.relations.sum_phi() == pytest.approx(
        grouped_right.relations.sum_phi()
    )
    assert frozenset(grouped_left.relations) == frozenset(grouped_right.relations)


def test_meet_requires_same_frame(xor_ces):
    # a structure over a different candidate system (proper subsystem) is a
    # detectable frame mismatch; structures from a different substrate with
    # identical node indices and state are not detectable from the objects
    # (meet then degrades to an empty intersection via value equality)
    from pyphi.system import System

    xor = examples.xor_system()
    sub_ces = System(xor.substrate, state=xor.state, node_indices=(0, 1)).ces()
    with pytest.raises(ValueError, match="not in the same frame"):
        xor_ces.meet(sub_ces)


def test_meet_across_substrates_same_shape_is_empty(xor_ces, grid3_ces):
    # same node indices and state on a different substrate: not detectable
    # as a frame mismatch, but value equality makes the intersection empty
    met = xor_ces.meet(grid3_ces)
    assert len(met.distinctions) == 0


def test_structure_view_save_raises_clear_error(tmp_path):
    import pytest

    from pyphi.models.ces import InducedSubstructure

    assert callable(getattr(InducedSubstructure, "save", None))
    # Any concrete view raises the documented error rather than an opaque
    # serializer-dispatch TypeError.
    with pytest.raises(NotImplementedError, match="view into its parent"):
        InducedSubstructure.save(object.__new__(InducedSubstructure), tmp_path / "x")
