"""Tests for relabeling cause-effect structures through index bijections."""

import pytest

import pyphi
from pyphi import examples
from pyphi.automorphism import structure_signature
from test import example_substrates as es

PERM = (2, 0, 1)  # new index i holds old node PERM[i]


@pytest.fixture(scope="module")
def grid3_ces():
    with pyphi.config.override(progress_bars=False):
        return examples.grid3_system().ces()


@pytest.fixture(scope="module")
def permuted_ces():
    with pyphi.config.override(progress_bars=False):
        return es.permuted_system(examples.grid3_system(), PERM).ces()


@pytest.fixture(scope="module")
def relabeled(grid3_ces):
    mapping = {old: PERM.index(old) for old in range(3)}
    return grid3_ces.relabel(mapping)


def test_relabel_matches_recomputation(relabeled, permuted_ces):
    # relabeling the structure equals recomputing on the permuted system,
    # at signature resolution
    assert structure_signature(relabeled) == structure_signature(permuted_ces)


def test_relabel_preserves_aggregates(grid3_ces, relabeled):
    assert relabeled.big_phi == pytest.approx(grid3_ces.big_phi)
    assert relabeled.sum_phi_distinctions == pytest.approx(
        grid3_ces.sum_phi_distinctions
    )
    assert relabeled.relations.num_relations() == grid3_ces.relations.num_relations()
    assert float(relabeled.sia.phi) == pytest.approx(float(grid3_ces.sia.phi))


def test_relabel_round_trip(grid3_ces, relabeled):
    mapping = {old: PERM.index(old) for old in range(3)}
    inverse = {new: old for old, new in mapping.items()}
    back = relabeled.relabel(inverse)
    assert structure_signature(back) == structure_signature(grid3_ces)


def test_identity_relabel_is_signature_noop(grid3_ces):
    identity = {i: i for i in range(3)}
    assert structure_signature(grid3_ces.relabel(identity)) == structure_signature(
        grid3_ces
    )


def test_relabel_repr_does_not_crash(relabeled):
    assert repr(relabeled)
    assert repr(relabeled.sia)
    assert repr(relabeled.distinctions[0])


def test_relabel_rejects_views(grid3_ces):
    view = grid3_ces.induce(list(grid3_ces.distinctions)[:2])
    with pytest.raises(ValueError, match="parent structure"):
        view.relabel({i: i for i in range(3)})


def test_relabel_rejects_non_bijection(grid3_ces):
    with pytest.raises(ValueError, match="injective"):
        grid3_ces.relabel({0: 0, 1: 0, 2: 2})


def test_relabel_rejects_partial_mapping(grid3_ces):
    with pytest.raises(ValueError, match="cover"):
        grid3_ces.relabel({0: 1, 1: 0})


def test_relabel_preserves_selection_margins(grid3_ces, relabeled):
    original = grid3_ces.sia
    assert (relabeled.sia.partition_margin is None) == (
        original.partition_margin is None
    )
    if original.partition_margin is not None:
        assert float(relabeled.sia.partition_margin) == pytest.approx(
            float(original.partition_margin)
        )
    for direction, margin in original.state_margins.items():
        relabeled_margin = relabeled.sia.state_margins[direction]
        if margin is None:
            assert relabeled_margin is None
        else:
            assert float(relabeled_margin) == pytest.approx(float(margin))
