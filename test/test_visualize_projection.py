"""Tests for the pure visualization projection layer."""

import pytest

from pyphi.visualize.projection import InclusionOrder
from pyphi.visualize.projection import RelationEdge
from pyphi.visualize.projection import _inclusion_order
from pyphi.visualize.projection import _sum_phi_relations


def test_sum_phi_relations_exact():
    edges = (
        RelationEdge(relata=(0, 1), degree=2, phi=0.5, overlap=(2,)),
        RelationEdge(relata=(0, 1, 2), degree=3, phi=1.0, overlap=(0, 1)),
    )
    assert _sum_phi_relations(4, edges) == (1.5, 1.5, 1.0, 0.0)


def test_inclusion_order_exact():
    # purview-unit sets: two points, a pair, the whole.
    unions = (
        frozenset({0}),
        frozenset({1}),
        frozenset({0, 1}),
        frozenset({0, 1, 2}),
    )
    order = _inclusion_order(unions)
    assert isinstance(order, InclusionOrder)
    # covers = transitive reduction: the whole covers only the pair;
    # the pair covers both points; points cover nothing.
    assert order.covers == ((), (), (0, 1), (2,))
    # rank = longest down-chain: points 0, pair 1, whole 2.
    assert order.rank == (0, 0, 1, 2)
    # size = cardinality of each set.
    assert order.size == (1, 1, 2, 3)


def test_inclusion_order_equal_unions_no_edge():
    # Mutually-equal purview unions: same rank, no cover edge between them.
    unions = (frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    order = _inclusion_order(unions)
    assert order.covers == ((2,), (2,), ())
    assert order.rank == (1, 1, 0)


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    ces = examples.xor_system().ces()
    return project_phi_structure(ces)


def test_project_xor_nodes(xor_projection):
    proj = xor_projection
    assert len(proj.nodes) == 4
    assert [n.mechanism for n in proj.nodes] == [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]
    assert [n.label for n in proj.nodes] == ["ab", "ac", "bc", "abc"]
    # First distinction's observed values (xor in (0,0,0)):
    assert proj.nodes[0].cause_purview == (0, 1, 2)
    assert proj.nodes[0].effect_purview == (2,)
    assert proj.nodes[0].phi == pytest.approx(0.5)


def test_project_xor_edges(xor_projection):
    proj = xor_projection
    assert len(proj.edges) == 15
    for e in proj.edges:
        assert all(0 <= i < 4 for i in e.relata)
        # Relation is a frozenset of relata, so a self-relation whose cause
        # and effect relata coincide has degree 1.
        assert e.degree >= 1
        assert e.phi >= 0


def test_project_xor_sum_phi_relations_consistent(xor_projection):
    proj = xor_projection
    # Recompute independently from the projected edges.
    expected = [0.0] * 4
    for e in proj.edges:
        for i in e.relata:
            expected[i] += e.phi
    for node, exp in zip(proj.nodes, expected, strict=True):
        assert node.sum_phi_relations == pytest.approx(exp)


def test_project_xor_inclusion_monotone(xor_projection):
    proj = xor_projection
    for order in (proj.mechanism_inclusion, proj.purview_union_inclusion):
        # rank strictly decreases along covers edges.
        for a, cov in enumerate(order.covers):
            for b in cov:
                assert order.rank[a] > order.rank[b]


def test_project_xor_mechanism_inclusion_exact(xor_projection):
    # Mechanisms ab, ac, bc, abc: the pairs are incomparable; abc covers all.
    order = xor_projection.mechanism_inclusion
    assert order.covers == ((), (), (), (0, 1, 2))
    assert order.rank == (0, 0, 0, 1)
    assert order.size == (2, 2, 2, 3)


def test_projection_inclusion_accessor(xor_projection):
    proj = xor_projection
    assert proj.inclusion("mechanism") is proj.mechanism_inclusion
    assert proj.inclusion("purview_union") is proj.purview_union_inclusion
    with pytest.raises(ValueError, match="order"):
        proj.inclusion("bogus")


def test_theme_frozen_with_defaults():
    import dataclasses

    from pyphi.visualize.theme import DEFAULT_THEME
    from pyphi.visualize.theme import Theme

    assert isinstance(DEFAULT_THEME, Theme)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THEME.colorscale = "Plasma"  # type: ignore[misc]
    dark = dataclasses.replace(DEFAULT_THEME, background="black")
    assert dark.background == "black" and DEFAULT_THEME.background == "white"
