"""Tests for the pure visualization projection layer."""

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


def test_inclusion_order_equal_unions_no_edge():
    # Mutually-equal purview unions: same rank, no cover edge between them.
    unions = (frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    order = _inclusion_order(unions)
    assert order.covers == ((2,), (2,), ())
    assert order.rank == (1, 1, 0)
