"""B7 — total edge-set interface on partitions/cuts."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi.direction import Direction
from pyphi.models.partitions import CompleteEdgeCut
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import DirectedJointPartition
from pyphi.models.partitions import DirectedSetPartition
from pyphi.models.partitions import EdgeCut
from pyphi.models.partitions import JointBipartition
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import JointTripartition
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import Part

# Substrate size large enough to embed every instance's indices.
N = 8

_JP = JointPartition(Part((0,), (1,)), Part((1,), (0,)))

PARTITIONS = [
    NullCut((0, 1, 2)),
    DirectedBipartition(Direction.EFFECT, (0,), (1, 2)),
    DirectedBipartition(Direction.CAUSE, (0, 3), (1, 2)),
    EdgeCut((0, 2, 3), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
    CompleteEdgeCut((1, 2, 4)),
    DirectedSetPartition(
        (0, 1, 2), np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), [[0], [1, 2]]
    ),
    _JP,
    JointBipartition(Part((0, 2), (1,)), Part((1,), (0, 2))),
    JointTripartition(Part((0,), (1,)), Part((1,), (2,)), Part((2,), (0,))),
    DirectedJointPartition(Direction.CAUSE, _JP),
]


@pytest.mark.parametrize("p", PARTITIONS, ids=lambda p: type(p).__name__)
def test_removed_edges_matches_cut_matrix(p):
    expected = frozenset((int(a), int(b)) for a, b in np.argwhere(p.cut_matrix(N)))
    assert p.removed_edges() == expected


def test_removed_edges_n_invariant():
    # The edge set must not change when cut_matrix is evaluated at larger n.
    p = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    small = frozenset((int(a), int(b)) for a, b in np.argwhere(p.cut_matrix(3)))
    assert p.removed_edges() == small


def test_removed_edges_returns_python_ints():
    p = EdgeCut((0, 2), np.array([[0, 1], [0, 0]]))
    (edge,) = p.removed_edges()
    assert type(edge[0]) is int and type(edge[1]) is int


@pytest.mark.parametrize("p", PARTITIONS, ids=lambda p: type(p).__name__)
def test_num_connections_cut_is_edge_count(p):
    assert p.num_connections_cut() == len(p.removed_edges())


def test_num_connections_cut_preserves_eq24_counts():
    # Values the deleted JointPartition Eq. 24 override produced.
    assert JointPartition(Part((0,), (1,)), Part((1,), (0,))).num_connections_cut() == 2
    jb = JointBipartition(Part((0, 2), (1,)), Part((1,), (0, 2)))
    assert jb.num_connections_cut() == 5


def test_nullcut_num_connections_cut_is_zero():
    assert NullCut((0, 1, 2)).num_connections_cut() == 0


def _edgecut(edges, n=4):
    m = np.zeros((n, n), dtype=int)
    for a, b in edges:
        m[a, b] = 1
    return EdgeCut(tuple(range(n)), m)


def test_refines_is_superset_of_severed_edges():
    coarse = _edgecut([(0, 1)])
    fine = _edgecut([(0, 1), (1, 0)])
    assert fine.refines(coarse)  # fine severs a superset
    assert not coarse.refines(fine)
    assert coarse.coarsens(fine)
    assert not fine.coarsens(coarse)


def test_refines_is_reflexive():
    p = _edgecut([(0, 1), (2, 3)])
    assert p.refines(p) and p.coarsens(p)


def test_refines_partial_order_has_incomparable_pairs():
    a = _edgecut([(0, 1)])
    b = _edgecut([(1, 0)])
    assert not a.refines(b) and not b.refines(a)  # genuinely partial


def test_refines_is_transitive():
    a = _edgecut([(0, 1), (1, 0), (2, 3)])
    b = _edgecut([(0, 1), (1, 0)])
    c = _edgecut([(0, 1)])
    assert a.refines(b) and b.refines(c) and a.refines(c)


def test_total_order_matches_lex_key():
    items = [
        DirectedBipartition(Direction.EFFECT, (1,), (2,)),
        NullCut((0, 1)),
        DirectedBipartition(Direction.EFFECT, (0,), (1,)),
    ]
    assert sorted(items) == sorted(items, key=lambda p: p.lex_key())


def test_total_order_operators():
    a = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    b = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    lo, hi = sorted([a, b], key=lambda p: p.lex_key())
    assert lo < hi and lo <= hi and hi > lo and hi >= lo
    assert not (hi < lo)


def test_nullcut_sorts_first():
    null = NullCut((0, 1))
    cut = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    assert sorted([cut, null]) == [null, cut]  # lex_key("") sorts first


def test_equality_unchanged():
    a = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    b = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    assert a == b and hash(a) == hash(b)


@pytest.mark.parametrize("p", PARTITIONS, ids=lambda p: type(p).__name__)
def test_normalization_factor_real_partition_is_not_none(p):
    # Every real partition has num_connections_cut(), so NUM_CONNECTIONS_CUT
    # normalization returns a real factor (1 for a zero-cut partition) — the
    # AttributeError fallback that previously masked this is gone.
    from pyphi.conf import config
    from pyphi.models.state_specification import normalization_factor

    with config.override(distinction_phi_normalization="NUM_CONNECTIONS_CUT"):
        assert normalization_factor(p) is not None


def test_normalization_factor_zero_cut_is_one():
    from pyphi.conf import config
    from pyphi.models.state_specification import normalization_factor

    with config.override(distinction_phi_normalization="NUM_CONNECTIONS_CUT"):
        assert normalization_factor(NullCut((0, 1))) == 1


def test_normalization_factor_none_partition_is_none():
    # A null/unconstrained analysis carries no partition; normalization is
    # undefined. This is now handled explicitly (was caught via AttributeError).
    from pyphi.conf import config
    from pyphi.models.state_specification import normalization_factor

    with config.override(distinction_phi_normalization="NUM_CONNECTIONS_CUT"):
        assert normalization_factor(None) is None
