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
