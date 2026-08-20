import itertools

import pytest

from pyphi import Direction
from pyphi import config
from pyphi.combinatorics import set_partitions as partitions
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.models import JointBipartition
from pyphi.models import JointPartition
from pyphi.models import JointTripartition
from pyphi.models import Part
from pyphi.partition import all_joint_partitions
from pyphi.partition import directed_bipartition
from pyphi.partition import directed_tripartition_indices
from pyphi.partition import joint_bipartitions
from pyphi.partition import k_partitions
from pyphi.partition import partition_types
from pyphi.partition import wedge_tripartitions


def test_partitions():
    assert list(partitions([])) == []
    assert list(partitions([0])) == [[[0]]]
    assert list(partitions(range(4))) == [
        [[0, 1, 2, 3]],
        [[0], [1, 2, 3]],
        [[0, 1], [2, 3]],
        [[1], [0, 2, 3]],
        [[0], [1], [2, 3]],
        [[0, 1, 2], [3]],
        [[1, 2], [0, 3]],
        [[0], [1, 2], [3]],
        [[0, 2], [1, 3]],
        [[2], [0, 1, 3]],
        [[0], [2], [1, 3]],
        [[0, 1], [2], [3]],
        [[1], [0, 2], [3]],
        [[1], [2], [0, 3]],
        [[0], [1], [2], [3]],
    ]


def test_directed_bipartition():
    answer = [
        ((), (1, 2, 3)),
        ((1,), (2, 3)),
        ((2,), (1, 3)),
        ((1, 2), (3,)),
        ((3,), (1, 2)),
        ((1, 3), (2,)),
        ((2, 3), (1,)),
        ((1, 2, 3), ()),
    ]
    assert answer == directed_bipartition((1, 2, 3))
    # Test with empty input
    assert directed_bipartition(()) == []


def test_directed_tripartition_indices():
    assert directed_tripartition_indices(0) == []
    assert directed_tripartition_indices(2) == [
        ((0, 1), (), ()),
        ((0,), (1,), ()),
        ((0,), (), (1,)),
        ((1,), (0,), ()),
        ((), (0, 1), ()),
        ((), (0,), (1,)),
        ((1,), (), (0,)),
        ((), (1,), (0,)),
        ((), (), (0, 1)),
    ]


def test_k_partition():
    # Special/edge cases
    for n, k in list(itertools.product(range(-1, 2), repeat=2))[:-1]:
        assert list(k_partitions(range(n), k)) == []
    assert list(k_partitions(range(1), 1)) == [[[0]]]
    assert list(k_partitions(range(3), 1)) == [[[0, 1, 2]]]
    assert list(k_partitions(range(3), 3)) == [[[0], [1], [2]]]
    # There are no partitions of an n-set into k > n nonempty blocks.
    assert list(k_partitions(range(3), 4)) == []
    assert list(k_partitions(range(4), 3)) == [
        [[0, 1], [2], [3]],
        [[0], [1, 2], [3]],
        [[0, 2], [1], [3]],
        [[0], [1], [2, 3]],
        [[0], [1, 3], [2]],
        [[0, 3], [1], [2]],
    ]
    assert list(k_partitions(range(5), 2)) == [
        [[0, 1, 2, 3], [4]],
        [[0, 2, 3], [1, 4]],
        [[0, 3], [1, 2, 4]],
        [[0, 1, 3], [2, 4]],
        [[0, 1], [2, 3, 4]],
        [[0], [1, 2, 3, 4]],
        [[0, 2], [1, 3, 4]],
        [[0, 1, 2], [3, 4]],
        [[0, 1, 2, 4], [3]],
        [[0, 2, 4], [1, 3]],
        [[0, 4], [1, 2, 3]],
        [[0, 1, 4], [2, 3]],
        [[0, 1, 3, 4], [2]],
        [[0, 3, 4], [1, 2]],
        [[0, 2, 3, 4], [1]],
    ]
    assert list(k_partitions(range(5), 3)) == [
        [[0, 1, 2], [3], [4]],
        [[0, 1], [2, 3], [4]],
        [[0], [1, 2, 3], [4]],
        [[0, 2], [1, 3], [4]],
        [[0, 2, 3], [1], [4]],
        [[0, 3], [1, 2], [4]],
        [[0, 1, 3], [2], [4]],
        [[0, 1], [2], [3, 4]],
        [[0], [1, 2], [3, 4]],
        [[0, 2], [1], [3, 4]],
        [[0], [1], [2, 3, 4]],
        [[0], [1, 3], [2, 4]],
        [[0, 3], [1], [2, 4]],
        [[0, 3], [1, 4], [2]],
        [[0], [1, 3, 4], [2]],
        [[0], [1, 4], [2, 3]],
        [[0, 2], [1, 4], [3]],
        [[0], [1, 2, 4], [3]],
        [[0, 1], [2, 4], [3]],
        [[0, 1, 4], [2], [3]],
        [[0, 4], [1, 2], [3]],
        [[0, 2, 4], [1], [3]],
        [[0, 4], [1], [2, 3]],
        [[0, 4], [1, 3], [2]],
        [[0, 3, 4], [1], [2]],
    ]
    assert list(k_partitions(range(6), 3)) == [
        [[0, 1, 2, 3], [4], [5]],
        [[0, 1, 2], [3, 4], [5]],
        [[0, 2], [1, 3, 4], [5]],
        [[0], [1, 2, 3, 4], [5]],
        [[0, 1], [2, 3, 4], [5]],
        [[0, 1, 3], [2, 4], [5]],
        [[0, 3], [1, 2, 4], [5]],
        [[0, 2, 3], [1, 4], [5]],
        [[0, 2, 3, 4], [1], [5]],
        [[0, 3, 4], [1, 2], [5]],
        [[0, 1, 3, 4], [2], [5]],
        [[0, 1, 4], [2, 3], [5]],
        [[0, 4], [1, 2, 3], [5]],
        [[0, 2, 4], [1, 3], [5]],
        [[0, 1, 2, 4], [3], [5]],
        [[0, 1, 2], [3], [4, 5]],
        [[0, 1], [2, 3], [4, 5]],
        [[0], [1, 2, 3], [4, 5]],
        [[0, 2], [1, 3], [4, 5]],
        [[0, 2, 3], [1], [4, 5]],
        [[0, 3], [1, 2], [4, 5]],
        [[0, 1, 3], [2], [4, 5]],
        [[0, 1], [2], [3, 4, 5]],
        [[0], [1, 2], [3, 4, 5]],
        [[0, 2], [1], [3, 4, 5]],
        [[0], [1], [2, 3, 4, 5]],
        [[0], [1, 3], [2, 4, 5]],
        [[0, 3], [1], [2, 4, 5]],
        [[0, 3], [1, 4], [2, 5]],
        [[0], [1, 3, 4], [2, 5]],
        [[0], [1, 4], [2, 3, 5]],
        [[0, 2], [1, 4], [3, 5]],
        [[0], [1, 2, 4], [3, 5]],
        [[0, 1], [2, 4], [3, 5]],
        [[0, 1, 4], [2], [3, 5]],
        [[0, 4], [1, 2], [3, 5]],
        [[0, 2, 4], [1], [3, 5]],
        [[0, 4], [1], [2, 3, 5]],
        [[0, 4], [1, 3], [2, 5]],
        [[0, 3, 4], [1], [2, 5]],
        [[0, 3, 4], [1, 5], [2]],
        [[0, 4], [1, 3, 5], [2]],
        [[0, 4], [1, 5], [2, 3]],
        [[0, 2, 4], [1, 5], [3]],
        [[0, 4], [1, 2, 5], [3]],
        [[0, 1, 4], [2, 5], [3]],
        [[0, 1], [2, 4, 5], [3]],
        [[0], [1, 2, 4, 5], [3]],
        [[0, 2], [1, 4, 5], [3]],
        [[0], [1, 4, 5], [2, 3]],
        [[0], [1, 3, 4, 5], [2]],
        [[0, 3], [1, 4, 5], [2]],
        [[0, 3], [1, 5], [2, 4]],
        [[0], [1, 3, 5], [2, 4]],
        [[0], [1, 5], [2, 3, 4]],
        [[0, 2], [1, 5], [3, 4]],
        [[0], [1, 2, 5], [3, 4]],
        [[0, 1], [2, 5], [3, 4]],
        [[0, 1, 3], [2, 5], [4]],
        [[0, 3], [1, 2, 5], [4]],
        [[0, 2, 3], [1, 5], [4]],
        [[0, 2], [1, 3, 5], [4]],
        [[0], [1, 2, 3, 5], [4]],
        [[0, 1], [2, 3, 5], [4]],
        [[0, 1, 2], [3, 5], [4]],
        [[0, 1, 2, 5], [3], [4]],
        [[0, 1, 5], [2, 3], [4]],
        [[0, 5], [1, 2, 3], [4]],
        [[0, 2, 5], [1, 3], [4]],
        [[0, 2, 3, 5], [1], [4]],
        [[0, 3, 5], [1, 2], [4]],
        [[0, 1, 3, 5], [2], [4]],
        [[0, 1, 5], [2], [3, 4]],
        [[0, 5], [1, 2], [3, 4]],
        [[0, 2, 5], [1], [3, 4]],
        [[0, 5], [1], [2, 3, 4]],
        [[0, 5], [1, 3], [2, 4]],
        [[0, 3, 5], [1], [2, 4]],
        [[0, 3, 5], [1, 4], [2]],
        [[0, 5], [1, 3, 4], [2]],
        [[0, 5], [1, 4], [2, 3]],
        [[0, 2, 5], [1, 4], [3]],
        [[0, 5], [1, 2, 4], [3]],
        [[0, 1, 5], [2, 4], [3]],
        [[0, 1, 4, 5], [2], [3]],
        [[0, 4, 5], [1, 2], [3]],
        [[0, 2, 4, 5], [1], [3]],
        [[0, 4, 5], [1], [2, 3]],
        [[0, 4, 5], [1, 3], [2]],
        [[0, 3, 4, 5], [1], [2]],
    ]


def test_joint_bipartitions():
    mechanism, purview = (0,), (1, 2)
    answer = {
        JointBipartition(Part((), (2,)), Part((0,), (1,))),
        JointBipartition(Part((), (1,)), Part((0,), (2,))),
        JointBipartition(Part((), (1, 2)), Part((0,), ())),
    }
    assert set(joint_bipartitions(mechanism, purview)) == answer


def test_wedge_tripartitions():
    mechanism, purview = (0,), (1, 2)
    assert set(wedge_tripartitions(mechanism, purview)) == {
        JointTripartition(Part((), ()), Part((), (1, 2)), Part((0,), ())),
    }

    mechanism, purview = (3, 4), (5, 6)
    assert set(wedge_tripartitions(mechanism, purview)) == {
        JointTripartition(Part((), ()), Part((), (5, 6)), Part((3, 4), ())),
        JointTripartition(Part((), ()), Part((3,), ()), Part((4,), (5, 6))),
        JointTripartition(Part((), ()), Part((3,), (5,)), Part((4,), (6,))),
        JointTripartition(Part((), ()), Part((3,), (5, 6)), Part((4,), ())),
        JointTripartition(Part((), ()), Part((3,), (6,)), Part((4,), (5,))),
        JointTripartition(Part((), (5,)), Part((3,), ()), Part((4,), (6,))),
        JointTripartition(Part((), (5,)), Part((3,), (6,)), Part((4,), ())),
        JointTripartition(Part((), (6,)), Part((3,), ()), Part((4,), (5,))),
        JointTripartition(Part((), (6,)), Part((3,), (5,)), Part((4,), ())),
    }


def test_partitioned_repertoire_with_tripartition(s):
    tripartition = JointTripartition(Part((), (1,)), Part((0,), ()), Part((), (2,)))

    assert (
        s.partitioned_repertoire(
            Direction.CAUSE,
            tripartition,
            mechanism_measure=resolve_mechanism_measure(
                config.formalism.iit.mechanism_phi_measure
            ),
            state=tuple(s.state[node] for node in tripartition.purview),
        )
        == 0.75
    )


def test_tripartitions_choses_smallest_purview(s):
    mechanism = (1, 2)

    # In phi-tie, chose the smaller purview (0,)
    with config.override(purview_tie_resolution=["PHI", "NEGATIVE_PURVIEW_SIZE"]):
        mie = s.mie(mechanism)
        assert mie.phi == 2.0
        assert mie.purview == (0,)


def test_all_joint_partitions():
    mechanism, purview = (0, 1), (2,)
    assert set(all_joint_partitions(mechanism, purview)) == {
        JointPartition(Part((0, 1), ()), Part((), (2,))),
        JointPartition(Part((0,), (2,)), Part((1,), ()), Part((), ())),
        JointPartition(Part((0,), ()), Part((1,), (2,)), Part((), ())),
    }

    mechanism, purview = (0, 1), (2, 3)
    assert set(all_joint_partitions(mechanism, purview)) == {
        JointPartition(Part((0, 1), ()), Part((), (2, 3))),
        JointPartition(Part((0,), ()), Part((1,), (2, 3)), Part((), ())),
        JointPartition(Part((0,), (2, 3)), Part((1,), ()), Part((), ())),
        JointPartition(Part((0,), ()), Part((1,), (3,)), Part((), (2,))),
        JointPartition(Part((0,), (2,)), Part((1,), ()), Part((), (3,))),
        JointPartition(Part((0,), ()), Part((1,), (2,)), Part((), (3,))),
        JointPartition(Part((0,), (3,)), Part((1,), (2,)), Part((), ())),
        JointPartition(Part((0,), (3,)), Part((1,), ()), Part((), (2,))),
        JointPartition(Part((0,), (2,)), Part((1,), (3,)), Part((), ())),
    }


def test_partition_types():
    assert partition_types["JOINT_BIPARTITION"] == joint_bipartitions
    assert partition_types["WEDGE_TRIPARTITION"] == wedge_tripartitions
    assert partition_types["JOINT_PARTITION_ALL"] == all_joint_partitions
    assert set(partition_types.all()) == {
        "JOINT_BIPARTITION",
        "WEDGE_TRIPARTITION",
        "JOINT_PARTITION_ALL",
    }


def test_bidirectional_cut_matrices_symmetric_and_complete():
    """Every yielded bidirectional cut matrix is symmetric, and the family
    covers all 2^(n(n-1)/2) - 1 nonzero symmetric matrices exactly once."""
    import numpy as np

    from pyphi.partition import _cut_matrices

    for n in range(2, 6):
        matrices = list(_cut_matrices(n, symmetric=True))
        assert all(np.array_equal(m, m.T) for m in matrices), (
            f"n={n}: asymmetric matrix in the bidirectional family"
        )
        distinct = {m.tobytes() for m in matrices}
        expected = 2 ** (n * (n - 1) // 2) - 1
        assert len(matrices) == expected, (n, len(matrices), expected)
        assert len(distinct) == expected, (n, len(distinct), expected)


def test_k_partitions_more_blocks_than_elements_is_empty():
    # There are no partitions of an n-set into k > n nonempty blocks.
    assert list(k_partitions(range(3), 4)) == []
    assert list(k_partitions(range(2), 5)) == []


@pytest.mark.parametrize(
    ("m", "p"),
    [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (4, 3)],
)
def test_all_joint_partitions_yields_unique_cuts(m, p):
    """Every induced edge cut appears exactly once: partitions sharing a cut
    are the same physical partition, and duplicates made identical-cut ties
    unresolvable downstream."""
    mechanism = tuple(range(m))
    purview = tuple(range(m, m + p))
    partitions = list(all_joint_partitions(mechanism, purview))
    keys = [x.lex_key() for x in partitions]
    assert len(keys) == len(set(keys))
    # The memoized sweep counts must match the generator exactly.
    from pyphi.cost import partition_sweep_count

    assert len(partitions) == partition_sweep_count(m, p)


_SYSTEM_SCHEMES = [
    "DIRECTED_BIPARTITION",
    "DIRECTED_BIPARTITION_CUT_ONE",
    "DIRECTED_BIPARTITION_SEQUENTIAL",
    "EDGE_CUT_ALL",
    "EDGE_CUT_BIDIRECTIONAL",
    "DIRECTED_SET_PARTITION",
]


@pytest.mark.parametrize("scheme", _SYSTEM_SCHEMES)
@pytest.mark.parametrize("n", [2, 3, 4])
def test_system_partition_schemes_yield_unique_cuts(scheme, n):
    """No system scheme may yield the same induced edge cut twice: the
    evaluation depends only on the cut, so a repeated cut is the same
    physical partition evaluated again."""
    from pyphi.partition import system_partition_types

    parts = list(system_partition_types[scheme](tuple(range(n))))
    keys = [p.lex_key() for p in parts]
    assert len(keys) == len(set(keys))


def test_directed_bipartition_of_one_single_element_yields_nothing():
    """A single-element sequence has no bipartition with two nonempty parts,
    so no cut exists; the empty-part (no-op) splits must not be yielded."""
    from pyphi.partition import directed_bipartition_of_one

    assert list(directed_bipartition_of_one((0,))) == []
    # Multi-element sequences are unaffected.
    assert list(directed_bipartition_of_one((0, 1))) == [((0,), (1,)), ((1,), (0,))]


def test_directed_bipartition_cut_one_scheme_empty_for_single_node():
    from pyphi.partition import system_partition_types

    parts = list(system_partition_types["DIRECTED_BIPARTITION_CUT_ONE"]((0,)))
    assert parts == []


def test_edge_cut_schemes_marked_as_possibly_non_disconnecting():
    """Both edge-cut schemes can yield cuts that leave the system strongly
    connected, so SIA searches key the Eq. 14 disconnection filter on this
    attribute."""
    from pyphi.partition import system_partition_types

    for scheme in ("EDGE_CUT_ALL", "EDGE_CUT_BIDIRECTIONAL"):
        assert getattr(
            system_partition_types[scheme], "may_yield_non_disconnecting_cuts", False
        )
    for scheme in (
        "DIRECTED_BIPARTITION",
        "DIRECTED_BIPARTITION_CUT_ONE",
        "DIRECTED_BIPARTITION_SEQUENTIAL",
        "DIRECTED_SET_PARTITION",
    ):
        assert not getattr(
            system_partition_types[scheme], "may_yield_non_disconnecting_cuts", False
        )
