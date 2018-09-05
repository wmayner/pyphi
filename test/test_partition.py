#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_partition.py

import itertools

import numpy as np

from pyphi import Direction, config
from pyphi.partition import (directed_bipartition,
                             directed_tripartition_indices, k_partitions,
                             partitions, partition_types, mip_bipartitions,
                             wedge_partitions, all_partitions)

from pyphi.models import Part, KPartition, Bipartition, Tripartition


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
        [[0], [1], [2], [3]]
    ]


def test_directed_bipartition():
    answer = [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)),
              ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,)), ((1, 2, 3), ())]
    assert answer == directed_bipartition((1, 2, 3))
    # Test with empty input
    assert [] == directed_bipartition(())


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
        ((), (), (0, 1))]


def test_k_partition():
    # Special/edge cases
    for n, k in list(itertools.product(range(-1, 2), repeat=2))[:-1]:
        assert list(k_partitions(range(n), k)) == []
    assert list(k_partitions(range(1), 1)) == [[[0]]]
    assert list(k_partitions(range(3), 1)) == [[[0, 1, 2]]]
    assert list(k_partitions(range(3), 3)) == [[[0], [1], [2]]]
    assert list(k_partitions(range(3), 4)) == [[[], [0], [1], [2]]]
    assert list(k_partitions(range(4), 3)) == [
        [[0, 1], [2], [3]],
        [[0], [1, 2], [3]],
        [[0, 2], [1], [3]],
        [[0], [1], [2, 3]],
        [[0], [1, 3], [2]],
        [[0, 3], [1], [2]]
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
        [[0, 2, 3, 4], [1]]]
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
        [[0, 3, 4], [1], [2]]
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
        [[0, 3, 4, 5], [1], [2]]]



def test_mip_bipartitions():
    mechanism, purview = (0,), (1, 2)
    answer = set([
        Bipartition(Part((), (2,)), Part((0,), (1,))),
        Bipartition(Part((), (1,)), Part((0,), (2,))),
        Bipartition(Part((), (1, 2)), Part((0,), ())),
    ])
    assert set(mip_bipartitions(mechanism, purview)) == answer


def test_wedge_partitions():
    mechanism, purview = (0,), (1, 2)
    assert set(wedge_partitions(mechanism, purview)) == set([
        Tripartition(Part((), ()), Part((), (1, 2)), Part((0,), ())),
    ])

    mechanism, purview = (3, 4), (5, 6)
    assert set(wedge_partitions(mechanism, purview)) == set([
        Tripartition(Part((), ()),   Part((),   (5, 6)), Part((3, 4), ())),
        Tripartition(Part((), ()),   Part((3,), ()),     Part((4,), (5, 6))),
        Tripartition(Part((), ()),   Part((3,), (5,)),   Part((4,), (6,))),
        Tripartition(Part((), ()),   Part((3,), (5, 6)), Part((4,), ())),
        Tripartition(Part((), ()),   Part((3,), (6,)),   Part((4,), (5,))),
        Tripartition(Part((), (5,)), Part((3,), ()),     Part((4,), (6,))),
        Tripartition(Part((), (5,)), Part((3,), (6,)),   Part((4,), ())),
        Tripartition(Part((), (6,)), Part((3,), ()),     Part((4,), (5,))),
        Tripartition(Part((), (6,)), Part((3,), (5,)),   Part((4,), ())),
    ])


def test_partitioned_repertoire_with_tripartition(s):
    tripartition = Tripartition(Part((), (1,)), Part((0,), ()), Part((), (2,)))

    assert np.array_equal(
        s.partitioned_repertoire(Direction.CAUSE, tripartition),
        np.array([[[0.25, 0.25], [0.25, 0.25]]]))


def test_tripartitions_choses_smallest_purview(s):
    mechanism = (1, 2)

    with config.override(PICK_SMALLEST_PURVIEW=False):
        mie = s.mie(mechanism)
        assert mie.phi == 0.5
        assert mie.purview == (0, 1)

    s.clear_caches()

    # In phi-tie, chose the smaller purview (0,)
    with config.override(PICK_SMALLEST_PURVIEW=True):
        mie = s.mie(mechanism)
        assert mie.phi == 0.5
        assert mie.purview == (0,)


def test_all_partitions():
    mechanism, purview = (0, 1), (2,)
    assert set(all_partitions(mechanism, purview)) == set([
        KPartition(Part((0, 1), ()), Part((), (2,))),
        KPartition(Part((0,), ()), Part((1,), ()), Part((), (2,))),
        KPartition(Part((0,), (2,)), Part((1,), ()), Part((), ())),
        KPartition(Part((0,), ()), Part((1,), (2,)), Part((), ()))])

    mechanism, purview = (0, 1), (2, 3)
    assert set(all_partitions(mechanism, purview)) == set([
        KPartition(Part((0, 1), ()), Part((), (2, 3))),
        KPartition(Part((0,), ()), Part((1,), (2, 3)), Part((), ())),
        KPartition(Part((0,), (2, 3)), Part((1,), ()), Part((), ())),
        KPartition(Part((0,), ()), Part((1,), ()), Part((), (2, 3))),
        KPartition(Part((0,), ()), Part((1,), (3,)), Part((), (2,))),
        KPartition(Part((0,), (2,)), Part((1,), ()), Part((), (3,))),
        KPartition(Part((0,), ()), Part((1,), (2,)), Part((), (3,))),
        KPartition(Part((0,), (3,)), Part((1,), (2,)), Part((), ())),
        KPartition(Part((0,), (3,)), Part((1,), ()), Part((), (2,))),
        KPartition(Part((0,), (2,)), Part((1,), (3,)), Part((), ()))])


def test_partition_types():
    assert partition_types['BI'] == mip_bipartitions
    assert partition_types['TRI'] == wedge_partitions
    assert partition_types['ALL'] == all_partitions
    assert set(partition_types.all()) == set(['BI', 'TRI', 'ALL'])
