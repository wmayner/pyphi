#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_macro.py

import numpy as np
from pyphi import macro


def test_list_all_partitions():
    assert macro.list_all_partitions(()) == ()
    assert macro.list_all_partitions((0, 1, 2, 3)) == (
        ((0, 1, 2), (3,)),
        ((0, 1, 3), (2,)),
        ((0, 1), (2, 3)),
        ((0, 1), (2,), (3,)),
        ((0, 2, 3), (1,)),
        ((0, 2), (1, 3)),
        ((0, 2), (1,), (3,)),
        ((0, 3), (1, 2)),
        ((0,), (1, 2, 3)),
        ((0,), (1, 2), (3,)),
        ((0, 3), (1,), (2,)),
        ((0,), (1, 3), (2,)),
        ((0,), (1,), (2, 3)),
        ((0, 1, 2, 3),)
    )


def test_list_all_groupings():
    assert macro.list_all_groupings(()) == (tuple(),)
    partition = ((0, 1), (2, 3))
    assert macro.list_all_groupings(partition) == (
        (((0, 1), (2,)), ((0, 1), (2,))),
        (((0, 1), (2,)), ((0, 2), (1,))),
        (((0, 1), (2,)), ((0,), (1, 2))),
        (((0, 2), (1,)), ((0, 1), (2,))),
        (((0, 2), (1,)), ((0, 2), (1,))),
        (((0, 2), (1,)), ((0,), (1, 2))),
        (((0,), (1, 2)), ((0, 1), (2,))),
        (((0,), (1, 2)), ((0, 2), (1,))),
        (((0,), (1, 2)), ((0,), (1, 2)))
    )


def test_make_mapping():
    partition = ((0, 1), (2, 3))
    grouping = (((0, 1), (2,)), ((0, 1), (2,)))
    mapping = macro.make_mapping(partition, grouping)
    assert np.array_equal(mapping, np.array(
        (0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 2., 2., 2., 3.)))
    partition = ((0, 1), (2,))
    grouping = (((0, 2), (1,)), ((0,), (1,)))
    mapping = macro.make_mapping(partition, grouping)
    assert np.array_equal(mapping, np.array((0., 1., 1., 0., 2., 3., 3., 2.)))
    partition = ((0, 1, 2),)
    grouping = (((0, 3), (1, 2)),)
    mapping = macro.make_mapping(partition, grouping)
    assert np.array_equal(mapping, np.array((0., 1., 1., 1., 1., 1., 1., 0.)))


def test_make_macro_tpm():
    answer_tpm = np.array([
        [.375, .125, .375, .125],
        [.375, .125, .375, .125],
        [.375, .125, .375, .125],
        [.375, .125, .375, .125]
    ])
    mapping = np.array([0., 0., 0., 1., 2., 2., 2., 3.])
    micro_tpm = np.zeros((8, 3)) + 0.5
    macro_tpm = macro.make_macro_tpm(micro_tpm, mapping)
    assert np.array_equal(answer_tpm, macro_tpm)
    micro_tpm = np.zeros((8, 8)) + 0.125
    macro_tpm = macro.make_macro_tpm(micro_tpm, mapping)
    assert np.array_equal(answer_tpm, macro_tpm)


def test_coarse_grain_indices():
    output_grouping = ((1, 2),)  # Node 0 not in system
    state_grouping = (((0,), (1, 2)))
    cg = macro.CoarseGrain(output_grouping, state_grouping)
    assert cg.micro_indices == (1, 2)
    assert cg.macro_indices == (0,)
    assert cg.reindex() == macro.CoarseGrain(((0, 1),), state_grouping)
