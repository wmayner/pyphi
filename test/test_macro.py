#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_macro.py

import pytest

import numpy as np
from pyphi import macro


def test_all_partitions():
    assert list(macro.all_partitions(())) == []
    assert list(macro.all_partitions((0, 1, 2, 3))) == [
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
    ]


def test_all_groupings():
    assert list(macro.all_groupings(())) == [()]
    partition = ((0, 1), (2, 3))
    assert list(macro.all_groupings(partition)) == [
        (((0, 1), (2,)), ((0, 1), (2,))),
        (((0, 1), (2,)), ((0, 2), (1,))),
        (((0, 1), (2,)), ((0,), (1, 2))),
        (((0, 2), (1,)), ((0, 1), (2,))),
        (((0, 2), (1,)), ((0, 2), (1,))),
        (((0, 2), (1,)), ((0,), (1, 2))),
        (((0,), (1, 2)), ((0, 1), (2,))),
        (((0,), (1, 2)), ((0, 2), (1,))),
        (((0,), (1, 2)), ((0,), (1, 2)))
    ]


def test_all_coarse_grainings():
    assert tuple(macro.all_coarse_grains((1,))) == (
        macro.CoarseGrain(partition=((1,),),
                          grouping=(((0,), (1,)),)),)


def test_all_blackboxes():
    assert list(macro.all_blackboxes((1, 2))) == [
        macro.Blackbox((), (1, 2)),
        macro.Blackbox((1,), (2,)),
        macro.Blackbox((2,), (1,)),
        macro.Blackbox((1, 2), ()),
    ]


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
    partition = ((1, 2),)  # Node 0 not in system
    grouping = (((0,), (1, 2)),)
    cg = macro.CoarseGrain(partition, grouping)
    assert cg.micro_indices == (1, 2)
    assert cg.macro_indices == (0,)
    assert cg.reindex() == macro.CoarseGrain(((0, 1),), grouping)


def test_coarse_grain_state():
    partition = ((0, 1),)
    grouping = (((0,), (1, 2)),)
    cg = macro.CoarseGrain(partition, grouping)
    with pytest.raises(AssertionError):
        assert cg.macro_state((1, 1, 0)) == (1,)

    assert cg.macro_state((0, 0)) == (0,)
    assert cg.macro_state((0, 1)) == (1,)
    assert cg.macro_state((1, 1)) == (1,)

    partition = ((1,), (2,))
    grouping = (((0,), (1,)), ((1,), (0,)))
    cg = macro.CoarseGrain(partition, grouping)
    assert cg.macro_state((0, 1)) == (0, 0)
    assert cg.macro_state((1, 1)) == (1, 0)


def test_blackbox_indices():
    bb = macro.Blackbox((1,), (3, 4))
    assert bb.micro_indices == (1, 3, 4)
    assert bb.macro_indices == (0, 1)
    assert bb.reindex() == macro.Blackbox((0,), (1, 2))
