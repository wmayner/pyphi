#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_macro.py

import numpy as np
import pyphi
from pyphi import macro, utils


def test_list_all_partitions():
    empty_net = pyphi.Network(np.array([]), (), ())
    net = pyphi.examples.macro_network()
    assert macro.list_all_partitions(empty_net.size) == ()
    assert macro.list_all_partitions(net.size) == (
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
    mapping = utils.make_mapping(partition, grouping)
    assert np.array_equal(mapping, np.array(
        (0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 2., 2., 2., 3.)))
    partition = ((0, 1), (2,))
    grouping = (((0, 2), (1,)), ((0,), (1,)))
    mapping = utils.make_mapping(partition, grouping)
    assert np.array_equal(mapping, np.array((0., 1., 1., 0., 2., 3., 3., 2.)))
    partition = ((0, 1, 2),)
    grouping = (((0, 3), (1, 2)),)
    mapping = utils.make_mapping(partition, grouping)
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
    macro_tpm = utils.make_macro_tpm(micro_tpm, mapping)
    assert np.array_equal(answer_tpm, macro_tpm)
    micro_tpm = np.zeros((8, 8)) + 0.125
    macro_tpm = utils.make_macro_tpm(micro_tpm, mapping)
    assert np.array_equal(answer_tpm, macro_tpm)
