#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_utils.py

from unittest.mock import patch

import numpy as np

from pyphi import constants, utils


def test_all_states():
    assert list(utils.all_states(0)) == []
    assert list(utils.all_states(1)) == [(0,), (1,)]
    states = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]
    assert list(utils.all_states(3)) == states
    assert list(utils.all_states(3, big_endian=True)) == [
        tuple(reversed(state)) for state in states
    ]


def test_eq():
    phi = 0.5
    close_enough = phi - constants.EPSILON / 2
    not_quite = phi - constants.EPSILON * 2
    assert utils.eq(phi, close_enough)
    assert not utils.eq(phi, not_quite)
    assert not utils.eq(phi, (phi - phi))


def test_combs_for_1D_input():
    n, k = 3, 2
    data = np.arange(n)
    assert np.array_equal(utils.combs(data, k),
                          np.asarray([[0, 1],
                                      [0, 2],
                                      [1, 2]]))


def test_combs_r_is_0():
    n, k = 3, 0
    data = np.arange(n)
    assert np.array_equal(utils.combs(data, k), np.asarray([]))


def test_comb_indices():
    n, k = 3, 2
    data = np.arange(6).reshape(2, 3)
    assert np.array_equal(data[:, utils.comb_indices(n, k)],
                          np.asarray([[[0, 1],
                                       [0, 2],
                                       [1, 2]],
                                      [[3, 4],
                                       [3, 5],
                                       [4, 5]]]))


def test_powerset():
    a = np.arange(2)
    assert list(utils.powerset(a)) == [(), (0,), (1,), (0, 1)]


def test_powerset_takes_iterable():
    a = iter([0, 1])
    assert list(utils.powerset(a)) == [(), (0,), (1,), (0, 1)]


def test_np_hashable():
    a = np.ones((2, 2))
    a_hashable = utils.np_hashable(a)
    s = set([a_hashable])
    assert a_hashable in s
    s.add(a_hashable)
    assert len(s) == 1

    b = np.zeros((2, 2))
    b_hashable = utils.np_hashable(b)
    assert b_hashable not in s
    s.add(b_hashable)
    assert len(s) == 2

    c = np.zeros((2, 2))
    c_hashable = utils.np_hashable(c)
    assert c_hashable == b_hashable
    assert c_hashable in s


def test_time_annotated():
    class Timeable:
        time = None

    retval = Timeable()

    @utils.time_annotated
    def func():
        return retval

    with patch('pyphi.utils.time', side_effect=[2, 5]):
        r = func()

    assert r == retval
    assert r.time == 3
