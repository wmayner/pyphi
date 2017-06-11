#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_utils.py

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
    assert list(utils.all_states(3, holi=True)) == [
        tuple(reversed(state)) for state in states
    ]


def test_phi_eq():
    phi = 0.5
    close_enough = phi - constants.EPSILON / 2
    not_quite = phi - constants.EPSILON * 2
    assert utils.phi_eq(phi, close_enough)
    assert not utils.phi_eq(phi, not_quite)
    assert not utils.phi_eq(phi, (phi - phi))


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
