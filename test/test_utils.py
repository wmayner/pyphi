#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_utils.py

import numpy as np

from pyphi import utils, constants


def test_phi_eq():
    phi = 0.5
    close_enough = phi - constants.EPSILON / 2
    not_quite = phi - constants.EPSILON * 2
    assert utils.phi_eq(phi, close_enough)
    assert not utils.phi_eq(phi, not_quite)
    assert not utils.phi_eq(phi, (phi - phi))


def test_marginalize_out(s):
    marginalized_distribution = utils.marginalize_out([0], s.tpm)
    assert np.array_equal(marginalized_distribution,
                          np.array([[[[0.0, 0.0, 0.5],
                                      [1.0, 1.0, 0.5]],
                                     [[1.0, 0.0, 0.5],
                                      [1.0, 1.0, 0.5]]]]))

    marginalized_distribution = utils.marginalize_out([0, 1], s.tpm)
    assert np.array_equal(marginalized_distribution,
                          np.array([[[[0.5, 0.0, 0.5],
                                      [1.0, 1.0, 0.5]]]]))


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


def test_get_inputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert utils.get_inputs_from_cm(0, cm) == (1,)
    assert utils.get_inputs_from_cm(1, cm) == (0, 1)
    assert utils.get_inputs_from_cm(2, cm) == (1,)


def test_get_outputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert utils.get_outputs_from_cm(0, cm) == (1,)
    assert utils.get_outputs_from_cm(1, cm) == (0, 1, 2)
    assert utils.get_outputs_from_cm(2, cm) == tuple()


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


def test_state_by_state():
    # State-by-state
    tpm = np.ones((8, 8))
    assert utils.state_by_state(tpm)

    # State-by-node, N-dimensional
    tpm = np.ones((2, 2, 2, 3))
    assert not utils.state_by_state(tpm)

    # State-by-node, 2-dimensional
    tpm = np.ones((8, 3))
    assert not utils.state_by_state(tpm)


def test_expand_tpm():
    tpm = np.ones((2, 1, 2))
    tpm[(0, 0)] = (0, 1)
    assert np.array_equal(utils.expand_tpm(tpm), np.array([
        [[0, 1],
         [0, 1]],
        [[1, 1],
         [1, 1]],
    ]))


def test_causally_significant_nodes():
    cm = np.array([
        [0, 0],
        [1, 0]
    ])
    assert utils.causally_significant_nodes(cm) == ()

    cm = np.array([
        [0, 1],
        [1, 0]
    ])
    assert utils.causally_significant_nodes(cm) == (0, 1)

    cm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ])
    assert utils.causally_significant_nodes(cm) == (1, 2)
