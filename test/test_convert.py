#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_convert.py

import numpy as np
from math import log2

from pyphi import convert


def test_le_index2state():
    assert convert.le_index2state(7, 8) == (1, 1, 1, 0, 0, 0, 0, 0)
    assert convert.le_index2state(1, 3) == (1, 0, 0)
    assert convert.le_index2state(8, 4) == (0, 0, 0, 1)


def test_be_index2state():
    assert convert.be_index2state(7, 8) == (0, 0, 0, 0, 0, 1, 1, 1)
    assert convert.be_index2state(1, 3) == (0, 0, 1)
    assert convert.be_index2state(8, 4) == (1, 0, 0, 0)


state_by_node = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0]
], dtype=float)
state_by_state = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]
], dtype=float)
state_by_node_nondet = np.array([
    [0.0, 0.0],
    [0.5, 0.5],
    [0.5, 0.5],
    [1.0, 1.0]
])
state_by_state_nondet = np.array([
    [1.00, 0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.00, 0.00, 0.00, 1.00]
])
nd_state_by_node = np.array([
    [[[[1, 0, 0, 1],
       [1, 0, 1, 1]],

      [[0, 0, 0, 0],
       [0, 1, 1, 0]]],


     [[[0, 0, 1, 0],
       [0, 0, 0, 0]],

      [[0, 0, 1, 1],
       [0, 1, 0, 1]]]],



    [[[[1, 0, 1, 1],
       [1, 1, 0, 1]],

      [[0, 1, 1, 0],
       [0, 1, 0, 0]]],


     [[[0, 0, 0, 0],
       [0, 1, 1, 0]],

      [[0, 1, 0, 1],
       [0, 1, 1, 1]]]]
], dtype=float)
twod_state_by_node = np.array([
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 1]
])
nonsquare_deterministic_1 = np.array([
    [1, 0, 0],
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 1],
])
nonsquare_deterministic_2 = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0],
])
nonsquare_nondeterministic_1 = np.array([
    [0.8, 0.1, 0.3],
    [0.9, 0.5, 0.3],
    [0.2, 0.8, 0.9],
    [0.3, 0.4, 0.1],
])
nonsquare_nondeterministic_2 = np.array([
    [0.0, 0.0],
    [0.9, 0.1],
    [0.3, 0.8],
    [0.3, 0.1],
    [0.6, 0.3],
    [0.5, 0.6],
    [0.9, 0.2],
    [0.8, 0.1],
])


def test_to_multidimensional():
    # Identity
    assert np.array_equal(convert.to_multidimensional(nd_state_by_node),
                          nd_state_by_node)

    for tpm in [
        state_by_node,
        twod_state_by_node,
        nonsquare_deterministic_1,
        nonsquare_deterministic_2,
        nonsquare_nondeterministic_1,
        nonsquare_nondeterministic_2,
    ]:
        S = tpm.shape[0]
        N = int(log2(S))
        result = convert.to_multidimensional(tpm)
        for i in range(S):
            state = convert.le_index2state(i, N)
            assert np.array_equal(result[state], tpm[i])


def test_to_2dimensional():
    # Identity
    assert np.array_equal(convert.to_2dimensional(state_by_node),
                          state_by_node)
    # Idempotency
    for tpm in [
        state_by_node,
        state_by_node_nondet,
        twod_state_by_node,
        nonsquare_deterministic_1,
        nonsquare_deterministic_2,
        nonsquare_nondeterministic_1,
        nonsquare_nondeterministic_2
    ]:
        nd = convert.to_multidimensional(tpm)
        assert np.array_equal(convert.to_2dimensional(nd), tpm)


def test_state_by_state2state_by_node():
    result = convert.state_by_state2state_by_node(state_by_state)
    expected = convert.to_multidimensional(state_by_node)
    print("Result:")
    print(result)
    print("Expected:")
    print(expected)
    assert np.array_equal(result, expected)


def test_state_by_node2state_by_state():
    sbn_tpm = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    expected = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).astype(float)
    result = convert.state_by_node2state_by_state(sbn_tpm)
    print("Result:")
    print(result)
    print("Expected:")
    print(expected)
    assert np.array_equal(result, expected)


def test_nondet_state_by_node2state_by_state():
    # Test for nondeterministic TPM.
    result = convert.state_by_node2state_by_state(state_by_node_nondet)
    expected = state_by_state_nondet
    print("Result:")
    print(result)
    print("Expected:")
    print(expected)
    assert np.array_equal(result, expected)


def test_nondet_state_by_state2state_by_node():
    # Test for nondeterministic TPM.
    result = convert.state_by_state2state_by_node(state_by_state_nondet)
    expected = convert.to_multidimensional(state_by_node_nondet)
    print("Result:")
    print(result)
    print("Expected:")
    print(expected)
    assert np.array_equal(result, expected)


def test_2_d_state_by_node2state_by_state():
    # Check with 2-D form.
    result = convert.state_by_node2state_by_state(state_by_node)
    expected = state_by_state
    print("Result:")
    print(result)
    print("Expected:")
    print(expected)
    assert np.array_equal(result, expected)


def test_n_d_state_by_node2state_by_state():
    # Check with N-D form.
    sbn = convert.to_multidimensional(state_by_node)
    result = convert.state_by_node2state_by_state(sbn)
    expected = state_by_state
    print("Result:")
    print(result)
    print("Expected:")
    print(expected)
    assert np.array_equal(result, expected)


def test_nonsquare_deterministic_1_state_by_node2state_by_state():
    result = convert.state_by_node2state_by_state(nonsquare_deterministic_1)
    assert np.array_equal(
        result,
        np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 1., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 1., 0., 0., 0.]])
    )


def test_nonsquare_deterministic_2_state_by_node2state_by_state():
    result = convert.state_by_node2state_by_state(nonsquare_deterministic_2)
    assert np.array_equal(
        result,
        np.array([[1., 0., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 0., 1.],
                  [1., 0., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.],
                  [0., 1., 0., 0.]])
    )


def test_nonsquare_nondeterministic_1_state_by_node2state_by_state():
    result = convert.state_by_node2state_by_state(nonsquare_nondeterministic_1)
    assert np.allclose(
        result,
        np.array([[0.126, 0.504, 0.014, 0.056, 0.054, 0.216, 0.006, 0.024],
                  [0.035, 0.315, 0.035, 0.315, 0.015, 0.135, 0.015, 0.135],
                  [0.016, 0.004, 0.064, 0.016, 0.144, 0.036, 0.576, 0.144],
                  [0.378, 0.162, 0.252, 0.108, 0.042, 0.018, 0.028, 0.012]])
    )


def test_nonsquare_nondeterministic_2_state_by_node2state_by_state():
    result = convert.state_by_node2state_by_state(nonsquare_nondeterministic_2)
    assert np.allclose(
        result,
        np.array([[1.  , 0.  , 0.  , 0.  ],
                  [0.09, 0.81, 0.01, 0.09],
                  [0.14, 0.06, 0.56, 0.24],
                  [0.63, 0.27, 0.07, 0.03],
                  [0.28, 0.42, 0.12, 0.18],
                  [0.2 , 0.2 , 0.3 , 0.3 ],
                  [0.08, 0.72, 0.02, 0.18],
                  [0.18, 0.72, 0.02, 0.08]])
    )
