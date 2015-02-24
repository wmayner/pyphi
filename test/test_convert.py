#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pyphi import convert


def test_loli_index2state():
    assert convert.loli_index2state(7, 8) == (1, 1, 1, 0, 0, 0, 0, 0)
    assert convert.loli_index2state(1, 3) == (1, 0, 0)
    assert convert.loli_index2state(8, 4) == (0, 0, 0, 1)


def test_holi_index2state():
    assert convert.holi_index2state(7, 8) == (0, 0, 0, 0, 0, 1, 1, 1)
    assert convert.holi_index2state(1, 3) == (0, 0, 1)
    assert convert.holi_index2state(8, 4) == (1, 0, 0, 0)


state_by_node = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0]
])
state_by_state = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]
])
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


def test_to_n_dimensional():
    N = state_by_node.shape[-1]
    S = state_by_node.shape[0]
    result = convert.to_n_dimensional(state_by_node)
    for i in range(S):
        state = convert.loli_index2state(i, N)
        assert np.array_equal(result[state], state_by_node[i])


def test_state_by_state2state_by_node():
    result = convert.state_by_state2state_by_node(state_by_state)
    expected = convert.to_n_dimensional(state_by_node)
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
    expected = convert.to_n_dimensional(state_by_node_nondet)
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
    sbn = convert.to_n_dimensional(state_by_node)
    result = convert.state_by_node2state_by_state(sbn)
    expected = state_by_state
    print("Result:")
    print(result)
    print("Expected:")
    print(expected)
    assert np.array_equal(result, expected)
