#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_connectivity/test_connectivity.py

import numpy as np

from pyphi import connectivity


def test_get_inputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert connectivity.get_inputs_from_cm(0, cm) == (1,)
    assert connectivity.get_inputs_from_cm(1, cm) == (0, 1)
    assert connectivity.get_inputs_from_cm(2, cm) == (1,)


def test_get_outputs_from_cm():
    cm = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
    assert connectivity.get_outputs_from_cm(0, cm) == (1,)
    assert connectivity.get_outputs_from_cm(1, cm) == (0, 1, 2)
    assert connectivity.get_outputs_from_cm(2, cm) == tuple()


def test_causally_significant_nodes():
    cm = np.array([
        [0, 0],
        [1, 0]
    ])
    assert connectivity.causally_significant_nodes(cm) == ()

    cm = np.array([
        [0, 1],
        [1, 0]
    ])
    assert connectivity.causally_significant_nodes(cm) == (0, 1)

    cm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
    ])
    assert connectivity.causally_significant_nodes(cm) == (1, 2)


def test_relevant_connections():
    cm = connectivity.relevant_connections(2, (0, 1), (1,))
    assert np.array_equal(cm, [
        [0, 1],
        [0, 1],
    ])
    cm = connectivity.relevant_connections(3, (0, 1), (0, 2))
    assert np.array_equal(cm, [
        [1, 0, 1],
        [1, 0, 1],
        [0, 0, 0],
    ])


def test_block_cm():
    cm1 = np.array([
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1]
    ])
    cm2 = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    cm3 = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1]
    ])
    cm4 = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0]
    ])
    cm5 = np.array([
        [1, 1],
        [0, 1]
    ])
    assert not connectivity.block_cm(cm1)
    assert connectivity.block_cm(cm2)
    assert connectivity.block_cm(cm3)
    assert not connectivity.block_cm(cm4)
    assert not connectivity.block_cm(cm5)


def test_block_reducible():
    cm1 = np.array([
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    cm2 = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    cm3 = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1]
    ])
    cm4 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert not connectivity.block_reducible(
        cm1,
        tuple(range(cm1.shape[0] - 1)),
        tuple(range(cm1.shape[1])))
    assert connectivity.block_reducible(cm2, (0, 1, 2), (0, 1, 2))
    assert connectivity.block_reducible(cm3, (0, 1), (0, 1, 2, 3, 4))
    assert not connectivity.block_reducible(cm4, (0, 1), (1, 2))


def test_is_strong():
    # Strongly connected
    cm = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])
    assert connectivity.is_strong(cm)

    # Disconnected
    cm = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0]])
    assert not connectivity.is_strong(cm)

    # Weakly connected
    cm = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 1, 0]])
    assert not connectivity.is_strong(cm)

    # Nodes (0, 1) are strongly connected
    cm = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 0]])
    assert connectivity.is_strong(cm, (0, 1))


def test_is_full():
    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    assert not connectivity.is_full(cm, (0,), (0, 1, 2))
    assert not connectivity.is_full(cm, (2,), (2,))
    assert not connectivity.is_full(cm, (0, 1), (1, 2))
    assert connectivity.is_full(cm, (), (0, 1, 2))
    assert connectivity.is_full(cm, (0,), ())
    assert connectivity.is_full(cm, (0, 1), (0, 2))
    assert connectivity.is_full(cm, (1, 2), (1, 2))
    assert connectivity.is_full(cm, (0, 1, 2), (0, 1, 2))


def test_apply_boundary_conditions():
    cm = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 0]])
    answer = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]])
    assert np.array_equal(
        connectivity.apply_boundary_conditions_to_cm((1,), cm), answer)
