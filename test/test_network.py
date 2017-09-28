#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_network.py

import numpy as np
import pytest

from pyphi import Direction
from pyphi.network import Network


@pytest.fixture()
def network():
    size = 3
    tpm = np.ones([2] * size + [size]).astype(float) / 2
    return Network(tpm)


def test_network_init_validation(network):
    with pytest.raises(ValueError):
        # Totally wrong shape
        tpm = np.arange(3).astype(float)
        Network(tpm)
    with pytest.raises(ValueError):
        # Non-binary nodes (4 states)
        tpm = np.ones((4, 4, 4, 3)).astype(float)
        Network(tpm)


def test_network_creates_fully_connected_cm_by_default():
    tpm = np.zeros((2*2*2, 3))
    network = Network(tpm, connectivity_matrix=None)
    target_cm = np.ones((3, 3))
    assert np.array_equal(network.connectivity_matrix, target_cm)


def test_potential_purviews(s):
    mechanism = (0,)
    assert (s.network.potential_purviews(Direction.PAST, mechanism)
            == [(1,), (2,), (1, 2)])
    assert (s.network.potential_purviews(Direction.FUTURE, mechanism)
            == [(2,)])


def test_node_labels(standard):
    labels = ('A', 'B', 'C')
    network = Network(standard.tpm, node_labels=labels)
    assert network.node_labels == labels

    labels = ('A', 'B')  # Too few labels
    with pytest.raises(ValueError):
        Network(standard.tpm, node_labels=labels)

    # Auto-generated labels
    network = Network(standard.tpm, node_labels=None)
    assert network.node_labels == ('n0', 'n1', 'n2')


def test_labels2indices(standard):
    network = Network(standard.tpm, node_labels=('A', 'B', 'C'))
    assert network.labels2indices(('A', 'B')) == (0, 1)
    assert network.labels2indices(('A', 'C')) == (0, 2)


def test_indices2labels(standard):
    # Example labels
    assert standard.indices2labels((0, 1)) == ('A', 'B')
    assert standard.indices2labels((0, 2)) == ('A', 'C')
    # Default labels
    network = Network(standard.tpm)
    assert network.indices2labels((0, 1)) == ('n0', 'n1')
    assert network.indices2labels((0, 2)) == ('n0', 'n2')


def test_parse_node_indices(standard):
    network = Network(standard.tpm, node_labels=('A', 'B', 'C'))
    assert network.parse_node_indices(('B', 'A')) == (0, 1)
    assert network.parse_node_indices((0, 2, 1)) == (0, 1, 2)
    assert standard.parse_node_indices(()) == ()  # No labels - regression

    with pytest.raises(ValueError):
        network.parse_node_indices((0, 'A'))


def test_num_states(standard):
    assert standard.num_states == 8


def test_repr(standard):
    print(repr(standard))


def test_str(standard):
    print(str(standard))
