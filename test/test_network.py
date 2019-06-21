#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_network.py

import numpy as np
import pytest

from pyphi import Direction, config, exceptions
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

    # Conditionally dependent
    tpm = np.array([
            [1, 0.0, 0.0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.0, 0.0, 1],
    ])
    with config.override(VALIDATE_CONDITIONAL_INDEPENDENCE=False):
        Network(tpm)
    with config.override(VALIDATE_CONDITIONAL_INDEPENDENCE=True):
        with pytest.raises(exceptions.ConditionallyDependentError):
            Network(tpm)


def test_network_creates_fully_connected_cm_by_default():
    tpm = np.zeros((2 * 2 * 2, 3))
    network = Network(tpm, cm=None)
    target_cm = np.ones((3, 3))
    assert np.array_equal(network.cm, target_cm)


def test_potential_purviews(s):
    mechanism = (0,)
    assert (s.network.potential_purviews(Direction.CAUSE, mechanism) ==
            [(1,), (2,), (1, 2)])
    assert (s.network.potential_purviews(Direction.EFFECT, mechanism) ==
            [(2,)])


def test_node_labels(standard):
    labels = ('A', 'B', 'C')
    network = Network(standard.tpm, node_labels=labels)
    assert network.node_labels.labels == labels

    labels = ('A', 'B')  # Too few labels
    with pytest.raises(ValueError):
        Network(standard.tpm, node_labels=labels)

    # Auto-generated labels
    network = Network(standard.tpm, node_labels=None)
    assert network.node_labels.labels == ('n0', 'n1', 'n2')


def test_num_states(standard):
    assert standard.num_states == 8


def test_repr(standard):
    print(repr(standard))


def test_str(standard):
    print(str(standard))


def test_len(standard):
    assert len(standard) == 3


def test_size(standard):
    assert standard.size == 3
