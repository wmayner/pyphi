#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from cyphi.network import Network
from cyphi.node import Node


@pytest.fixture()
def network():
    size = 3
    current_state = np.array([0., 1., 0.])
    past_state = np.array([1, 1, 0])
    tpm = np.ones([2] * size + [size]).astype(float) / 2
    return Network(tpm, current_state, past_state)


def test_network_init_validation(network):
    with pytest.raises(ValueError):
        # Totally wrong shape
        tpm = np.arange(3).astype(float)
        state = np.array([0, 1, 0])
        past_state = np.array([1, 1, 0])
        Network(tpm, state, past_state)
    with pytest.raises(ValueError):
        # Non-binary nodes (4 states)
        tpm = np.ones((4, 4, 4, 3)).astype(float)
        state = np.array([0, 1, 0])
        Network(tpm, state, past_state)
    with pytest.raises(ValueError):
        state = np.array([0, 1])
        Network(network.tpm, state, network.past_state)
    with pytest.raises(ValueError):
        state = np.array([0, 1])
        Network(network.tpm, network.current_state, state)
    # TODO test state validation (are current and past states congruent to
    # TPM?)


def test_network_init_nodes(network):
    nodes = [Node(network, node_index) for node_index in range(network.size)]
    assert nodes == network.nodes
