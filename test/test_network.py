#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from pyphi.network import Network
from pyphi.node import Node


@pytest.fixture()
def network():
    size = 3
    current_state = (0, 1, 0)
    past_state = (1, 1, 0)
    tpm = np.ones([2] * size + [size]).astype(float) / 2
    return Network(tpm, current_state, past_state)


def test_network_init_validation(network):
    with pytest.raises(ValueError):
        # Totally wrong shape
        tpm = np.arange(3).astype(float)
        state = (0, 1, 0)
        past_state = (1, 1, 0)
        Network(tpm, state, past_state)
    with pytest.raises(ValueError):
        # Non-binary nodes (4 states)
        tpm = np.ones((4, 4, 4, 3)).astype(float)
        state = (0, 1, 0)
        Network(tpm, state, past_state)
    with pytest.raises(ValueError):
        state = (0, 1)
        Network(network.tpm, state, network.past_state)
    with pytest.raises(ValueError):
        state = (0, 1)
        Network(network.tpm, network.current_state, state)
    # TODO test state validation (are current and past states congruent to
    # TPM?)


def test_network_state_by_state_tpm():
    sbs_tpm = np.array([[0.5, 0.5, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.2, 0.0, 0.8],
                        [0.0, 0.3, 0.7, 0.0]])
    sbn_tpm = np.array([[[0.0, 0.5],
                         [0.0, 1.0]],
                        [[0.8, 1.0],
                         [0.7, 0.3]]])
    state = (0, 0)
    assert (Network(sbs_tpm, state, state) == Network(sbn_tpm, state, state))


def test_repr(standard):
    print(repr(standard))


def test_str(standard):
    print(str(standard))
