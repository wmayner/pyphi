#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_network.py

import pytest
import numpy as np

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
    assert (s.network._potential_purviews('past', mechanism) ==
            [(1,), (2,), (1, 2)])
    assert (s.network._potential_purviews('future', mechanism) ==
            [(2,)])


def test_repr(standard):
    print(repr(standard))


def test_str(standard):
    print(str(standard))
