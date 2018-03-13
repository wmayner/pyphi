#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_iit_2_0.py

import pytest
import numpy as np

from pyphi import Network
from pyphi.subsystem_2_0 import Subsystem_2_0


@pytest.fixture
def and_network():
    """A 3-node AND network, from Figure 1.

    Diagram::

                +~~~~~~~+
          +~~~~>|   A   |<~~~~+
          |     | (AND) +~~~+ |
          |     +~~~~~~~+   | |
          |                 | |
          |                 v |
        +~+~~~~~~+      +~~~~~+~+
        |   B    |<~~~~~+   C   |
        | (AND) +~~~~~->| (AND) |
        +~~~~~~~~+      +~~~~~~~+
    """
    tpm = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ])
    return Network(tpm, node_labels="ABC")


@pytest.fixture
def and_whole_system(and_network):
    return Subsystem_2_0(and_network, (0, 0, 1), (0, 1, 2))


def test_prior_repertoire(and_whole_system):
    assert np.array_equal(and_whole_system.prior_repertoire(),
                          np.ones([2, 2, 2]) / 8)


def test_posterior_repertoire(and_whole_system):
    """x_0 = (1, 1, 0) is the unique cause of x_1 = (0, 0, 1)"""
    r = np.array([[[0, 0], [0, 0]], [[0, 0], [1, 0]]])
    assert np.array_equal(and_whole_system.posterior_repertoire(), r)


def test_effective_information(and_whole_system):
    assert and_whole_system.effective_information() == 3


def test_effective_information_and_000(and_network):
    """From Figure 2B, AND-network entering state (0, 0, 0)"""
    system = Subsystem_2_0(and_network, (0, 0, 0), (0, 1, 2))
    assert system.effective_information() == 1
