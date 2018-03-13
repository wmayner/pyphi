#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_iit_2_0.py

import pytest
import numpy as np

from pyphi import Network
from pyphi.subsystem_2_0 import Subsystem_2_0


@pytest.fixture
def and_network():
    """A 3-node AND network.

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
