#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_subsystem.py

import pytest
import example_networks

from pyphi.subsystem import Subsystem
from pyphi.models import Cut
from pyphi import config
import numpy as np


def test_subsystem_validation(s):
    # Wrong state length.
    with pytest.raises(ValueError):
        s = Subsystem(s.network, [0, 0], s.node_indices)
    # Wrong state values.
    with pytest.raises(ValueError):
        s = Subsystem(s.network, [2, 0, 0], s.node_indices)
    # Unreachable state.
    initial_option = config.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI
    config.VALIDATE_NETWORK_STATE = True
    with pytest.raises(ValueError):
        net = example_networks.simple()
        s = Subsystem(net, [1, 1, 1], s.node_indices)
    config.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI = initial_option


def test_empty_init(s):
    # Empty mechanism
    s = Subsystem(s.network, s.state, ())
    assert s.nodes == ()


def test_eq(subsys_n0n2, subsys_n1n2):
    assert subsys_n0n2 == subsys_n0n2
    assert subsys_n0n2 != subsys_n1n2


def test_cmp(subsys_n0n2, subsys_n1n2, s):
    assert s > subsys_n0n2
    assert s > subsys_n1n2
    assert subsys_n0n2 >= subsys_n1n2
    assert s >= subsys_n0n2
    assert subsys_n0n2 < s
    assert subsys_n1n2 < s
    assert subsys_n0n2 <= s
    assert subsys_n0n2 <= subsys_n1n2


def test_len(s, big_subsys_0_thru_3, big_subsys_all):
    assert len(s) == 3
    assert len(big_subsys_0_thru_3) == 4
    assert len(big_subsys_all) == 5


def test_hash(s):
    print(hash(s))


def test_find_cut_matrix(s, big_subsys_0_thru_3):
    cut = Cut((0, ), (1, 2))
    cut_s = Subsystem(
        s.network, s.state, s.node_indices, cut=cut, mice_cache=s._mice_cache)
    answer_s = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    cut = Cut((0, 1), (2, 3))
    cut_big = Subsystem(big_subsys_0_thru_3.network,
                        big_subsys_0_thru_3.state,
                        big_subsys_0_thru_3.node_indices,
                        cut=cut,
                        mice_cache=big_subsys_0_thru_3._mice_cache)
    answer_big = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    assert np.array_equal(cut_s.cut_matrix, answer_s)
    assert np.array_equal(cut_big.cut_matrix, answer_big)
