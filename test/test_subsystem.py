#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_subsystem.py

import numpy as np
import pytest

import example_networks
from pyphi import config, validate
from pyphi.models import Cut, Part
from pyphi.subsystem import Subsystem


@config.override(VALIDATE_SUBSYSTEM_STATES=True)
def test_subsystem_validation(s):
    # Wrong state length.
    with pytest.raises(ValueError):
        s = Subsystem(s.network, (0, 0), s.node_indices)
    # Wrong state values.
    with pytest.raises(ValueError):
        s = Subsystem(s.network, (2, 0, 0), s.node_indices)
    # Disallow impossible states at subsystem level (we don't want to return a
    # phi-value associated with an impossible state).
    net = example_networks.simple()
    with pytest.raises(validate.StateUnreachableError):
        s = Subsystem(net, (0, 1, 0), s.node_indices)


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


def test_indices2nodes(s):
    subsys = s  # 3-node subsystem
    assert subsys.indices2nodes(()) == ()
    assert subsys.indices2nodes((1,)) == (subsys.nodes[1],)
    assert subsys.indices2nodes((0, 2)) == (subsys.nodes[0], subsys.nodes[2])


def test_indices2nodes_with_bad_indices(subsys_n1n2):
    with pytest.raises(ValueError):
        subsys_n1n2.indices2nodes((3, 4))  # indices not in network
    with pytest.raises(ValueError):
        subsys_n1n2.indices2nodes((0,))  # index n0 in network but not subsytem


def test_mip_bipartition():
    mechanism, purview = (0,), (1, 2)
    answer = [
        (Part((), (2,)), Part((0,), (1,))),
        (Part((), (1,)), Part((0,), (2,))),
        (Part((), (1, 2)), Part((0,), ())),
    ]
    assert set(Subsystem._mip_bipartition(mechanism, purview)) == set(answer)


def test_is_cut(s):
    assert s.is_cut() is False
    s = Subsystem(s.network, s.state, s.node_indices, cut=Cut((0,), (1, 2)))
    assert s.is_cut() is True

