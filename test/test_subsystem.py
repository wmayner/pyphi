#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_subsystem.py

from unittest import mock

import numpy as np
import pytest

import example_networks
from pyphi import config, Network, utils, validate
from pyphi.models import Cut, Part
from pyphi.subsystem import Subsystem, mip_bipartitions


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


def test_validate_cut_nodes_equal_subsystem_nodes(s):
    assert s.node_indices == (0, 1, 2)

    cut = Cut((0,), (1, 2))  # A-ok
    Subsystem(s.network, s.state, s.node_indices, cut=cut)

    cut = Cut((0,), (1,))  # missing node 2 in cut
    with pytest.raises(ValueError):
        Subsystem(s.network, s.state, s.node_indices, cut=cut)

    cut = Cut((0,), (1, 2))  # missing node 2 in subsystem
    with pytest.raises(ValueError):
        Subsystem(s.network, s.state, (0, 1), cut=cut)


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


def test_size(s, big_subsys_0_thru_3, big_subsys_all):
    assert s.size == 3
    assert big_subsys_0_thru_3.size == 4
    assert big_subsys_all.size == 5


def test_hash(s):
    print(hash(s))


def test_find_cut_matrix(s, big_subsys_0_thru_3):
    cut = Cut((0, ), (1, 2))
    cut_s = Subsystem(s.network, s.state, s.node_indices, cut=cut)
    answer_s = np.array([
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])
    assert np.array_equal(cut_s.cut_matrix, answer_s)

    cut = Cut((0, 1), (2, 3))
    cut_big = Subsystem(big_subsys_0_thru_3.network,
                        big_subsys_0_thru_3.state,
                        big_subsys_0_thru_3.node_indices,
                        cut=cut)
    answer_big = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    assert np.array_equal(cut_big.cut_matrix, answer_big)

    null_cut_matrix = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])
    assert np.array_equal(s.cut_matrix, null_cut_matrix)


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
    assert set(mip_bipartitions(mechanism, purview)) == set(answer)


def test_is_cut(s):
    assert s.is_cut is False
    s = Subsystem(s.network, s.state, s.node_indices, cut=Cut((0,), (1, 2)))
    assert s.is_cut is True


def test_proper_state(s, subsys_n0n2):
    # subsys_n0n2.state == (1, 0, 0)
    assert subsys_n0n2.proper_state == (1, 0)
    subsys_n0n2.proper_state = (0, 1)
    assert subsys_n0n2.proper_state == (0, 1)
    assert subsys_n0n2.state == (0, 0, 1)


def test_apply_cut(s):
    cut = Cut((0, 1), (2,))
    cut_s = s.apply_cut(cut)
    assert s.network == cut_s.network
    assert s.state == cut_s.state
    assert s.node_indices == cut_s.node_indices
    assert np.array_equal(cut_s.tpm, s.tpm)
    assert np.array_equal(cut_s.connectivity_matrix,
                          utils.apply_cut(cut, s.connectivity_matrix))


def test_cut_indices(s, subsys_n1n2):
    assert s.cut_indices == (0, 1, 2)
    assert subsys_n1n2.cut_indices == (1, 2)


def test_specify_elements_with_labels(standard):
    network = Network(standard.tpm, node_labels=('A', 'B', 'C'))
    subsystem = Subsystem(network, (0, 0, 0), ('B', 'C'))
    assert subsystem.node_indices == (1, 2)
    assert tuple(node.label for node in subsystem.nodes) == ('B', 'C')
    assert str(subsystem) == 'Subsystem((B, C))'
