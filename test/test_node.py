#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pyphi.network import Network
from pyphi.subsystem import Subsystem
from pyphi.node import Node


def test_node_init_tpm(s):
    answer = [
        np.array([[[[1., 0.],
                    [0., 0.]]],
                  [[[0., 1.],
                    [1., 1.]]]]),
        np.array([[[[1., 0.]]],
                  [[[0., 1.]]]]),
        np.array([[[[1.],
                    [0.]],
                   [[0.],
                    [1.]]],
                  [[[0.],
                    [1.]],
                   [[1.],
                    [0.]]]])
    ]
    for node in s.nodes:
        assert np.array_equal(node.past_tpm, answer[node.index])
        assert np.array_equal(node.current_tpm, answer[node.index])


def test_node_init_inputs(s):
    answer = [
        s.nodes[1:],
        s.nodes[2:3],
        s.nodes[:2]
    ]
    for node in s.nodes:
        assert set(node.inputs) == set(answer[node.index])


def test_node_eq(s):
    assert s.nodes[1] == Node(s.network, 1, s)


def test_node_neq_by_index(s):
    assert s.nodes[0] != Node(s.network, 1, s)


def test_node_neq_by_context(s):
    other_network = Network(s.network.tpm, (0, 0, 0), (0, 0, 0))
    other_s = Subsystem(s.node_indices, other_network)
    assert s.nodes[0] != Node(other_network, 0, other_s)


def test_repr(s):
    print(repr(s.nodes[0]))


def test_str(s):
    print(str(s.nodes[0]))
