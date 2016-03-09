#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_node.py

import numpy as np

from pyphi.network import Network
from pyphi.subsystem import Subsystem
from pyphi.node import Node, expand_node_tpm


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
        assert np.array_equal(node.tpm, answer[node.index])


def test_node_init_inputs(s):
    answer = [
        s.nodes[1:],
        s.nodes[2:3],
        s.nodes[:2]
    ]
    for node in s.nodes:
        assert set(node.inputs) == set(answer[node.index])


def test_node_eq(s):
    assert s.nodes[1] == Node(s, 1)


def test_node_neq_by_index(s):
    assert s.nodes[0] != Node(s, 1)


def test_node_neq_by_context(s):
    other_network = Network(s.network.tpm)
    other_s = Subsystem(other_network, (0, 0, 0), s.node_indices)
    assert s.nodes[0] != Node(other_s, 0)


def test_repr(s):
    print(repr(s.nodes[0]))


def test_str(s):
    print(str(s.nodes[0]))


def test_expand_tpm():
    tpm = np.array([
        [[0, 1]]
    ])
    answer = np.array([
        [[0, 1],
         [0, 1]],
        [[0, 1],
         [0, 1]]
    ])
    assert np.array_equal(expand_node_tpm(tpm), answer)
