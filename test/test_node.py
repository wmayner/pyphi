#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from cyphi.network import Network
from cyphi.node import Node
from cyphi import utils


def test_node_init_tpm(m):
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
    for node in m.nodes:
        assert np.array_equal(node.tpm, answer[node.index])

def test_node_init_inputs(m):
    answer = [
        m.nodes[1:],
        m.nodes[2:3],
        m.nodes[:2]
    ]
    for node in m.nodes:
        assert node.inputs == set(answer[node.index])


def test_node_eq(m):
    assert m.nodes[1] == Node(m, 1)


def test_node_neq_by_index(m):
    assert m.nodes[0] != Node(m, 1)


def test_node_neq_by_network(m):
    other_network = Network(m.tpm, (0, 0, 0), (0, 0, 0))
    assert m.nodes[0] != Node(other_network, 0)
