#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from cyphi.network import Network
from cyphi.node import Node



def test_node_init_tpm(standard):
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
    for node in standard.nodes:
        assert np.array_equal(node.tpm, answer[node.index])


def test_node_init_inputs(standard):
    answer = [
        standard.nodes[1:],
        standard.nodes[2:3],
        standard.nodes[:2]
    ]
    for node in standard.nodes:
        assert set(node.inputs) == set(answer[node.index])


def test_node_eq(standard):
    assert standard.nodes[1] == Node(standard, 1)


def test_node_neq_by_index(standard):
    assert standard.nodes[0] != Node(standard, 1)


def test_node_neq_by_network(standard):
    other_network = Network(standard.tpm, (0, 0, 0), (0, 0, 0))
    assert standard.nodes[0] != Node(other_network, 0)


def test_repr(standard):
    print(repr(standard.nodes[0]))


def test_str(standard):
    print(str(standard.nodes[0]))
