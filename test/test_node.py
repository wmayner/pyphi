#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from cyphi.network import Network
from cyphi.node import Node


def test_node_init(m):
    node_tpm = m.nodes[2].tpm
    answer = m.tpm[:, :, :, 2]
    assert np.array_equal(node_tpm, answer)


def test_node_eq(m):
    assert m.nodes[1] == Node(m, 1)


def test_node_neq_by_index(m):
    assert m.nodes[0] != Node(m, 1)


def test_node_neq_by_network(m):
    other_network = Network(m.tpm, np.array([0, 0, 0]), m.past_state)
    assert m.nodes[0] != Node(other_network, 0)
