#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from cyphi.node import Node
from cyphi.network import Network


class TestNode(unittest.TestCase):

    def setUp(self):
        self.current_state = np.array([1, 0, 0])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.array([[0, 0, 0],
                             [0, 0, 1],
                             [1, 0, 1],
                             [1, 0, 0],
                             [1, 1, 0],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 0]]).reshape([2] * 3 + [3],
                                                 order="F").astype(float)
        self.network = Network(self.tpm,
                                      self.current_state,
                                      self.past_state)

    def test_node_init(self):
        tpm = self.network.nodes[2].tpm
        answer = self.network.tpm[:, :, :, 2]

        print("TPM", tpm, "Answer", answer, "", sep="\n\n")

        assert np.array_equal(tpm, answer)

    def test_node_eq(self):
        assert self.network.nodes[1] == Node(self.network, 1)

    def test_node_neq_by_index(self):
        assert self.network.nodes[0] != Node(self.network, 1)

    def test_node_neq_by_network(self):
        other_network = Network(self.tpm, np.array([0, 0, 0]),
                                       self.past_state)
        assert self.network.nodes[0] != Node(other_network, 0)
