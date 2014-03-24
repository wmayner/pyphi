#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from itertools import zip_longest

import numpy as np
from cyphi.network import Network
from cyphi.node import Node


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.current_state = np.array([0., 1., 0.])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.zeros([2] * self.size + [self.size]).astype(float)
        self.network = Network(self.tpm,
                               self.current_state,
                               self.past_state)
        self.nodes = [Node(self.network, node_index)
                      for node_index in range(self.network.size)]

    def tearDown(self):
        pass

    def test_network_init_validation(self):
        with self.assertRaises(ValueError):
            # Totally wrong shape
            tpm = np.arange(3).astype(float)
            state = np.array([0, 1, 0])
            past_state = np.array([1, 1, 0])
            Network(tpm, state, past_state)
        with self.assertRaises(ValueError):
            # Non-binary nodes (4 states)
            tpm = np.ones((4, 4, 4, 3)).astype(float)
            state = np.array([0, 1, 0])
            Network(tpm, state, past_state)
        with self.assertRaises(ValueError):
            state = np.array([0, 1])
            Network(self.tpm, state, self.past_state)
        with self.assertRaises(ValueError):
            state = np.array([0, 1])
            Network(self.tpm, self.current_state, state)
        # TODO test state validation (are current and past states congruent to
        # TPM?)

    def test_network_init(self):
        assert np.array_equal(self.tpm, self.network.tpm)
        assert self.size == self.network.size
        assert self.nodes == self.network.nodes
