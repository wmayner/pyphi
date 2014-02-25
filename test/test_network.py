#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import unittest
from itertools import zip_longest

import numpy as np
import cyphi.utils as utils
from cyphi.network import Network
from cyphi.node import Node
from cyphi.exceptions import ValidationException


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.current_state = np.array([0., 1., 0.])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.zeros([2] * self.size + [self.size]).astype(float)
        self.powerset = utils.powerset(np.arange(3))
        self.network = Network(self.tpm,
                               self.current_state,
                               self.past_state)
        self.nodes = [Node(self.network, node_index)
                      for node_index in range(self.network.size)]

    def tearDown(self):
        pass

    def test_network_tpm_validation(self):
        with self.assertRaises(ValidationException):
            tpm = np.arange(3).astype(float)
            state = np.array([0, 1, 0])
            past_state = np.array([1, 1, 0])
            Network(tpm, state, past_state)
        with self.assertRaises(ValidationException):
            tpm = np.array([[1, 2, 3]]).astype(float)
            state = np.array([0, 1, 0])
            Network(tpm, state, past_state)
        # TODO test state validation (are current and past states congruent to
        # TPM?)

    def test_network_init(self):
        assert np.array_equal(self.tpm, self.network.tpm)
        assert self.size == self.network.size
        assert _generator_equal(self.powerset, self.network.powerset)
        assert self.nodes == self.network.nodes


def _generator_equal(generator_1, generator_2):
    sentinel = object()
    return all(a == b for a, b in zip_longest(generator_1,
                                              generator_2,
                                              fillvalue=sentinel))
