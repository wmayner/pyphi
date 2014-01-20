#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for CyPhi"""

import unittest

import numpy as np
import cyphi

# TODO: look at using pytest fixtures vs. setUp/tearDown

# NOTE: ``py.test`` and ``nose`` only display ``print`` statements in a test if
# it failed.

class TestUtils(unittest.TestCase):

    def test_comb_indices(self):
        n, k = 3, 2
        data = np.arange(6).reshape(2, 3)
        print(data[:, cyphi.comb_indices(n, k)])
        assert 0

    def test_powerset(self):
        n = 3
        k = 2
        print(cyphi.powerset())
        assert 0

class TestModels(unittest.TestCase):

    def test_network_init(self):
        """"""
        connectivity_matrix = np.zeros([3, 3])
        tpm = np.zeros([8, 8])

        network = cyphi.Network(connectivity_matrix, tpm)

        assert np.array_equal(network.connectivity_matrix, connectivity_matrix)
        assert np.array_equal(network.tpm, tpm)
