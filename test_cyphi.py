# -*- coding: utf-8 -*-

"""Tests for CyPhi"""

import unittest

import numpy as np
import cyphi

# TODO: look at using pytest fixtures vs. setUp/tearDown

# NOTE: ``py.test`` and ``nose`` only display ``print`` statements in a test if
# it failed.


class TestModels(unittest.TestCase):

    def test_network_init(self):
        """"""
        connectivity_matrix = np.zeros([3, 3])
        tpm = np.zeros([8, 8])

        network = cyphi.Network(connectivity_matrix, tpm)

        assert np.array_equal(network.connectivity_matrix, connectivity_matrix)
        assert np.array_equal(network.tpm, tpm)
