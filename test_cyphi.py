#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for CyPhi"""

import unittest

import numpy as np
import cyphi.models as models
import cyphi.utils as utils

# TODO: look at using pytest fixtures vs. setUp/tearDown

# NOTE: ``py.test`` and ``nose`` only display ``print`` statements in a test if
# it failed.


class TestUtils(unittest.TestCase):

    def test_combs_for_1D_input(self):
        n, k = 3, 2
        data = np.arange(n)
        assert np.array_equal(
            utils.combs(data, k),
            np.asarray([
                [0, 1],
                [0, 2],
                [1, 2]
            ]))

    def test_comb_indices(self):
        n, k = 3, 2
        data = np.arange(6).reshape(2, 3)
        assert np.array_equal(
            data[:, utils.comb_indices(n, k)],
            np.asarray([[
                [0, 1],
                [0, 2],
                [1, 2]],

               [[3, 4],
                [3, 5],
                [4, 5]]]))

    def test_powerset(self):
        a = np.arange(2)
        assert list(utils.powerset(a)) == [(), (0,), (1,), (0, 1)]


class TestModels(unittest.TestCase):

    def test_network_init(self):
        connectivity_matrix = np.zeros((3, 3))
        tpm = np.zeros((8, 8))
        powerset = utils.powerset(np.arange(3))
        print(powerset)

        network = models.Network(connectivity_matrix, tpm)

        assert np.array_equal(network.connectivity_matrix, connectivity_matrix)
        assert np.array_equal(network.tpm, tpm)
        assert network.powerset == powerset

    def test_network_connectivity_matrix_validation(self):
        connectivity_matrix = np.zeros((2, 3))
        tpm = np.zeros((1, 1))

        with self.assertRaises(utils.ValidationException):
            models.Network(connectivity_matrix, tpm)
