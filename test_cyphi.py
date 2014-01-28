#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for CyPhi"""

from itertools import zip_longest

import numpy as np
import cyphi.models as models
import cyphi.utils as utils
from cyphi.exceptions import ValidationException

import unittest

# TODO look at using pytest fixtures vs. setUp/tearDown

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

    @classmethod
    def setUpClass(cls):
        cls.size = 3
        cls.tpm = np.zeros((2 ** cls.size, cls.size))
        cls.powerset = utils.powerset(np.arange(3))
        cls.network = models.Network(cls.tpm)
        cls.nodes = [models.Node(cls.network, node_index)
                     for node_index in range(cls.network.size)]

    @classmethod
    def tearDownClass(cls):
        pass

    def test_network_tpm_validation(self):
        with self.assertRaises(ValidationException):
            # Network TPMs must be 2-dimensional
            tpm = np.arange(3)
            models.Network(tpm)
        with self.assertRaises(ValidationException):
            tpm = np.array([[1,2,3]])
            models.Network(tpm)

    def test_network_init(self):
        assert np.array_equal(self.tpm, self.network.tpm)
        assert self.size == self.network.size
        assert _generator_equal(self.powerset, self.network.powerset)
        assert self.nodes == self.network.nodes

    def test_purview_max_entropy_distribution(self):
        purview = models.Purview(self.network.nodes[0:2])
        max_ent = purview.max_entropy_distribution()
        assert max_ent.shape == (2,2,1)
        assert np.array_equal(
            max_ent,
            np.divide(np.ones(4), 4).reshape((2,2,1)))
        assert max_ent[0][1][0] == 0.25


def _generator_equal(generator_1, generator_2):
    """Quickly compare two generators for equality"""
    sentinel = object()
    return all(a == b for a, b in zip_longest(generator_1,
                                              generator_2,
                                              fillvalue=sentinel))
