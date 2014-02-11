#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for CyPhi"""

from itertools import zip_longest

import numpy as np
import cyphi.models as models
import cyphi.utils as utils
from cyphi.exceptions import ValidationException

import unittest

# TODO look into using pytest fixtures vs. setUp/tearDown

# NOTE: ``py.test`` and ``nose`` only display ``print`` statements in a test if
# it failed.


class TestUtils(unittest.TestCase):

    def test_combs_for_1D_input(self):
        n, k = 3, 2
        data = np.arange(n)
        assert np.array_equal(
            utils.combs(data, k),
            np.asarray([[0, 1],
                        [0, 2],
                        [1, 2]]))

    def test_comb_indices(self):
        n, k = 3, 2
        data = np.arange(6).reshape(2, 3)
        assert np.array_equal(
            data[:, utils.comb_indices(n, k)],
            np.asarray([[[0, 1],
                         [0, 2],
                         [1, 2]],

                        [[3, 4],
                         [3, 5],
                         [4, 5]]]))

    def test_powerset(self):
        a = np.arange(2)
        assert list(utils.powerset(a)) == [(), (0,), (1,), (0, 1)]


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.state = np.array([0, 1, 0])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.zeros([2] * self.size + [self.size])
        self.powerset = utils.powerset(np.arange(3))
        self.network = models.Network(self.tpm, self.state, self.past_state)
        self.nodes = [models.Node(self.network, node_index)
                      for node_index in range(self.network.size)]

    def tearDown(self):
        pass

    def test_network_tpm_validation(self):
        with self.assertRaises(ValidationException):
            tpm = np.arange(3)
            state = np.array([0, 1, 0])
            past_state = np.array([1, 1, 0])
            models.Network(tpm, state, past_state)
        with self.assertRaises(ValidationException):
            tpm = np.array([[1, 2, 3]])
            state = np.array([0, 1, 0])
            models.Network(tpm, state, past_state)
        # TODO test state validation (are current and past states congruent to
        # TPM?)

    def test_network_init(self):
        assert np.array_equal(self.tpm, self.network.tpm)
        assert self.size == self.network.size
        assert _generator_equal(self.powerset, self.network.powerset)
        assert self.nodes == self.network.nodes


class TestNode(unittest.TestCase):

    def setUp(self):
        self.state = np.array([1, 0, 0])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.array([[0, 0, 0],
                             [0, 0, 1],
                             [1, 0, 1],
                             [1, 0, 0],
                             [1, 1, 0],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 0]]).reshape([2] * 3 + [3])
        self.network = models.Network(self.tpm, self.state, self.past_state)

    def test_node_init(self):
        assert np.array_equal(self.network.nodes[0].tpm,
                              np.array(self.network.tpm[:, :, :, 0]))

    def test_node_eq(self):
        assert self.network.nodes[1] == models.Node(self.network, 1)

    def test_node_neq_by_index(self):
        assert self.network.nodes[0] != models.Node(self.network, 1)

    def test_node_neq_by_network(self):
        other_network = models.Network(self.tpm, np.array([0, 0, 0]),
                                       self.past_state)
        assert self.network.nodes[0] != models.Node(other_network, 0)

    def tearDown(self):
        pass


class TestMechanism(unittest.TestCase):

    def setUp(self):
        # Matlab default network
        # ======================
        self.state = np.array([1, 0, 0])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.array([[0, 0, 0],
                             [0, 0, 1],
                             [1, 0, 1],
                             [1, 0, 0],
                             [1, 1, 0],
                             [1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 0]]).reshape([2] * 3 + [3])
        self.network = models.Network(self.tpm, self.state, self.past_state)

        # Simple 'AND' network
        # ==================
        #       +---+
        #   +-->| A |<--+
        #   |   +---+   |
        #   |    AND    |
        # +-+-+       +-+-+
        # | B |       | C |
        # +---+       +---+
        self.s_state = np.array([1, 0, 0])
        self.s_past_state = np.array([0, 1, 1])
        self.s_tpm = np.array([
        # Current state | Past state
        #    A, B, C    |  A, B, C
        # --------------#-----------
            [0, 0, 0],  #  0, 0, 0
            [0, 0, 0],  #  0, 0, 1
            [0, 0, 0],  #  0, 1, 0
            [1, 0, 0],  #  0, 1, 1
            [0, 0, 0],  #  1, 0, 0
            [0, 0, 0],  #  1, 0, 1
            [0, 0, 0],  #  1, 1, 0
            [0, 0, 0]   #  1, 1, 1
            ]).reshape([2] * 3 + [3])
        self.s_network = models.Network(self.s_tpm,
                                        self.s_state,
                                        self.s_past_state)

    def tearDown(self):
        pass

    def test_empty_init(self):
        mechanism = models.Mechanism([],
                                     self.network.state,
                                     self.network.past_state,
                                     self.network)
        assert mechanism.nodes == []

    def test_marginalize_out(self):
        marginalized_distribution = models.Mechanism._marginalize_out(
            self.network.nodes[0], self.network.tpm)
        assert np.array_equal(marginalized_distribution,
                              np.array([[[[0.5,  0.5,  0.0],
                                          [0.5,  0.5,  1.0]],

                                         [[1.0,  0.5,  1.0],
                                          [1.0,  0.5,  0.0]]]]))

    def test_cause_repertoire(self):

        # Test against Matlab default network
        # ===================================

        # Mechanism(['n0'])
        mechanism = models.Mechanism([self.network.nodes[0]],
                                     self.network.state,
                                     self.network.past_state,
                                     self.network)
        # Subsystem(['n0'])
        purview = models.Subsystem([self.network.nodes[0]], self.network)
        assert np.array_equal(mechanism.cause_repertoire(purview),
                              utils.uniform_distribution(self.network.size))

        # Mechanism(['n0', 'n1'])
        mechanism = models.Mechanism(self.network.nodes[0:2],
                                     self.network.state,
                                     self.network.past_state,
                                     self.network)
        # Subsystem(['n0', 'n2'])
        purview = models.Subsystem(self.network.nodes[0:3:2], self.network)
        assert np.array_equal(mechanism.cause_repertoire(purview),
                              np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2))

        # Test against simple 'AND' network
        # =================================

        a_just_turned_on = np.array([1, 0, 0])
        a_about_to_be_on = np.array([0, 1, 1])
        all_off = np.array([0, 0, 0])

        # Mechanism(['n0']), 'A' just turned on
        mechanism = models.Mechanism([self.s_network.nodes[0]],
                                     a_just_turned_on,
                                     a_about_to_be_on,
                                     self.s_network)
        # Subsystem(['n0', 'n2', 'n3'])
        purview = models.Subsystem(self.s_network.nodes, self.s_network)
        # Cause repertoire is maximally selective; past state must have been
        # {0,1,1}
        answer = np.array([[[0., 0.],
                            [0., 1.]],
                           [[0., 0.],
                            [0., 0.]]])
        assert np.array_equal(mechanism.cause_repertoire(purview), answer)

        # Mechanism(['n0']), all nodes off
        mechanism = models.Mechanism([self.s_network.nodes[0]],
                                     all_off,
                                     all_off,
                                     self.s_network)
        # Cause repertoire is minimally selective; only {0,1,1} is ruled out
        answer = np.ones((2, 2, 2))
        answer[0][1][1] = 0
        answer = answer / 7
        assert np.array_equal(mechanism.cause_repertoire(purview), answer)

    def test_effect_repertoire(self):

        # Test against Matlab default network
        # ===================================

        # Mechanism(['n0'])
        mechanism = models.Mechanism([self.network.nodes[0]],
                                     self.network.state,
                                     self.network.past_state,
                                     self.network)
        # Subsystem(['n0'])
        purview = models.Subsystem([self.network.nodes[0]], self.network)

        effect_repertoire = mechanism.effect_repertoire(purview)
        print(effect_repertoire)
        print(effect_repertoire.shape)
        answer = np.array([0.25, 0.75].reshape(2, 1, 1))
        assert np.array_equal(effect_repertoire, answer)


class TestSubsystem(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.state = np.array([0, 1, 0])
        self.past_state = np.array([1, 1, 0])
        print((2 ** self.size) * self.size)
        self.tpm = np.zeros([2] * self.size + [self.size])
        self.powerset = utils.powerset(np.arange(3))
        self.network = models.Network(self.tpm, self.state, self.past_state)
        self.nodes = [models.Node(self.network, node_index)
                      for node_index in range(self.network.size)]

    def tearDown(self):
        pass

    def test_purview_max_entropy_distribution(self):
        purview = models.Subsystem(self.network.nodes[0:2], self.network)
        max_ent = purview.max_entropy_distribution()
        assert max_ent.shape == (2, 2, 1)
        assert np.array_equal(
            max_ent,
            np.divide(np.ones(4), 4).reshape((2, 2, 1)))
        assert max_ent[0][1][0] == 0.25


# Utilities for testing
#==============================================================================


def _generator_equal(generator_1, generator_2):
    """Quickly compare two generators for equality"""
    sentinel = object()
    return all(a == b for a, b in zip_longest(generator_1,
                                              generator_2,
                                              fillvalue=sentinel))


def pprint(a_list):
    """Prints the ``__str__`` output for objects in a list"""
    print(list(map(str, a_list)))
