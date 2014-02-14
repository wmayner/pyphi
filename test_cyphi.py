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
        self.state = np.array([0., 1., 0.])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.zeros([2] * self.size + [self.size]).astype(float)
        self.powerset = utils.powerset(np.arange(3))
        self.network = models.Network(self.tpm, self.state, self.past_state)
        self.nodes = [models.Node(self.network, node_index)
                      for node_index in range(self.network.size)]

    def tearDown(self):
        pass

    def test_network_tpm_validation(self):
        with self.assertRaises(ValidationException):
            tpm = np.arange(3).astype(float)
            state = np.array([0, 1, 0])
            past_state = np.array([1, 1, 0])
            models.Network(tpm, state, past_state)
        with self.assertRaises(ValidationException):
            tpm = np.array([[1, 2, 3]]).astype(float)
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
                             [1, 1, 0]]).reshape([2] * 3 + [3]).astype(float)
        self.network = models.Network(self.tpm, self.state, self.past_state)

    def test_node_init(self):
        tpm_on = self.network.nodes[2].tpm[:, :, :, 1]

        answer_on = self.network.tpm[:, :, :, 2]
        answer_off = 1 - self.network.tpm[:, :, :, 2]

        print("\nTPM for node on:")
        print(tpm_on)
        print("\nAnswer for node on:")
        print(answer_on)
        print("\nTPM for node off:")
        print(tpm_off)
        print("\nAnswer for node off:")
        print(answer_off)

        assert np.array_equal(tpm_on, answer_on)
        assert np.array_equal(tpm_off, answer_off)

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

        ########################
        # Matlab default network
        #######################################################################

        # TODO: make these into dictionaries?

        self.m_state = np.array([1, 0, 0])
        self.m_past_state = np.array([1, 1, 0])
        self.m_tpm = np.array([[0, 0, 0],
                               [0, 0, 1],
                               [1, 0, 1],
                               [1, 0, 0],
                               [1, 1, 0],
                               [1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 0]]).reshape([2] * 3 + [3]).astype(float)
        self.m_network = models.Network(self.m_tpm, self.m_state,
                                        self.m_past_state)

        # Mechanism(['n0'])
        self.m_mechanism_nZero = models.Mechanism([self.m_network.nodes[0]],
                                                  self.m_network.state,
                                                  self.m_network.past_state,
                                                  self.m_network)
        # Subsystem(['n0'])
        self.m_purview_nZero = models.Subsystem([self.m_network.nodes[0]],
                                                self.m_network)

        # Mechanism(['n0', 'n1'])
        self.m_mechanism_nZeroOne = models.Mechanism(self.m_network.nodes[0:2],
                                                     self.m_network.state,
                                                     self.m_network.past_state,
                                                     self.m_network)
        # Subsystem(['n0', 'n2'])
        self.m_purview_nZeroTwo = models.Subsystem(self.m_network.nodes[0:3:2],
                                                   self.m_network)

        #######################################################################

        ########################
        # Simple 'AND' network #
        #######################################################################
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
            ]).reshape([2] * 3 + [3]).astype(float)
        self.s_network = models.Network(self.s_tpm,
                                        self.s_state,
                                        self.s_past_state)

        # Name meaningful states
        self.a_just_turned_on = np.array([1, 0, 0])
        self.a_about_to_be_on = np.array([0, 1, 1])
        self.all_off = np.array([0, 0, 0])

        # Subsystem(['n0', 'n2', 'n3'])
        self.s_purview_all = models.Subsystem(self.s_network.nodes, self.s_network)


        #######################################################################

    def tearDown(self):
        pass

    def test_empty_init(self):
        # Empty mechanism
        mechanism = models.Mechanism([],
                                     self.m_network.state,
                                     self.m_network.past_state,
                                     self.m_network)
        assert mechanism.nodes == []

    def test_marginalize_out(self):
        marginalized_distribution = utils.marginalize_out(
            self.m_network.nodes[0], self.m_network.tpm)
        assert np.array_equal(marginalized_distribution,
                              np.array([[[[0.5,  0.5,  0.0],
                                          [0.5,  0.5,  1.0]],

                                         [[1.0,  0.5,  1.0],
                                          [1.0,  0.5,  0.0]]]]))

    def test_cjd(self):
        # Test against Matlab default network with Mechanism(['n0'])
        cjd = self.m_mechanism_nZero.cjd(self.m_mechanism_nZero.past_state,
                                         self.m_purview_nZero.nodes,
                                         self.m_mechanism_nZero.state,
                                         self.m_mechanism_nZero.nodes)
        answer = utils.uniform_distribution(self.m_network.size)

        print("\nCause repertoire:")
        print(cjd),
        print("\nAnswer:")
        print(answer)

        assert np.array_equal(cjd, answer)

    def test_cause_repertoire_matlab_default_mech_and_purview_same(self):
        """
        Test against Matlab default network with same mechanism and purview
        """
        cause_repertoire = \
            self.m_mechanism_nZero.cause_repertoire(self.m_purview_nZero)
        answer = utils.uniform_distribution(self.m_network.size)

        print(self.m_mechanism_nZero)
        print("\nCause repertoire:")
        print(cause_repertoire)
        print("\nAnswer:")
        print(answer)

        assert np.array_equal(cause_repertoire, answer)

    def test_cause_repertoire_matlab_default_mech_and_purview_different(self):
        """
        Test against Matlab default network with different mechanism and
        purview
        """
        cause_repertoire = \
            self.m_mechanism_nZeroOne.cause_repertoire(self.m_purview_nZeroTwo)
        answer = np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2)

        print("\nCause repertoire:")
        print(cause_repertoire)
        print("\nAnswer:")
        print(answer)

        assert np.array_equal(cause_repertoire, answer)

    def test_cause_repertoire_simple_AND_mech_and_purview_same(self):
        """
        Test against simple 'AND' network with same mechanism and purview
        """
        # Mechanism(['n0']), 'A' just turned on
        mechanism = models.Mechanism([self.s_network.nodes[0]],
                                     self.a_just_turned_on,
                                     self.a_about_to_be_on,
                                     self.s_network)

        cause_repertoire = mechanism.cause_repertoire(self.s_purview_all)
        # Cause repertoire is maximally selective; the past state must have
        # been {0,1,1}, so `answer[(0,1,1)]` should be 1 and everything else
        # should be 0
        answer = np.array([[[0., 0.],
                            [0., 1.]],
                           [[0., 0.],
                            [0., 0.]]])

        print("\nCause repertoire:")
        print(cause_repertoire)
        print("\nAnswer:")
        print(answer)

        assert np.array_equal(cause_repertoire, answer)

    def test_cause_repertoire_simple_AND_mech_and_purview_different(self):
        """
        Test against simple 'AND' network with different mechanism and
        purview
        """
        # Mechanism(['n0']), all nodes off
        mechanism = models.Mechanism([self.s_network.nodes[0]],
                                     self.all_off,
                                     self.all_off,
                                     self.s_network)

        cause_repertoire = mechanism.cause_repertoire(self.s_purview_all)
        # Cause repertoire is minimally selective; only {0,1,1} is ruled out,
        # so probability density should be uniformly distributed among all
        # states not including {0,1,1}
        answer = np.ones((2, 2, 2))
        answer[0][1][1] = 0
        answer = answer / 7

        print("\nCause repertoire:")
        print(cause_repertoire)
        print("\nAnswer:")
        print(answer)

        assert np.array_equal(cause_repertoire, answer)

    def test_effect_repertoire_matlab_same_mech_and_purview(self):
        """
        Test against Matlab default network with same mechanism and purview
        """

        mechanism = self.m_mechanism_nZero
        print("Mechanism:")
        print(mechanism)
        print("Purview:")
        print(self.m_purview_nZero)

        effect_repertoire = \
            self.m_mechanism_nZero.effect_repertoire(self.m_purview_nZero)
        answer = np.array([0.25, 0.75]).reshape(2, 1, 1)

        print("\nEffect repertoire:")
        print(effect_repertoire)
        print("\nAnswer:")
        print(answer)

        assert np.array_equal(effect_repertoire, answer)

    def test_effect_repertoire_matlab_diff_mech_and_purview(self):
        """
        Test against Matlab default network with different mechanism and
        purview
        """

        mechanism = self.m_mechanism_nZeroOne
        print("Mechanism:")
        print(mechanism)
        print("Purview:")
        print(self.m_purview_nZeroTwo)

        effect_repertoire = \
            self.m_mechanism_nZeroOne.effect_repertoire(self.m_purview_nZeroTwo)
        answer = np.array([0, 0, 0.5, 0.5]).reshape(2, 1, 2)

        print("\nEffect repertoire:")
        print(effect_repertoire)
        print("\nAnswer:")
        print(answer)

        assert np.array_equal(effect_repertoire, answer)


class TestSubsystem(unittest.TestCase):

    def setUp(self):
        self.size = 3
        self.state = np.array([0, 1, 0])
        self.past_state = np.array([1, 1, 0])
        print((2 ** self.size) * self.size)
        self.tpm = np.zeros([2] * self.size + [self.size]).astype(float)
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
