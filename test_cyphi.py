#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for CyPhi."""

from itertools import zip_longest

import numpy as np
import cyphi.models as models
import cyphi.utils as utils
from cyphi.exceptions import ValidationException

import unittest

# TODO look into using pytest fixtures vs. setUp/tearDown

# NOTE: ``py.test`` and ``nose`` only display ``print`` statements in a test if
# it failed.


# Class for common setUp and tearDown methods
class TestWithExampleNetworks(unittest.TestCase):

    def setUp(self):

        ########################
        # Matlab default network
        #######################################################################

        # TODO: make these into dictionaries/named tuples?

        self.m_current_state = np.array([1, 0, 0])
        self.m_past_state = np.array([1, 1, 0])
        self.m_tpm = np.array([[0, 0, 0],
                               [0, 0, 1],
                               [1, 0, 1],
                               [1, 0, 0],
                               [1, 1, 0],
                               [1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 0]]).reshape([2] * 3 + [3],
                                                   order="F").astype(float)
        self.m_network = models.Network(self.m_tpm,
                                        self.m_current_state,
                                        self.m_past_state)

        # Subsystem(['n0'])
        self.m_subsys_nZero = models.Subsystem([self.m_network.nodes[0]],
                                               self.m_current_state,
                                               self.m_past_state,
                                               self.m_network)
        # Mechanism {n0}
        self.m_mechanism_nZero = [self.m_network.nodes[0]]
        # Purview {n0}
        self.m_purview_nZero = [self.m_network.nodes[0]]
        # Subsystem(['n0', 'n1', 'n3'])
        self.m_subsys_all = models.Subsystem(self.m_network.nodes,
                                             self.m_current_state,
                                             self.m_past_state,
                                             self.m_network)
        # Mechanism {n0, n1}
        self.m_mechanism_nZeroOne = self.m_network.nodes[0:2]
        # Purview {n0, n1}
        self.m_purview_nZeroTwo = self.m_network.nodes[0:3:2]

        ########################
        # Simple 'AND' network #
        #######################################################################
        # Diagram:
        #
        #       +---+
        #   +-->| A |<--+
        #   |   +---+   |
        #   |    AND    |
        # +-+-+       +-+-+
        # | B |       | C |
        # +---+       +---+
        #
        # TPM:
        #
        #   Past state --> Current state
        # --------------+---------------
        #    A, B, C    |    A, B, C
        # --------------+---------------
        #   {0, 0, 0}   |   {0, 0, 0}
        #   {0, 0, 1}   |   {0, 0, 0}
        #   {0, 1, 0}   |   {0, 0, 0}
        #   {0, 1, 1}   |   {1, 0, 0}
        #   {1, 0, 0}   |   {0, 0, 0}
        #   {1, 0, 1}   |   {0, 0, 0}
        #   {1, 1, 0}   |   {0, 0, 0}
        #   {1, 1, 1}   |   {0, 0, 0}

        # Name meaningful states
        self.a_just_turned_on = np.array([1, 0, 0])
        self.a_about_to_be_on = np.array([0, 1, 1])
        self.all_off = np.array([0, 0, 0])

        self.s_state = self.a_just_turned_on
        self.s_past_state = self.a_about_to_be_on
        self.s_tpm = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0],
                               [1, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]]).reshape([2] * 3 + [3]).astype(float)
        self.s_network = models.Network(self.s_tpm,
                                        self.s_state,
                                        self.s_past_state)

        # Subsystem(['n0', 'n2', 'n3']), 'A' just turned on
        self.s_subsys_all_a_just_on = models.Subsystem(self.s_network.nodes,
                                                       self.s_state,
                                                       self.s_past_state,
                                                       self.s_network)

        # Subsystem(['n0', 'n2', 'n3']), All nodes off
        self.s_subsys_all_off = models.Subsystem(self.s_network.nodes,
                                                 self.all_off,
                                                 self.all_off,
                                                 self.s_network)

    def tearDown(self):
        pass


class TestUtils(TestWithExampleNetworks):

    def test_marginalize_out(self):
        marginalized_distribution = utils.marginalize_out(
            self.m_network.nodes[0], self.m_network.tpm)
        assert np.array_equal(marginalized_distribution,
                              np.array([[[[0.,  0.,  0.5],
                                          [1.,  1.,  0.5]],

                                         [[1.,  0.,  0.5],
                                          [1.,  1.,  0.5]]]]))

    def test_purview_max_entropy_distribution(self):
        # Individual setUp
        size = 3
        state = np.array([0, 1, 0])
        past_state = np.array([1, 1, 0])
        tpm = np.zeros([2] * size + [size]).astype(float)
        network = models.Network(tpm, state, past_state)

        max_ent = utils.max_entropy_distribution(network.nodes[0:2],
                                                 network)
        assert max_ent.shape == (2, 2, 1)
        assert np.array_equal(
            max_ent,
            np.divide(np.ones(4), 4).reshape((2, 2, 1)))
        assert max_ent[0][1][0] == 0.25

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
        self.current_state = np.array([0., 1., 0.])
        self.past_state = np.array([1, 1, 0])
        self.tpm = np.zeros([2] * self.size + [self.size]).astype(float)
        self.powerset = utils.powerset(np.arange(3))
        self.network = models.Network(self.tpm,
                                      self.current_state,
                                      self.past_state)
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
        self.network = models.Network(self.tpm,
                                      self.current_state,
                                      self.past_state)

    def test_node_init(self):
        tpm = self.network.nodes[2].tpm
        answer = self.network.tpm[:, :, :, 2]

        print("TPM", tpm, "Answer", answer, "", sep="\n\n")

        assert np.array_equal(tpm, answer)

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


class TestSubsystem(TestWithExampleNetworks):

    def test_empty_init(self):
        # Empty mechanism
        subsys = models.Subsystem([],
                                  self.m_network.current_state,
                                  self.m_network.past_state,
                                  self.m_network)
        assert subsys.nodes == []

    # Cause/effect repertoire test helper
    # =========================================================================

    @staticmethod
    def cause_or_effect_repertoire_is_correct(function, t):
        """Test `effect_repertoire` or `cause_repertoire`.

        :param function: The function to test (either "cause_repertoire" or
        "effect_repertoire") :param t: A dictionary containing the parameters
        for the test: candidate_system, mechanism, purview, and answer :type t:
            ``dict``

        """
        result = getattr(t['candidate_system'],
                         function)(t['mechanism'], t['purview'])

        print("Mechanism:", t['mechanism'], "Purview:", t['purview'], "",
              sep="\n\n")

        print("Effect repertoire:", result, "Shape:", result.shape, "Answer:",
              t['answer'], "Shape:", t['answer'].shape, "", sep="\n\n")

        return np.array_equal(result, t['answer'])

    # Cause repertoire tests
    # =========================================================================

    # Matlab default network
    # ----------------------

    def test_cause_rep_matlab_mech_n0_purview_n0(self):
        # Mechanism {n0}
        # Purview {n0}
        test_params = {
            'candidate_system': self.m_subsys_all,
            'mechanism': self.m_mechanism_nZero,
            'purview': self.m_purview_nZero,
            'answer': np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_matlab_mech_n0n1_purview_n0n2(self):
        # Mechanism {n0, n1}
        # Purview {n0, n2}
        test_params = {
            'candidate_system': self.m_subsys_all,
            'mechanism': self.m_mechanism_nZeroOne,
            'purview': self.m_purview_nZeroTwo,
            'answer': np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2,
                                                             order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    # Simple 'AND' network
    # --------------------

    # State:
    # 'A' just turned on

    def test_cause_rep_simple_AND_a_just_on_mech_n0_purview_n0(self):
        # Mechanism {n0}
        # Purview {n0}
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': [self.s_network.nodes[0]],
            'purview': [self.s_network.nodes[0]],
            # Cause repertoire is maximally selective; the past state must have
            # been {0,1,1}, so `answer[(0,1,1)]` should be 1 and everything
            # else should be 0
            'answer': np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_AND_a_just_on_mech_empty_purview_n0(self):
        # Mechanism {}
        # Purview {n0}
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': [],
            'purview': [self.s_network.nodes[0]],
            # No matter the state of the purview (n0), the probability it will
            # be on in the next timestep is 1/8
            'answer': np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_AND_a_just_on_mech_n1_purview_n0n1n2(self):
        # Mechanism {n1}
        # Purview {n0, n1, n2}
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': [self.s_network.nodes[1]],
            'purview': self.s_network.nodes,
            'answer': np.ones((2, 2, 2)) / 8
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_AND_a_just_on_mech_n0n1_purview_n0n2(self):
        # Mechanism {n0, n1}
        # Purview {n0, n2}
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': self.s_network.nodes[0:2],
            'purview': self.s_network.nodes[0:3:2],
            'answer': np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2,
                                                             order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    # State:
    # All nodes off

    def test_cause_rep_simple_AND_all_off_mech_n0_purview_n0(self):
        # Mechanism {n0}
        # Purview {n0}
        test_params = {
            'candidate_system': self.s_subsys_all_off,
            'mechanism': [self.s_network.nodes[0]],
            'purview': [self.s_network.nodes[0]],
            'answer': np.array([(3 / 7), (4 / 7)]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_AND_all_off_mech_n0_purview_n0n1n2(self):
        # Mechanism {n0}
        # Purview {n0, n1, n2}

        # Cause repertoire is minimally selective; only {0,1,1} is ruled out,
        # so probability density should be uniformly distributed among all
        # states not including {0,1,1} when purview is whole network
        answer = np.ones((2, 2, 2))
        answer[0][1][1] = 0.0
        answer = answer / 7
        test_params = {
            'candidate_system': self.s_subsys_all_off,
            'mechanism': [self.s_network.nodes[0]],
            'purview': self.s_network.nodes,
            'answer': answer
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    # Effect repertoire tests
    # =========================================================================

    # Matlab default network
    # ----------------------

    def test_effect_rep_matlab_mech_n0_purview_n0(self):
        # Mechanism {n0}
        # Purview {n0}
        test_params = {
            'candidate_system': self.m_subsys_all,
            'mechanism': self.m_mechanism_nZero,
            'purview': self.m_purview_nZero,
            'answer': np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_matlab_mech_n0n1_purview_n0n2(self):
        # Mechanism {n0, n1}
        # Purview {n0, n2}
        test_params = {
            'candidate_system': self.m_subsys_all,
            'mechanism': self.m_mechanism_nZeroOne,
            'purview': self.m_purview_nZeroTwo,
            'answer': np.array([0.0, 0.0, 0.5, 0.5]).reshape(2, 1, 2,
                                                             order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    # Simple 'AND' network
    # --------------------

    # State:
    # 'A' just turned on

    def test_effect_rep_simple_AND_a_just_on_mech_n0_purview_n0(self):
        # Mechanism {n0}
        # Purview {n0}
        # 'A' just turned on
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': [self.s_network.nodes[0]],
            'purview': [self.s_network.nodes[0]],
            'answer': np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_AND_a_just_on_mech_empty_purview_n0(self):
        # Mechanism {}
        # Purview {n0}
        # 'A' just turned on
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': [],
            'purview': [self.s_network.nodes[0]],
            # No matter the state of the purview (n0), the probability it will
            # be on in the next timestep is 1/8
            'answer': np.array([0.875, 0.125]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_AND_a_just_on_mech_n1_purview_n0n1n2(self):
        # Mechanism {n1}
        # Purview {n0, n1, n2}
        # 'A' just turned on
        answer = np.zeros((2, 2, 2))
        answer[(0, 0, 0)] = 1
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': [self.s_network.nodes[1]],
            'purview': self.s_network.nodes,
            'answer': answer
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_AND_a_just_on_mech_n0n1_purview_n0n2(self):
        # Mechanism {n0, n1}
        # Purview {n0, n2}
        # 'A' just turned on
        test_params = {
            'candidate_system': self.s_subsys_all_a_just_on,
            'mechanism': [self.s_network.nodes[1]],
            'purview': self.s_network.nodes[0:3:2],
            'answer': np.array([1.0, 0.0, 0.0, 0.0]).reshape(2, 1, 2,
                                                             order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    # State:
    # All nodes off

    def test_effect_rep_simple_AND_all_off_mech_n0_purview_n0(self):
        # Mechanism {n0}
        # Purview {n0}
        test_params = {
            'candidate_system': self.s_subsys_all_off,
            'mechanism': [self.s_network.nodes[0]],
            'purview': [self.s_network.nodes[0]],
            'answer': np.array([0.75, 0.25]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_AND_all_off_mech_n0_purview_n0n1n2(self):
        # Mechanism {n0}
        # Purview {n0, n1, n2}
        answer = np.zeros((2, 2, 2))
        answer[(0, 0, 0)] = 0.75
        answer[(1, 0, 0)] = 0.25
        test_params = {
            'candidate_system': self.s_subsys_all_off,
            'mechanism': [self.s_network.nodes[0]],
            'purview': self.s_network.nodes,
            'answer': answer
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

# Utilities for testing
#==============================================================================


def _generator_equal(generator_1, generator_2):
    sentinel = object()
    return all(a == b for a, b in zip_longest(generator_1,
                                              generator_2,
                                              fillvalue=sentinel))


def nprint(a_list):
    """Print the ``__str__`` output for objects in a list."""
    print(list(map(str, a_list)))


#==============================================================================


suite = unittest.TestLoader().loadTestsFromTestCase(TestSubsystem)
unittest.TextTestRunner(verbosity=2).run(suite)
