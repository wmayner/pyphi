#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .example_networks import WithExampleNetworks
from cyphi.subsystem import Subsystem


class TestSubsystem(WithExampleNetworks):

    def test_empty_init(self):
        # Empty mechanism
        subsys = Subsystem([],
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
