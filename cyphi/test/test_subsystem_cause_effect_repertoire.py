#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pprint import pprint
import numpy as np
from cyphi.utils import print_repertoire, print_repertoire_horiz
from .example_networks import WithExampleNetworks
from cyphi.subsystem import Subsystem, a_cut, a_mip, a_part


# TODO test against other matlab examples

class TestCauseEffectRepertoires(WithExampleNetworks):

    # Test helper function {{{

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

        if result is 1 or t['answer'] is 1:
            print("Result:", result, "Answer:", t['answer'], "", sep="\n\n")
        else:
            print("Result:", result, "Shape:", result.shape, "Answer:",
                  t['answer'], "Shape:", t['answer'].shape, "", sep="\n\n")

        return np.array_equal(result, t['answer'])

    # }}}
    # Cause repertoire tests {{{
    # ==========================
        # Matlab default network {{{
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Full network, no cut {{{
            # ------------------------

    def test_cause_rep_matlab_0(self):
        test_params = {
            'mechanism': [self.m0],
            'purview': [self.m0],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_matlab_1(self):
        test_params = {
            'mechanism': [self.m0, self.m1],
            'purview': [self.m0, self.m2],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2,
                                                             order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_matlab_2(self):
        test_params = {
            'mechanism': [self.m1],
            'purview': [self.m2],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([1.0, 0.0]).reshape(1, 1, 2, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_matlab_3(self):
        test_params = {
            'mechanism': [],
            'purview': [self.m2],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_matlab_4(self):
        test_params = {
            'mechanism': [self.m1],
            'purview': [],
            'candidate_system': self.m_subsys_all,
            'answer': 1
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

            # }}}
            # Full network, with cut {{{
            # --------------------------

    def test_cause_rep_matlab_5(self):
        subsystem = self.m_subsys_all
        subsystem.cut(self.m2, (self.m0, self.m1))
        test_params = {
            'mechanism': [self.m0],
            'purview': [self.m1],
            'candidate_system': subsystem,
            'answer': np.array([1/3, 2/3]).reshape(1, 2, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

            # }}}
            # Subset, with cut {{{
            # --------------------

    def test_cause_rep_matlab_6(self):
        subsystem = self.m_subsys_n1n2
        subsystem.cut(self.m1, self.m2)
        test_params = {
            'mechanism': [self.m2],
            'purview': [self.m1, self.m2],
            'candidate_system': subsystem,
            'answer': np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2,
                                                                 order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_matlab_6(self):
        subsystem = self.m_subsys_n1n2
        subsystem.cut(self.m1, self.m2)
        test_params = {
            'mechanism': [self.m2],
            'purview': [self.m2],
            'candidate_system': subsystem,
            'answer': np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_matlab_8(self):
        subsystem = self.m_subsys_n0n2
        subsystem.cut(self.m0,  self.m2)
        test_params = {
            'mechanism': [self.m2],
            'purview': [self.m0],
            'candidate_system': subsystem,
            'answer': np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)


            # }}}
        # }}}
        # Simple 'AND' network {{{
        # ~~~~~~~~~~~~~~~~~~~~~~~~
            # State: 'A' just turned on {{{
            # -----------------------------

    def test_cause_rep_simple_0(self):
        test_params = {
            'mechanism': [self.s0],
            'purview': [self.s0],
            'candidate_system': self.s_subsys_all_a_just_on,
            # Cause repertoire is maximally selective; the past state must have
            # been {0,1,1}, so `answer[(0,1,1)]` should be 1 and everything
            # else should be 0
            'answer': np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_1(self):
        test_params = {
            'mechanism': [],
            'purview': [self.s0],
            'candidate_system': self.s_subsys_all_a_just_on,
            # No matter the state of the purview (m0), the probability it will
            # be on in the next timestep is 1/8
            'answer': np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_2(self):
        test_params = {
            'mechanism': [self.s1],
            'purview': [self.s0, self.s1, self.s2],
            'candidate_system': self.s_subsys_all_a_just_on,
            'answer': np.ones((2, 2, 2)) / 8
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_3(self):
        test_params = {
            'mechanism': [self.s0, self.s1],
            'purview': [self.s0, self.s2],
            'candidate_system': self.s_subsys_all_a_just_on,
            'answer': np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2,
                                                             order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

            # }}}
            # State: all nodes off {{{
            # ------------------------

    def test_cause_rep_simple_4(self):
        test_params = {
            'mechanism': [self.s0],
            'purview': [self.s0],
            'candidate_system': self.s_subsys_all_off,
            'answer': np.array([(3 / 7), (4 / 7)]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

    def test_cause_rep_simple_5(self):
        # Cause repertoire is minimally selective; only {0,1,1} is ruled out,
        # so probability density should be uniformly distributed among all
        # states not including {0,1,1} when purview is whole network
        answer = np.ones((2, 2, 2))
        answer[0][1][1] = 0.0
        answer = answer / 7
        test_params = {
            'mechanism': [self.s0],
            'purview': [self.s0, self.s1, self.s2],
            'candidate_system': self.s_subsys_all_off,
            'answer': answer
        }
        assert self.cause_or_effect_repertoire_is_correct('cause_repertoire',
                                                          test_params)

            # }}}
        # }}}
    # }}}
    # Effect repertoire tests {{{
    # =========================================================================
        # Matlab default network {{{
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Full network, no cut {{{
            # ------------------------

    def test_effect_rep_matlab_0(self):
        test_params = {
            'mechanism': [self.m0],
            'purview': [self.m0],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_matlab_1(self):
        test_params = {
            'mechanism': [self.m0, self.m1],
            'purview': [self.m0, self.m2],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.0, 0.0, 0.5, 0.5]).reshape(2, 1, 2,
                                                             order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_matlab_2(self):
        test_params = {
            'mechanism': [self.m1],
            'purview': [self.m2],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_matlab_3(self):
        test_params = {
            'mechanism': [],
            'purview': [self.m1],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)
    def test_effect_rep_matlab_4(self):
        test_params = {
            'mechanism': [self.m2],
            'purview': [],
            'candidate_system': self.m_subsys_all,
            'answer': 1
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)
    def test_effect_rep_matlab_5(self):
        test_params = {
            'mechanism': [],
            'purview': [self.m0],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)
    def test_effect_rep_matlab_6(self):
        test_params = {
            'mechanism': [self.m0],
            'purview': [self.m2],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)
    def test_effect_rep_matlab_7(self):
        test_params = {
            'mechanism': [self.m1, self.m2],
            'purview': [self.m0],
            'candidate_system': self.m_subsys_all,
            'answer': np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_matlab_8(self):
        test_params = {
            'mechanism': [self.m1],
            'purview': [],
            'candidate_system': self.m_subsys_all,
            'answer': 1
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

            # }}}
            # Full network, with cut {{{
            # --------------------------

    def test_effect_rep_matlab_full_network_with_cut_1(self):
        subsystem = self.m_subsys_all
        subsystem.cut((self.m0, self.m2), self.m1)
        test_params = {
            'mechanism': [self.m0],
            'purview': [self.m2],
            'candidate_system': subsystem,
            'answer': np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_matlab_full_network_with_cut_1(self):
        subsystem = self.m_subsys_all
        subsystem.cut((self.m0, self.m2), self.m1)
        test_params = {
            'mechanism': [self.m0, self.m1, self.m2],
            'purview': [self.m0, self.m2],
            'candidate_system': subsystem,
            'answer': np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

            # }}}
            # Subset, with cut {{{
            # --------------------
            # TODO remove line numbers of matlab output eventually

        def test_effect_rep_matlab_subset_with_cut_1(self):
            # 18788
            subsystem = self.m_subsys_n1n2
            subsystem.cut(self.m1, self.m2)
            test_params = {
                'mechanism': [self.m1],
                'purview': [self.m1, self.m2],
                'candidate_system': subsystem,
                'answer': np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2,
                                                                    order="F")
            }
            assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                              test_params)

        def test_effect_rep_matlab_2(self):
            # 18724
            subsystem = self.m_subsys_n1n2
            subsystem.cut(self.m1, self.m2)
            test_params = {
                'mechanism': [],
                'purview': [self.m1],
                'candidate_system': subsystem,
                'answer': np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
            }
            assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                              test_params)

        def test_effect_rep_matlab_3(self):
            # 18740
            subsystem = self.m_subsys_n1n2
            subsystem.cut(self.m1, self.m2)
            test_params = {
                'mechanism': [self.m1],
                'purview': [self.m2],
                'candidate_system': subsystem,
                'answer': np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
            }
            assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                              test_params)


            # TODO
            # Future: m1 / m2
            # Future: m0 / m2
            # Future: (m0, m2) / (m0, m1, m2)
            # Future: m0 / m1

            # }}}
        # }}}
        # Simple 'AND' network {{{
        # ~~~~~~~~~~~~~~~~~~~~~~~~
            # State: 'A' just turned on {{{
            # -----------------------------

    def test_effect_rep_simple_0(self):
        # 'A' just turned on
        test_params = {
            'mechanism': [self.s0],
            'purview': [self.s0],
            'candidate_system': self.s_subsys_all_a_just_on,
            'answer': np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_1(self):
        # 'A' just turned on
        test_params = {
            'mechanism': [],
            'purview': [self.s0],
            'candidate_system': self.s_subsys_all_a_just_on,
            # No matter the state of the purview {m0}, the probability it will
            # be on in the next timestep is 1/8
            'answer': np.array([0.875, 0.125]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_2(self):
        # 'A' just turned on
        answer = np.zeros((2, 2, 2))
        answer[(0, 0, 0)] = 1
        test_params = {
            'mechanism': [self.s1],
            'purview': [self.s0, self.s1, self.s2],
            'candidate_system': self.s_subsys_all_a_just_on,
            'answer': answer
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_3(self):
        # 'A' just turned on
        test_params = {
            'mechanism': [self.s1],
            'purview': [self.s0, self.s2],
            'candidate_system': self.s_subsys_all_a_just_on,
            'answer': np.array([1.0, 0.0, 0.0, 0.0]).reshape(2, 1, 2,
                                                            order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                            test_params)

            # }}}
            # State: all nodes off {{{
            # ------------------------

    def test_effect_rep_simple_4(self):
        test_params = {
            'mechanism': [self.s0],
            'purview': [self.s0],
            'candidate_system': self.s_subsys_all_off,
            'answer': np.array([0.75, 0.25]).reshape(2, 1, 1, order="F")
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

    def test_effect_rep_simple_5(self):
        answer = np.zeros((2, 2, 2))
        answer[(0, 0, 0)] = 0.75
        answer[(1, 0, 0)] = 0.25
        test_params = {
            'mechanism': [self.s0],
            'purview': [self.s0, self.s1, self.s2],
            'candidate_system': self.s_subsys_all_off,
            'answer': answer
        }
        assert self.cause_or_effect_repertoire_is_correct('effect_repertoire',
                                                          test_params)

            # }}}
        # }}}
    # }}}

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
