#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pprint import pprint
import numpy as np
import cyphi.utils as utils
from cyphi.utils import print_repertoire, print_repertoire_horiz
from .example_networks import WithExampleNetworks
from cyphi.subsystem import Subsystem, a_cut, a_mip, a_part


# TODO test against other matlab examples
class TestSubsystem(WithExampleNetworks):

    def test_empty_init(self):
        # Empty mechanism
        subsys = Subsystem([],
                           self.m_network.current_state,
                           self.m_network.past_state,
                           self.m_network)
        assert subsys.nodes == ()

    def test_eq(self):
        a = Subsystem([self.m0, self.m1],
                      self.m_network.current_state,
                      self.m_network.past_state,
                      self.m_network)
        b = Subsystem([self.m0, self.m1],
                      self.m_network.current_state,
                      self.m_network.past_state,
                      self.m_network)
        assert a == b

    def test_hash(self):
        subsys = Subsystem([self.m0, self.m1],
                           self.m_network.current_state,
                           self.m_network.past_state,
                           self.m_network)
        h = hash(subsys)

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

        if result is 1:
            print("Result:", result, "Answer:", t['answer'], "", sep="\n\n")
        else:
            print("Result:", result, "Shape:", result.shape, "Answer:",
                  t['answer'], "Shape:", t['answer'].shape, "", sep="\n\n")

        return np.array_equal(result, t['answer'])

    # Cause repertoire tests
    # =========================================================================

    # Matlab default network
    # ----------------------

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

    # Simple 'AND' network
    # --------------------

    # State:
    # 'A' just turned on

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

    # State:
    # All nodes off

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

    # Effect repertoire tests
    # =========================================================================

    # Matlab default network
    # ----------------------

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

        # Future: m1 / m2
        # Future: m0 / m2
        # Future: (m0, m2) / (m0, m1, m2)
        # Future: m0 / m1

    # Simple 'AND' network
    # --------------------

    # State:
    # 'A' just turned on

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

    # State:
    # All nodes off

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

    # Unconstrained cause repertoire tests
    # =========================================================================

    # Matlab default network
    # ----------------------

    def test_unconstrained_cause_repertoire_matlab_0(self):
        # Purview {m0}
        assert np.array_equal(
            self.m_subsys_all.unconstrained_cause_repertoire(
                [self.m0]),
            np.array([[[0.5]], [[0.5]]]))

    def test_unconstrained_cause_repertoire_matlab_1(self):
        purview = [self.m0, self.m1]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_cause_repertoire(purview),
            np.array([[[0.25], [0.25]], [[0.25], [0.25]]]))

    def test_unconstrained_cause_repertoire_matlab_2(self):
        purview = [self.m0, self.m1, self.m2]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_cause_repertoire(purview),
            np.array([[[0.125, 0.125],
                       [0.125, 0.125]],
                      [[0.125, 0.125],
                       [0.125, 0.125]]]))

    # Unconstrained effect repertoire tests
    # =========================================================================

    # Matlab default network
    # ----------------------

    def test_unconstrained_effect_repertoire_matlab_0(self):
        purview = [self.m0]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_effect_repertoire(purview),
            np.array([[[0.25]], [[0.75]]]))

    def test_unconstrained_effect_repertoire_matlab_1(self):
        purview = [self.m0, self.m1]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_effect_repertoire(purview),
            np.array([[[0.125], [0.125]], [[0.375], [0.375]]]))

    def test_unconstrained_effect_repertoire_matlab_2(self):
        purview = [self.m0, self.m1, self.m2]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_effect_repertoire(purview),
            np.array([[[0.0625, 0.0625],
                       [0.0625, 0.0625]],
                      [[0.1875, 0.1875],
                       [0.1875, 0.1875]]]))

    # Cause-effect information tests
    # =========================================================================

    def test_cause_info(self):
        mechanism = [self.m0, self.m1]
        purview = [self.m0, self.m2]
        answer = utils.emd(
            self.m_subsys_all.cause_repertoire(mechanism, purview),
            self.m_subsys_all.unconstrained_cause_repertoire(purview))
        assert self.m_subsys_all.cause_info(mechanism, purview) == answer

    def test_effect_info(self):
        mechanism = [self.m0, self.m1]
        purview = [self.m0, self.m2]
        answer = utils.emd(
            self.m_subsys_all.effect_repertoire(mechanism, purview),
            self.m_subsys_all.unconstrained_effect_repertoire(purview))
        assert self.m_subsys_all.effect_info(mechanism, purview) == answer

    def test_cause_effect_info(self):
        mechanism = [self.m0, self.m1]
        purview = [self.m0, self.m2]
        answer = min(self.m_subsys_all.cause_info(mechanism, purview),
                     self.m_subsys_all.effect_info(mechanism, purview))
        assert (self.m_subsys_all.cause_effect_info(mechanism, purview) ==
                answer)

    # Phi tests
    # =========================================================================

    def test_find_mip_bad_direction(self):
        mechanism = [self.m0]
        purview = [self.m0]
        with self.assertRaises(ValueError):
            self.m_subsys_all.find_mip('doge', mechanism, purview)

    def test_find_mip_reducible(self):
        mechanism = [self.m0]
        purview = [self.m0]
        mip = self.m_subsys_all.find_mip('past', mechanism, purview)
        assert mip is None

    def test_find_mip_irreducible_1(self):
        mechanism = [self.m1]
        purview = [self.m2]
        mip = self.m_subsys_all.find_mip('past', mechanism, purview)

        part0 = a_part(mechanism=(),
                       purview=(self.m2,))
        part1 = a_part(mechanism=(self.m1,),
                       purview=())
        partitioned_repertoire = np.array([0.5, 0.5]).reshape(1, 1, 2)
        phi = 0.5
        assert mip_eq(mip, a_mip(partition=(part0, part1),
                                 repertoire=partitioned_repertoire,
                                 difference=phi))

    def test_find_mip_irreducible_2(self):
        mechanism = [self.m1]
        purview = [self.m2]
        mip = self.m_subsys_all.find_mip('future', mechanism, purview)
        part0 = a_part(mechanism=(),
                       purview=(self.m1,))
        part1 = a_part(mechanism=(self.m2,),
                       purview=())
        partitioned_repertoire = np.array([0.5, 0.5]).reshape(1, 1, 2)
        phi = 0.5
        assert mip_eq(mip, a_mip(partition=(part0, part1),
                                 repertoire=partitioned_repertoire,
                                 difference=phi))

    # def test_find_mip_irreducible3(self):
    #     # Past: m0 / m2
    #     s = self.m_subsys_all
    #     nodes = self.m_network.nodes
    #     mechanism = [nodes[0]]
    #     purview = [nodes[2]]
    #     mip = s.find_mip('past', mechanism, purview)

    #     part0 = a_part(mechanism=nodes[0],
    #                    purview=nodes[0])
    #     part1 = a_part(mechanism=nodes[1],
    #                    purview=nodes[1])

    #     answer = a_mip(partition=(part0, part1),
    #                    repertoire=partitioned_repertoire,
    #                    difference=difference)
    #     assert mip == answer

    # def test_find_mip_irreducible4(self):
    #     # Future: (m0, m1) / (m0, m1, m2)
    #     s = self.m_subsys_all
    #     nodes = self.m_network.nodes
    #     mechanism = nodes[0:2]
    #     purview = nodes
    #     mip = s.find_mip('past', mechanism, purview)

    #     part0 = a_part(mechanism=nodes[0],
    #                    purview=nodes[0])
    #     part1 = a_part(mechanism=nodes[1],
    #                    purview=nodes[1])

    #     answer = a_mip(partition=(part0, part1),
    #                    repertoire=partitioned_repertoire,
    #                    difference=difference)
    #     assert mip == answer

    # def test_find_mip_irreducible5(self):
    #     # Future: m0 / m1
    #     s = self.m_subsys_all
    #     nodes = self.m_network.nodes
    #     mechanism = [nodes[0]]
    #     purview = [nodes[1]]
    #     mip = s.find_mip('past', mechanism, purview)
    #     part0 = a_part(mechanism=nodes[0],
    #                    purview=nodes[0])
    #     part1 = a_part(mechanism=nodes[1],
    #                    purview=nodes[1])
    #     answer = a_mip(partition=(part0, part1),
    #                    repertoire=partitioned_repertoire,
    #                    difference=difference)
    #     assert mip == answer

    # def test_find_mip(self):
    #     s = self.m_subsys_all
    #     mechanism = self.m_network.nodes
    #     purview = self.m_network.nodes[0:3:2]
    #     mip = s.find_mip('past', mechanism, purview)
    #     assert

    #     print(mip.partition)
    #     utils.print_partition(mip.partition)
    #     print(mip.difference, '\n')
    #     print(mip.repertoire)
    #     print_repertoire_horiz(mip.repertoire)

    #     assert 1

    def test_find_mip_full_mech_and_purview(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        mip = s.find_mip('past', mechanism, purview)

        print(mip.partition)
        utils.print_partition(mip.partition)
        print(mip.difference, '\n')
        print(mip.repertoire)
        print_repertoire_horiz(mip.repertoire)
        assert 1

    def test_mip_past(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        mip_past = s.find_mip('past', mechanism, purview)
        assert tuple_eq(mip_past, s.mip_past(mechanism, purview))

    def test_mip_future(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        mip_future = s.find_mip('future', mechanism, purview)
        assert tuple_eq(mip_future, s.mip_future(mechanism, purview))

    def test_phi_mip_past(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        assert (s.phi_mip_past(mechanism, purview) ==
                s.mip_past(mechanism, purview).difference)

    def test_phi_mip_future(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m2]
        assert (s.phi_mip_future(mechanism, purview) ==
                s.mip_future(mechanism, purview).difference)

    # TODO finish this test
    def test_phi(self):
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        print("\n*** Past ***\n")
        mip = self.m_subsys_all.mip_past(mechanism, purview)
        unpartitioned_repertoire = \
            self.m_subsys_all.cause_repertoire(mechanism, purview)
        print("partition:")
        pprint(mip.partition)
        print("phi:", mip.difference)
        print("whole repertoire:")
        print_repertoire_horiz(unpartitioned_repertoire)
        print("mip repertoire:")
        print_repertoire_horiz(mip.repertoire)

        print("\n*** Future ***\n")
        mip = self.m_subsys_all.mip_future(mechanism, purview)
        unpartitioned_repertoire = \
            self.m_subsys_all.effect_repertoire(mechanism, purview)
        print("partition:")
        pprint(mip.partition)
        print("phi:", mip.difference)
        print("whole repertoire:")
        print_repertoire_horiz(unpartitioned_repertoire)
        print("mip repertoire:")
        print_repertoire_horiz(mip.repertoire)

        assert 1

    # Phi_max tests
    # =========================================================================

    def test_find_mice_bad_direction(self):
        mechanism = [self.m0]
        with self.assertRaises(ValueError):
            self.m_subsys_all.find_mice('doge', mechanism)

    # TODO finish
    def test_find_mice(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        past_mice = s.find_mice('past', mechanism)
        future_mice = s.find_mice('future', mechanism)
        assert 1

    def test_core_cause(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        assert s.core_cause(mechanism) == s.find_mice('past', mechanism)

    def test_core_effect(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        assert s.core_effect(mechanism) == s.find_mice('future', mechanism)

    def test_phi_max(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        assert 1
        # assert 0.5 == round(s.phi_max(mechanism), 4)


# Helper for checking tuple equality when values can be numpy arrays
# =========================================================================

def tuple_eq(a, b):
    """Return whether two tuples are equal, using ``np.array_equal`` for
    numpy arrays.

    If values are numpy arrays, ``np.array_equal`` is used for checking
    equality.
    """
    if len(a) != len(b):
        return False
    result = True
    for i in range(len(a)):
        if isinstance(a[i], type(())) and isinstance(b[i], type(())):
            if not tuple_eq(a[i], b[i]):
                return False
        if isinstance(a[i], np.ndarray) and isinstance(a[i], np.ndarray):
            if not np.array_equal(a[i], b[i]):
                return False
        elif not a[i] == b[i]:
            return False
    return result


def mip_eq(a, b):
    """Return whether two MIPs are equal."""
    return ((a.partition == b.partition or a.partition == (b.partition[1],
                                                           b.partition[0])) and
            (a.difference == b.difference) and
            (np.array_equal(a.repertoire, b.repertoire)))
