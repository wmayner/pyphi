#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from cyphi.network import Network
from cyphi.subsystem import Subsystem


class WithExampleNetworks(unittest.TestCase):

    """This class provides example objects to test against. Any test case that
    needs access to these examples should extend this class."""

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
        self.m_network = Network(self.m_tpm,
                                 self.m_current_state,
                                 self.m_past_state)

        # Subsystem(['n0'])
        self.m_subsys_nZero = Subsystem([self.m_network.nodes[0]],
                                        self.m_current_state,
                                        self.m_past_state,
                                        self.m_network)
        # Mechanism {n0}
        self.m_mechanism_nZero = [self.m_network.nodes[0]]
        # Purview {n0}
        self.m_purview_nZero = [self.m_network.nodes[0]]
        # Subsystem(['n0', 'n1', 'n3'])
        self.m_subsys_all = Subsystem(self.m_network.nodes,
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
        self.s_network = Network(self.s_tpm,
                                 self.s_state,
                                 self.s_past_state)

        # Subsystem(['n0', 'n2', 'n3']), 'A' just turned on
        self.s_subsys_all_a_just_on = Subsystem(self.s_network.nodes,
                                                self.s_state,
                                                self.s_past_state,
                                                self.s_network)

        # Subsystem(['n0', 'n2', 'n3']), All nodes are off
        self.s_subsys_all_off = Subsystem(self.s_network.nodes,
                                          self.all_off,
                                          self.all_off,
                                          self.s_network)

    def tearDown(self):
        pass
