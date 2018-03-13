#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem_2_0.py

"""Implementation of IIT 2.0"""

from pyphi.distribution import max_entropy_distribution


class Subsystem_2_0:

    def __init__(self, network, state, node_indices):
        self.network = network
        self.state = tuple(state)
        self.node_indices = node_indices

    def __len__(self):
        return len(self.node_indices)

    def prior_repertoire(self):
        """The a priori repertoire of the system."""
        return max_entropy_distribution(self.node_indices, len(self))
