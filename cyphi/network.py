#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network
~~~~~~~

Represents the network of interest. This is the primary object of CyPhi and the
context of all |phi| and |big_phi| computation.
"""

import numpy as np
from .node import Node
from .subsystem import Subsystem
from . import validate
from . import utils


class Network:

    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.
    """

    # TODO implement network definition via connectivity_matrix
    def __init__(self, tpm, current_state, past_state,
                 connectivity_matrix=None):
        """
        :param tpm: The network's transition probability matrix **in
            state-by-node form**, so that ``tpm[0][1][0]`` gives the
            probabilities of each node being on if the past state is |0,1,0|.
            The shape of this TPM should thus be
            ``(number of states for each node) + (number of nodes)``.
        :type tpm: ``np.ndarray``
        :param state: An array describing the network's current state;
            ``state[i]`` gives the state of ``self.nodes[i]``
        :type state: ``np.ndarray``
        :param past_state: An array describing the network's past state;
            ``state[i]`` gives the past state of ``self.nodes[i]``
        :type state: ``np.ndarray``
        """
        # TODO make tpm also optional when implementing logical network
        # definition
        self.tpm = tpm
        # TODO! test connectivity matrix
        self.connectivity_matrix = connectivity_matrix
        self.current_state = current_state
        self.past_state = past_state
        # Make these properties immutable (for hashing)
        self.tpm.flags.writeable = False
        self.current_state.flags.writeable = False
        self.past_state.flags.writeable = False
        if self.connectivity_matrix != None:
            self.connectivity_matrix.flags.writeable = False
        # The number of nodes in the Network (TPM is in state-by-node form, so
        # number of nodes is given by the size of the last dimension)
        self.size = tpm.shape[-1]

        # Validate this network
        validate.network(self)

        # Generate the nodes
        self.nodes = [Node(self, node_index)
                      for node_index in range(self.size)]
        # TODO extend to nonbinary nodes
        self.num_states = 2 ** self.size

    def __repr__(self):
        return ("Network(" + ", ".join([repr(self.tpm),
                                          repr(self.current_state),
                                          repr(self.past_state)]) +
                ", connectivity_matrix=" + repr(self.connectivity_matrix) + ")")

    def __str__(self):
        return ("Network(" + str(self.tpm) + ", connectivity_matrix=" +
                str(self.connectivity_matrix) + ")")

    def __eq__(self, other):
        """Two networks are equal if they have the same TPM, current state, and
        past state."""
        return (np.array_equal(self.tpm, other.tpm) and
                np.array_equal(self.current_state, other.current_state) and
                np.array_equal(self.past_state, other.past_state) and
                np.array_equal(self.connectivity_matrix, other.connectivity_matrix))

    def __ne__(self, other):
        return not self.__eq__(other)

    # TODO don't use tostring(not unique for large arrays)
    def __hash__(self):
        return hash((self.tpm.tostring(),
                     self.current_state.tostring(),
                     self.past_state.tostring(),
                     (self.connectivity_matrix.tostring() if
                      self.connectivity_matrix != None else None)))

    def subsystems(self):
        """Return a generator of all possible subsystems of this network."""
        for subset in utils.powerset(self.nodes):
            yield Subsystem(subset, self.current_state, self.past_state, self)
