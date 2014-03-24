#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from . import utils
from .exceptions import ValidationException
from .node import Node


class Network:

    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.
    """

    # TODO implement network definition via connectivity_matrix
    def __init__(self, tpm, current_state, past_state):
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
        # Validate the TPM
        if (tpm.shape[-1] is not len(tpm.shape) - 1):
            raise ValidationException(
                """Invalid TPM: There must be a dimension for each node, each
                one being the size of the corresponding node's state space,
                plus one dimension that is the same size as the network.""")
        # TODO extend to nonbinary nodes
        if (tpm.shape[:-1] != tuple([2] * tpm.shape[-1])):
            raise ValidationException(
                """Invalid TPM: We can only handle binary nodes at the moment.
                Each dimension except the last must be of size 2.""")

        self.tpm = tpm
        self.current_state = current_state
        self.past_state = past_state
        # Make these properties immutable (for hashing)
        self.tpm.flags.writeable = False
        self.current_state.flags.writeable = False
        self.past_state.flags.writeable = False
        # The number of nodes in the Network (TPM is in state-by-node form, so
        # number of nodes is given by the size of the last dimension)
        self.size = tpm.shape[-1]

        # Validate the state
        if (current_state.size is not self.size):
            raise ValidationException(
                "Invalid state: there must be one entry per node in the " +
                "network; this state has " + str(current_state.size) +
                "entries, " + "but there are " + str(self.size) + " nodes.")

        # Generate the nodes
        self.nodes = [Node(self, node_index)
                      for node_index in range(self.size)]
        # TODO extend to nonbinary nodes
        self.num_states = 2 ** self.size
        self.uniform_distribution = utils.uniform_distribution(self.size)

    def __repr__(self):
        return "Network(" + repr(self.tpm) + ")"

    def __str__(self):
        return "Network(" + str(self.tpm) + ")"

    def __eq__(self, other):
        """Two networks are equal if they have the same TPM, current state, and
        past state."""
        return (np.array_equal(self.tpm, other.tpm) and
                np.array_equal(self.current_state, other.current_state) and
                np.array_equal(self.past_state, other.past_state))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.tpm.tostring(), self.current_state.tostring(),
                    self.past_state.tostring()))
