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
from . import validate, utils


class Network:

    """A network of nodes.

    Represents the network we're analyzing and holds auxilary data about it.

    Examples:
        In a 3-node network, ``a_network.tpm[(0, 0, 1)]`` gives the transition
        probabilities for each node at |t_0| given that state at |t_{-1}| was
        |0,0,1|.

    Attributes:
        size (int):
            The number of nodes in the network.
        tpm (np.ndarray):
            The transition probability matrix for this network. Must be
            provided in state-by-node form. It can be either 2-dimensional, so
            that ``tpm[i]`` gives the probabilities of each node being on if
            the past state is given by the binary representation of ``i``, or
            in N-D form, so that ``tpm[0][1][0]`` gives the probabilities of
            each node being on if the past state is |0,1,0|. The shape of the
            2-dimensional form of the TPM must be ``(S, N)``, and the shape of
            the N-D form of the TPM must be ``[2] * N + [N]``, where ``S`` is
            the number of states and ``N`` is the number of nodes in the
            network.
        current_state (tuple):
            The current state of the network. ``current_state[i]`` gives the
            current state of node ``i``.
        past_state (tuple):
            The past state of the network. ``past_state[i]`` gives the past
            state of node ``i``.
        connectivity_matrix (np.ndarray):
            A matrix describing the network's connectivity.
            ``connectivity_matrix[i][j] == 1`` means that node
            ``i`` is connected to node ``j``.
        nodes (list(Node)):
            A list of nodes in the network.

    """

    def __init__(self, tpm, current_state, past_state,
                 connectivity_matrix=None):
        # Coerce input to np.arrays
        tpm = np.array(tpm)
        if connectivity_matrix is not None:
            connectivity_matrix = np.array(connectivity_matrix)
        # Get the number of nodes in the network.
        # The TPM can be either 2-dimensional or in N-D form, where transition
        # probabilities can be indexed by state-tuples. In either case, the
        # size of last dimension is the number of nodes.
        self.size = tpm.shape[-1]
        # TODO make tpm also optional when implementing logical network
        # definition
        self.tpm = tpm.reshape([2] * self.size + [self.size]).astype(float)
        # TODO! test connectivity matrix
        self.connectivity_matrix = connectivity_matrix
        self.current_state = current_state
        self.past_state = past_state
        # Make the TPM and connectivity matrix immutable (for hashing)
        self.tpm.flags.writeable = False
        if self.connectivity_matrix is not None:
            self.connectivity_matrix.flags.writeable = False

        tpm_hash = utils.np_hash(self.tpm)
        cm_hash = (utils.np_hash(self.connectivity_matrix)
                   if self.connectivity_matrix is not None else None)
        # Only compute hash once
        self._hash = hash((tpm_hash,
                           self.current_state,
                           self.past_state,
                           cm_hash))

        # Validate this network
        validate.network(self)

        # Generate the nodes
        self.nodes = tuple([Node(self, node_index) for node_index in
                            range(self.size)])

        # TODO extend to nonbinary nodes
        self.num_states = 2 ** self.size

    def __repr__(self):
        return ("Network(" + ", ".join([repr(self.tpm),
                                        repr(self.current_state),
                                        repr(self.past_state)]) +
                ", connectivity_matrix=" + repr(self.connectivity_matrix) +
                ")")

    def __str__(self):
        return ("Network(" + str(self.tpm) + ", connectivity_matrix=" +
                str(self.connectivity_matrix) + ")")

    def __eq__(self, other):
        """Return whether this network equals the other object.

        Two networks are equal if they have the same TPM, current state, and
        past state.
        """
        return ((np.array_equal(self.tpm, other.tpm) and
                np.array_equal(self.current_state, other.current_state) and
                np.array_equal(self.past_state, other.past_state) and
                np.array_equal(self.connectivity_matrix,
                               other.connectivity_matrix))
                if isinstance(other, type(self)) else False)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self._hash

    def subsystems(self):
        """Return a generator of all possible subsystems of this network.

        This is the just powerset of the network's set of nodes."""
        for subset in utils.powerset(self.nodes):
            yield Subsystem(subset, self.current_state, self.past_state, self)
