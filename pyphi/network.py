#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network
~~~~~~~

Represents the network of interest. This is the primary object of PyPhi and the
context of all |phi| and |big_phi| computation.
"""

import numpy as np
from .node import Node
from . import validate, utils


# TODO!!! raise error if user tries to change TPM or CM, double-check and document
# that states can be changed


class Network:

    """A network of nodes.

    Represents the network we're analyzing and holds auxilary data about it.

    Example:
        In a 3-node network, ``a_network.tpm[(0, 0, 1)]`` gives the transition
        probabilities for each node at |t_0| given that state at |t_{-1}| was
        |0,0,1|.

    Args:
        tpm (np.ndarray): See the corresponding attribute.
        current_state (tuple): See the corresponding attribute.
        past_state (tuple): See the corresponding attribute.

    Keyword Args:
        connectivity_matrix (array or sequence): A square binary adjacency
            matrix indicating the connections between nodes in the network.
            ``connectivity_matrix[i][j] == 1`` means that node |i| is connected
            to node |j|. If no connectivity matrix is given, every node is
            connected to every node **(including itself)**.

    Attributes:
        tpm (np.ndarray):
            The transition probability matrix that encodes the network's
            mechanism. It can be provided in either state-by-node or
            state-by-state form. In state-by-state form, decimal indices must
            correspond to states so that lower-order bits of the binary
            representation of the index give the state of low-index nodes. See
            :func:`utils.state_by_state2state_by_node` for more info. If given
            in state-by-node form, it can be either 2-dimensional, so that
            ``tpm[i]`` gives the probabilities of each node being on if the
            past state is given by the binary representation of |i|, or in N-D
            form, so that ``tpm[0][1][0]`` gives the probabilities of each node
            being on if the past state is |0,1,0|. The shape of the
            2-dimensional form of the TPM must be ``(S, N)``, and the shape of
            the N-D form of the TPM must be ``[2]
            * N + [N]``, where ``S`` is the number of states and ``N`` is the
            number of nodes in the network.
        current_state (tuple):
            The current state of the network. ``current_state[i]`` gives the
            current state of node |i|.
        past_state (tuple):
            The past state of the network. ``past_state[i]`` gives the past
            state of node |i|.
        connectivity_matrix (np.ndarray):
            A square binary adjacency matrix indicating the connections between
            nodes in the network.
        size (int):
            The number of nodes in the network.
        num_states (int):
            The number of possible states of the network.
    """

    def __init__(self, tpm, current_state, past_state,
                 connectivity_matrix=None):
        # Cast TPM to np.array.
        tpm = np.array(tpm)
        # Convert to state-by-node if we were given a square state-by-state
        # TPM.
        if tpm.ndim == 2 and tpm.shape[0] == tpm.shape[1]:
            tpm = utils.state_by_state2state_by_node(tpm)
        # Get the number of nodes in the network.
        # The TPM can be either 2-dimensional or in N-D form, where transition
        # probabilities can be indexed by state-tuples. In either case, the
        # size of last dimension is the number of nodes.
        self.size = tpm.shape[-1]
        self.node_indices = tuple(range(self.size))

        # Get the connectivity matrix.
        if connectivity_matrix is not None:
            connectivity_matrix = np.array(connectivity_matrix)
        else:
            # If none was provided, assume all are connected.
            connectivity_matrix = np.ones((self.size, self.size))

        # TODO make tpm also optional when implementing logical network
        # definition

        # We use Fortran ordering here so that low-order bits correspond to
        # low-index nodes. Note that this does not change the memory layout (C-
        # or Fortran-contiguous), so there is no performance loss.
        self.tpm = tpm.reshape([2] * self.size + [self.size],
                               order="F").astype(float)

        self.connectivity_matrix = connectivity_matrix
        self.current_state = current_state
        self.past_state = past_state
        # Make the TPM and connectivity matrix immutable (for hashing).
        self.tpm.flags.writeable = False
        self.connectivity_matrix.flags.writeable = False

        self._tpm_hash = utils.np_hash(self.tpm)
        self._cm_hash = utils.np_hash(self.connectivity_matrix)

        # Validate this network.
        validate.network(self)

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
        return hash((self._tpm_hash, self.current_state, self.past_state,
                     self._cm_hash))
