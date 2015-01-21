#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# network.py
"""
Represents the network of interest. This is the primary object of PyPhi and the
context of all |phi| and |big_phi| computation.
"""

import numpy as np
from . import validate, utils, json, convert


# TODO!!! raise error if user tries to change TPM or CM, double-check and document
# that states can be changed


class Network:

    """A network of nodes.

    Represents the network we're analyzing and holds auxilary data about it.

    Example:
        In a 3-node network, ``a_network.tpm[(0, 0, 1)]`` gives the transition
        probabilities for each node at |t_0| given that state at |t_{-1}| was
        |N_0 = 0, N_1 = 0, N_2 = 1|.

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
            The network's transition probability matrix. It can be provided in
            either state-by-node (either 2-D or N-D) or state-by-state form. In
            either form, row indices must follow the **LOLI** convention (see
            discussion in :mod:`pyphi.examples`), and in state-by-state form,
            so must column indices. If given in state-by-node form, it can be
            either 2-dimensional, so that ``tpm[i]`` gives the probabilities of
            each node being on if the past state is encoded by |i| according to
            **LOLI**, or in N-D form, so that ``tpm[(0, 0, 1)]`` gives the
            probabilities of each node being on if the past state is |N_0 = 0,
            N_1 = 0, N_2 = 1|. The shape of the 2-dimensional form of a
            state-by-node TPM must be ``(S, N)``, and the shape of the N-D form
            of the TPM must be ``[2] * N + [N]``, where ``S`` is the number of
            states and ``N`` is the number of nodes in the network.
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
                 connectivity_matrix=None, perturb_vector=None):
        # Cast TPM to np.array.
        tpm = np.array(tpm)
        # Validate TPM.
        # The TPM can be either 2-dimensional or in N-D form, where transition
        # probabilities can be indexed by state-tuples.
        validate.tpm(tpm)
        # Convert to N-D state-by-node if we were given a square state-by-state
        # TPM. Otherwise, force conversion to N-D format.
        if tpm.ndim == 2 and tpm.shape[0] == tpm.shape[1]:
            tpm = convert.state_by_state2state_by_node(tpm)
        else:
            tpm = convert.to_n_dimensional(tpm)

        self.tpm = tpm
        # Get the number of nodes in the network.
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

        # Get pertubation vector.
        if perturb_vector is not None:
            perturb_vector = np.array(perturb_vector)
        else:
            # If none was provided, assume maximum-entropy.
            perturb_vector = np.ones(self.size) / 2

        self.perturb_vector = perturb_vector
        self.connectivity_matrix = connectivity_matrix
        # Coerce current and past state to tuples so they can be properly used
        # as np.array indices.
        self.current_state = tuple(current_state)
        self.past_state = tuple(past_state)
        # Make the TPM, pertubation vector  and connectivity matrix immutable
        # (for hashing).
        self.tpm.flags.writeable = False
        self.connectivity_matrix.flags.writeable = False
        self.perturb_vector.flags.writeable = False

        self._pv_hash = utils.np_hash(self.perturb_vector)
        self._tpm_hash = utils.np_hash(self.tpm)
        self._cm_hash = utils.np_hash(self.connectivity_matrix)

        # TODO extend to nonbinary nodes
        self.num_states = 2 ** self.size

        # Validate the entire network.
        validate.network(self)

    def __repr__(self):
        return ("Network(" + ", ".join([repr(self.tpm),
                                        repr(self.current_state),
                                        repr(self.past_state)]) +
                ", connectivity_matrix=" + repr(self.connectivity_matrix) +
                ", perturb_vector=" + repr(self.perturb_vector) +
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
                               other.connectivity_matrix) and
                np.array_equal(self.perturb_vector,
                               other.perturb_vector))
                if isinstance(other, type(self)) else False)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._tpm_hash, self.current_state, self.past_state,
                     self._cm_hash, self._pv_hash))

    def json_dict(self):
        return {
            'tpm': json.make_encodable(self.tpm),
            'current_state': json.make_encodable(self.current_state),
            'past_state': json.make_encodable(self.past_state),
            'connectivity_matrix':
                json.make_encodable(self.connectivity_matrix),
            'size': json.make_encodable(self.size),
        }
