#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# network.py
"""
Represents the network of interest. This is the primary object of PyPhi and the
context of all |small_phi| and |big_phi| computation.
"""

import json
import numpy as np
from . import validate, utils, convert, config
from .json import make_encodable
from .constants import DIRECTIONS, PAST, FUTURE

# TODO!!! raise error if user tries to change TPM or CM, double-check and
# document that states can be changed


# Methods to compute reducible purviews for any mechanism, so they do not have
# to be checked in concept calculation.


def from_json(filename):
    """Convert a JSON representation of a network to a PyPhi network.

    Args:
        filename (str): A path to a JSON file representing a network.

    Returns:
       ``Network`` -- The corresponding PyPhi network object.
    """
    with open(filename) as f:
        network_dictionary = json.load(f)
    tpm = network_dictionary['tpm']
    current_state = network_dictionary['currentState']
    past_state = network_dictionary['pastState']
    cm = network_dictionary['connectivityMatrix']
    network = Network(tpm, current_state, past_state, connectivity_matrix=cm)
    return network


def list_past_purview(self, mechanism):
    return _build_purview_list(self, mechanism, 'past')


def list_future_purview(self, mechanism):
    return _build_purview_list(self, mechanism, 'future')


def _build_purview_list(self, mechanism, direction):
    if direction == DIRECTIONS[PAST]:
        return [purview for purview in utils.powerset(self._node_indices)
                if utils.not_block_reducible(self.connectivity_matrix, purview,
                                             mechanism)]
    elif direction == DIRECTIONS[FUTURE]:
        return [purview for purview in utils.powerset(self._node_indices)
                if utils.not_block_reducible(self.connectivity_matrix,
                                             mechanism, purview)]


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

    # TODO make tpm also optional when implementing logical network definition
    def __init__(self, tpm, current_state, past_state,
                 connectivity_matrix=None, perturb_vector=None,
                 purview_cache=None):
        self.tpm = tpm

        self._size = self.tpm.shape[-1]
        # TODO extend to nonbinary nodes
        self._num_states = 2 ** self.size
        self._node_indices = tuple(range(self.size))

        self._current_state = tuple(current_state)
        self._past_state = tuple(past_state)
        self.perturb_vector = perturb_vector
        self.connectivity_matrix = connectivity_matrix
        if purview_cache is None:
            purview_cache = dict()
        self.purview_cache = purview_cache
        # If CACHE_POTENTIAL_PURVIEWS is set to True then pre-compute the list
        # for each mechanism. This will save time if results are desired for
        # more than one subsystem of the network.
        if config.CACHE_POTENTIAL_PURVIEWS:
            self.build_purview_cache()
        # Validate the entire network.
        validate.network(self)

    @property
    def size(self):
        return self._size

    @property
    def num_states(self):
        return self._num_states

    @property
    def node_indices(self):
        return self._node_indices

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, current_state):
        # Cast current state to a tuple so it can be hashed and properly used
        # as np.array indices.
        current_state = tuple(current_state)
        # Validate it.
        validate.current_state_length(current_state, self.size)
        if config.VALIDATE_NETWORK_STATE:
            validate.state_reachable(self.past_state, current_state, self.tpm)
            validate.state_reachable_from(self.past_state, current_state,
                                          self.tpm)
        self._current_state = current_state

    @property
    def past_state(self):
        return self._past_state

    @past_state.setter
    def past_state(self, past_state):
        # Cast past state to a tuple so it can be hashed and properly used
        # as np.array indices.
        past_state = tuple(past_state)
        # Validate it.
        validate.past_state_length(past_state, self.size)
        if config.VALIDATE_NETWORK_STATE:
            validate.state_reachable_from(past_state, self.current_state, self.tpm)
        self._past_state = past_state

    @property
    def tpm(self):
        return self._tpm

    @tpm.setter
    def tpm(self, tpm):
        # Cast TPM to np.array.
        tpm = np.array(tpm)
        # Validate TPM.
        # The TPM can be either 2-dimensional or in N-D form, where transition
        # probabilities can be indexed by state-tuples.
        validate.tpm(tpm)
        # Convert to N-D state-by-node if we were given a square state-by-state
        # TPM. Otherwise, force conversion to N-D format.
        if tpm.ndim == 2 and tpm.shape[0] == tpm.shape[1]:
            self._tpm = convert.state_by_state2state_by_node(tpm)
        else:
            self._tpm = convert.to_n_dimensional(tpm)
        # Make the underlying attribute immutable.
        self._tpm.flags.writeable = False
        # Update hash.
        self._tpm_hash = utils.np_hash(self.tpm)

    @property
    def connectivity_matrix(self):
        return self._connectivity_matrix

    @connectivity_matrix.setter
    def connectivity_matrix(self, cm):
        # Get the connectivity matrix.
        if cm is not None:
            self._connectivity_matrix = np.array(cm)
        else:
            # If none was provided, assume all are connected.
            self._connectivity_matrix = np.ones((self.size, self.size))
        # Make the underlying attribute immutable.
        self._connectivity_matrix.flags.writeable = False
        # Update hash.
        self._cm_hash = utils.np_hash(self.connectivity_matrix)

    @property
    def perturb_vector(self):
        return self._perturb_vector

    @perturb_vector.setter
    def perturb_vector(self, perturb_vector):
        # Get pertubation vector.
        if perturb_vector is not None:
            self._perturb_vector = np.array(perturb_vector)
        else:
            # If none was provided, assume maximum-entropy.
            self._perturb_vector = np.ones(self.size) / 2
        # Make the underlying attribute immutable.
        self._perturb_vector.flags.writeable = False
        # Update hash.
        self._pv_hash = utils.np_hash(self.perturb_vector)

    def build_purview_cache(self):
        for index in utils.powerset(self._node_indices):
            for direction in DIRECTIONS:
                key = (direction, index)
                self.purview_cache[key] = _build_purview_list(self, index,
                                                              direction)

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
        # TODO: hash only once?
        return hash((self._tpm_hash, self.current_state, self.past_state,
                     self._cm_hash, self._pv_hash))

    def json_dict(self):
        return {
            'tpm': make_encodable(self.tpm),
            'current_state': make_encodable(self.current_state),
            'past_state': make_encodable(self.past_state),
            'connectivity_matrix':
                make_encodable(self.connectivity_matrix),
            'size': make_encodable(self.size),
        }
