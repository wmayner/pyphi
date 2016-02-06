#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

"""
Represents the network of interest. This is the primary object of PyPhi and the
context of all |small_phi| and |big_phi| computation.
"""

import json

import numpy as np

from . import cache, convert, utils, validate
from .constants import DIRECTIONS, FUTURE, PAST
from .jsonify import jsonify


# TODO!!! raise error if user tries to change TPM or CM, double-check and
# document that states can be changed

def from_json(filename):
    """Convert a JSON representation of a network to a PyPhi network.

    Args:
        filename (str): A path to a JSON file representing a network.

    Returns:
       network (``Network``): The corresponding PyPhi network object.
    """
    with open(filename) as f:
        network_dictionary = json.load(f)
    tpm = network_dictionary['tpm']
    cm = network_dictionary['cm']
    network = Network(tpm, connectivity_matrix=cm)
    return network


class Network:
    """A network of nodes.

    Represents the network we're analyzing and holds auxilary data about it.

    Example:
        In a 3-node network, ``a_network.tpm[(0, 0, 1)]`` gives the transition
        probabilities for each node at |t_0| given that state at |t_{-1}| was
        |N_0 = 0, N_1 = 0, N_2 = 1|.

    Args:
        tpm (np.ndarray): See the corresponding attribute.

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
            discussion in the |examples| module), and in state-by-state form,
            so must column indices. If given in state-by-node form, it can be
            either 2-dimensional, so that ``tpm[i]`` gives the probabilities of
            each node being on if the past state is encoded by |i| according to
            **LOLI**, or in N-D form, so that ``tpm[(0, 0, 1)]`` gives the
            probabilities of each node being on if the past state is |N_0 = 0,
            N_1 = 0, N_2 = 1|. The shape of the 2-dimensional form of a
            state-by-node TPM must be ``(S, N)``, and the shape of the N-D form
            of the TPM must be ``[2] * N + [N]``, where ``S`` is the number of
            states and ``N`` is the number of nodes in the network.
        connectivity_matrix (np.ndarray):
            A square binary adjacency matrix indicating the connections between
            nodes in the network.
        size (int):
            The number of nodes in the network.
        num_states (int):
            The number of possible states of the network.
    """

    # TODO make tpm also optional when implementing logical network definition
    def __init__(self, tpm, connectivity_matrix=None,
                 perturb_vector=None, purview_cache=None):
        self.tpm = tpm
        self._size = self.tpm.shape[-1]
        # TODO extend to nonbinary nodes
        self._num_states = 2 ** self.size
        self._node_indices = tuple(range(self.size))
        self.connectivity_matrix = connectivity_matrix
        self.perturb_vector = perturb_vector
        self.purview_cache = purview_cache or cache.PurviewCache()
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

    # TODO: this should really be a Subsystem method, but we're
    # interested in caching at the Network-level...
    @cache.method('purview_cache')
    def _potential_purviews(self, direction, mechanism):
        """All purviews which are not clearly reducible for mechanism.

        Args:
            direction (str): |past| or |future|
            mechanism (tuple(int)): The mechanism which all purviews
                are checked for reducibility over.
        Returns:
            purviews (list(tuple(int))): All purviews which are
                irreducible over `mechanism`.
        """
        all_purviews = utils.powerset(self._node_indices)

        if direction == DIRECTIONS[PAST]:
            return [purview for purview in all_purviews
                    if not utils.block_reducible(self.connectivity_matrix,
                                                 purview, mechanism)]
        elif direction == DIRECTIONS[FUTURE]:
            return [purview for purview in all_purviews
                    if not utils.block_reducible(self.connectivity_matrix,
                                                 mechanism, purview)]

    def __repr__(self):
        return ('Network({}, connectivity_matrix={}, '
                'perturb_vector={})'.format(repr(self.tpm),
                                            repr(self.connectivity_matrix),
                                            repr(self.perturb_vector)))

    def __str__(self):
        return 'Network({}, connectivity_matrix={})'.format(
            self.tpm, self.connectivity_matrix)

    def __eq__(self, other):
        """Return whether this network equals the other object.

        Two networks are equal if they have the same TPM, connectivity matrix,
        and perturbation vector.
        """
        return (np.array_equal(self.tpm, other.tpm)
                and np.array_equal(self.connectivity_matrix,
                                   other.connectivity_matrix)
                and np.array_equal(self.perturb_vector, other.perturb_vector)
                if isinstance(other, type(self)) else False)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # TODO: hash only once?
        return hash((self._tpm_hash, self._cm_hash, self._pv_hash))

    def to_json(self):
        return {
            'tpm': jsonify(self.tpm),
            'cm': jsonify(self.connectivity_matrix),
            'size': jsonify(self.size)
        }
