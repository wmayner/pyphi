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
from .constants import Direction
from .node import default_labels


def immutable(array):
    """Make a numpy array immutable."""
    array.flags.writeable = False
    return array


class Network:
    """A network of nodes.

    Represents the network we're analyzing and holds auxilary data about it.

    Args:
        tpm (np.ndarray): The transition probability matrix of the network.

            The TPM can be provided in either state-by-node (either |2-D| or
            |N-D|) or state-by-state form. In either form, row indices must
            follow the **LOLI** convention (see discussion in the |examples|
            module.) In state-by-state form column indices must also follow
            **LOLI** convention.

            If given in state-by-node form, the TPM can be either
            2-dimensional, so that ``tpm[i]`` gives the probabilities of each
            node being on if the past state is encoded by |i| according to
            **LOLI**, or in |N-D| form, so that ``tpm[(0, 0, 1)]`` gives the
            probabilities of each node being on if the past state is
            |N_0 = 0, N_1 = 0, N_2 = 1|.

            The shape of the 2-dimensional form of a state-by-node TPM must be
            ``(S, N)``, and the shape of the |N-D| form of the TPM must be
            ``[2] * N + [N]``, where ``S`` is the number of states and |N| is
            the number of nodes in the network.

    Keyword Args:
        connectivity_matrix (np.ndarray): A square binary adjacency matrix
            indicating the connections between nodes in the network.
            ``connectivity_matrix[i][j] == 1`` means that node |i| is connected
            to node |j|. If no connectivity matrix is given, every node is
            connected to every node **(including itself)**.
        node_labels (tuple[str]): Human readable labels for each node in the
            network.

    Example:
        In a 3-node network, ``a_network.tpm[(0, 0, 1)]`` gives the transition
        probabilities for each node at |t_0| given that state at |t_{-1}| was
        |N_0 = 0, N_1 = 0, N_2 = 1|.
    """

    # TODO make tpm also optional when implementing logical network definition
    def __init__(self, tpm, connectivity_matrix=None, node_labels=None,
                 purview_cache=None):

        self._tpm, self._tpm_hash = self._build_tpm(tpm)
        self._cm, self._cm_hash = self._build_cm(connectivity_matrix)
        self._node_indices = tuple(range(self.size))
        self._node_labels = node_labels or default_labels(self._node_indices)
        self.purview_cache = purview_cache or cache.PurviewCache()

        validate.network(self)

    @property
    def tpm(self):
        """np.ndarray: The network's transition probability matrix, in |N-D|
        form."""
        return self._tpm

    def _build_tpm(self, tpm):
        """Validate the TPM passed by the user and convert to |N-D| form. """
        tpm = np.array(tpm)

        validate.tpm(tpm)

        # Convert to N-D state-by-node form
        if utils.state_by_state(tpm):
            tpm = convert.state_by_state2state_by_node(tpm)
        else:
            tpm = convert.to_n_dimensional(tpm)

        immutable(tpm)

        return (tpm, utils.np_hash(tpm))

    @property
    def cm(self):
        """np.ndarray: The network's connectivity matrix.

        A square binary adjacency matrix indicating the connections between
        nodes in the network.
        """
        return self._cm

    def _build_cm(self, cm):
        """Convert the passed CM to the proper format, or construct the
        unitary CM if none was provided."""
        if cm is None:
            # Assume all are connected.
            cm = np.ones((self.size, self.size))
        else:
            cm = np.array(cm)

        immutable(cm)

        return (cm, utils.np_hash(cm))

    @property
    def connectivity_matrix(self):
        """np.ndarray: Alias for `Network.cm`."""
        return self._cm

    @property
    def size(self):
        """int: The number of nodes in the network."""
        return self.tpm.shape[-1]

    # TODO extend to nonbinary nodes
    @property
    def num_states(self):
        """int: The number of possible states of the network."""
        return 2 ** self.size

    @property
    def node_indices(self):
        """tuple[int]: The indices of nodes in the network.

        This is ``0..network.size``.
        """
        return self._node_indices

    @property
    def node_labels(self):
        """tuple[str]: The labels of nodes in the network."""
        return self._node_labels

    def labels2indices(self, labels):
        """Convert a tuple of node labels to node indices."""
        _map = dict(zip(self.node_labels, self.node_indices))
        return tuple(_map[label] for label in labels)

    def indices2labels(self, indices):
        """Convert a tuple of node indices to node labels."""
        _map = dict(zip(self.node_indices, self.node_labels))
        return tuple(_map[index] for index in indices)

    def parse_node_indices(self, nodes):
        """Returns the nodes indices for nodes, where ``nodes`` is either
        already integer indices or node labels."""
        if not nodes:
            indices = ()
        elif all(isinstance(node, str) for node in nodes):
            indices = self.labels2indices(nodes)
        else:
            indices = map(int, nodes)
        return tuple(sorted(set(indices)))

    # TODO: this should really be a Subsystem method, but we're
    # interested in caching at the Network-level...
    @cache.method('purview_cache')
    def _potential_purviews(self, direction, mechanism):
        """All purviews which are not clearly reducible for mechanism.

        Args:
            direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
            :const:`~pyphi.constants.Direction.FUTURE`.
            mechanism (tuple[int]): The mechanism which all purviews are
                checked for reducibility over.

        Returns:
            list[tuple[int]]: All purviews which are irreducible over
                ``mechanism``.
        """
        all_purviews = utils.powerset(self._node_indices)
        return irreducible_purviews(self.cm, direction, mechanism,
                                    all_purviews)

    def __repr__(self):
        return 'Network({}, connectivity_matrix={})'.format(self.tpm, self.cm)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this network equals the other object.

        Two networks are equal if they have the same TPM and CM.
        """
        return (np.array_equal(self.tpm, other.tpm)
                and np.array_equal(self.cm, other.cm)
                if isinstance(other, type(self)) else False)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # TODO: hash only once?
        return hash((self._tpm_hash, self._cm_hash))

    def to_json(self):
        return {
            'tpm': self.tpm,
            'cm': self.cm,
            'size': self.size,
            'labels': self.node_labels
        }

    @classmethod
    def from_json(cls, json):
        return Network(json['tpm'], json['cm'], node_labels=json['labels'])


def irreducible_purviews(cm, direction, mechanism, purviews):
    """Returns all purview which are irreducible for the mechanism.

    Args:
        cm (np.ndarray): A |N x N| connectivity matrix.
        direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
            :const:`~pyphi.constants.Direction.FUTURE`.
        purviews (list[tuple[int]]): The purviews to check.
        mechanism (tuple[int]): The mechanism in question.

    Returns:
        list[tuple[int]]: All purviews in ``purviews`` which are not reducible
            over ``mechanism``.

    Raises:
        ValueError: If ``direction`` is invalid.
    """
    def reducible(purview):
        # Returns True if purview is trivially reducible.
        if direction == Direction.PAST:
            _from, to = purview, mechanism
        elif direction == Direction.FUTURE:
            _from, to = mechanism, purview
        else:
            # TODO: test that ValueError is raised
            validate.direction(direction)
        return utils.block_reducible(cm, _from, to)

    return [purview for purview in purviews if not reducible(purview)]


def from_json(filename):
    """Convert a JSON representation of a network to a PyPhi network.

    Args:
        filename (str): A path to a JSON file representing a network.

    Returns:
       |Network|: The corresponding PyPhi network object.
    """
    with open(filename) as f:
        loaded = json.load(f)

    return Network.from_json(loaded)
