#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

"""
Represents the network of interest. This is the primary object of PyPhi and the
context of all |small_phi| and |big_phi| computation.
"""

import numpy as np

from . import cache, connectivity, jsonify, utils, validate
from .labels import NodeLabels
from .tpm import ExplicitTPM


class Network:
    """A network of nodes.

    Represents the network under analysis and holds auxilary data about it.

    Args:
        tpm (np.ndarray): The transition probability matrix of the network.
            See :func:`pyphi.tpm.ExplicitTPM`.

    Keyword Args:
        cm (np.ndarray): A square binary adjacency matrix indicating the
            connections between nodes in the network. ``cm[i][j] == 1`` means
            that node |i| is connected to node |j| (see :ref:`cm-conventions`).
            **If no connectivity matrix is given, PyPhi assumes that every node
            is connected to every node (including itself)**.
        node_labels (tuple[str] or |NodeLabels|): Human-readable labels for
            each node in the network.
    """

    # TODO make tpm also optional when implementing logical network definition
    def __init__(self, tpm, cm=None, node_labels=None, purview_cache=None):
        # Initialize _tpm according to argument type.
        if isinstance(tpm, ExplicitTPM):
            self._tpm = tpm
        elif isinstance(tpm, np.ndarray):
            self._tpm = ExplicitTPM(tpm, validate=True)
        elif isinstance(tpm, dict):
            # From JSON.
            self._tpm = ExplicitTPM(tpm["_tpm"], validate=True)
        else:
            raise TypeError(f"Invalid tpm of type {type(tpm)}.")

        self._cm, self._cm_hash = self._build_cm(cm)
        self._node_indices = tuple(range(self.size))
        self._node_labels = NodeLabels(node_labels, self._node_indices)
        self.purview_cache = purview_cache or cache.PurviewCache()

        validate.network(self)

    @property
    def tpm(self):
        """pyphi.tpm.ExplicitTPM: The TPM object which contains this
        network's transition probability matrix, in multidimensional
        form.
        """
        return self._tpm

    @property
    def cm(self):
        """np.ndarray: The network's connectivity matrix.

        A square binary adjacency matrix indicating the connections between
        nodes in the network.
        """
        return self._cm

    def _build_cm(self, cm):
        """Convert the passed CM to the proper format, or construct the
        unitary CM if none was provided.
        """
        if cm is None:
            # Assume all are connected.
            cm = np.ones((self.size, self.size))
        else:
            cm = np.array(cm)

        utils.np_immutable(cm)

        return (cm, utils.np_hash(cm))

    @property
    def connectivity_matrix(self):
        """np.ndarray: Alias for ``cm``."""
        return self._cm

    @property
    def causally_significant_nodes(self):
        """See :func:`pyphi.connectivity.causally_significant_nodes`."""
        return connectivity.causally_significant_nodes(self.cm)

    @property
    def size(self):
        """int: The number of nodes in the network."""
        return len(self)

    # TODO extend to nonbinary nodes
    @property
    def num_states(self):
        """int: The number of possible states of the network."""
        return 2 ** self.size

    @property
    def node_indices(self):
        """tuple[int]: The indices of nodes in the network.

        This is equivalent to ``tuple(range(network.size))``.
        """
        return self._node_indices

    @property
    def node_labels(self):
        """tuple[str]: The labels of nodes in the network."""
        return self._node_labels

    # TODO: this should really be a Subsystem method, but we're
    # interested in caching at the Network-level...
    @cache.method("purview_cache")
    def potential_purviews(self, direction, mechanism):
        """All purviews which are not clearly reducible for mechanism.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism which all purviews are
                checked for reducibility over.

        Returns:
            list[tuple[int]]: All purviews which are irreducible over
            ``mechanism``.
        """
        all_purviews = utils.powerset(self._node_indices)
        return irreducible_purviews(self.cm, direction, mechanism, all_purviews)

    def __len__(self):
        """int: The number of nodes in the network."""
        return self.tpm.shape[-1]

    def __repr__(self):
        return "Network({}, cm={})".format(self.tpm, self.cm)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this network equals the other object.

        Networks are equal if they have the same TPM and CM.
        """
        return (
            isinstance(other, Network)
            and self.tpm.array_equal(other.tpm)
            and np.array_equal(self.cm, other.cm)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((hash(self.tpm), self._cm_hash))

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            "tpm": self.tpm,
            "cm": self.cm,
            "size": self.size,
            "node_labels": self.node_labels,
        }

    @classmethod
    def from_json(cls, json_dict):
        """Return a |Network| object from a JSON dictionary representation."""
        del json_dict["size"]
        return Network(**json_dict)


def irreducible_purviews(cm, direction, mechanism, purviews):
    """Return all purviews which are irreducible for the mechanism.

    Args:
        cm (np.ndarray): An |N x N| connectivity matrix.
        direction (Direction): |CAUSE| or |EFFECT|.
        purviews (list[tuple[int]]): The purviews to check.
        mechanism (tuple[int]): The mechanism in question.

    Returns:
        list[tuple[int]]: All purviews in ``purviews`` which are not reducible
        over ``mechanism``.

    Raises:
        ValueError: If ``direction`` is invalid.
    """

    def reducible(purview):
        """Return ``True`` if purview is trivially reducible."""
        _from, to = direction.order(mechanism, purview)
        return connectivity.block_reducible(cm, _from, to)

    return [purview for purview in purviews if not reducible(purview)]


def from_json(filename):
    """Convert a JSON network to a PyPhi network.

    Args:
        filename (str): A path to a JSON file representing a network.

    Returns:
       Network: The corresponding PyPhi network object.
    """
    with open(filename) as f:
        return jsonify.load(f)
