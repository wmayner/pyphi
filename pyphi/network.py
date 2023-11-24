#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

"""
Represents the network of interest. This is the primary object of PyPhi and the
context of all |small_phi| and |big_phi| computation.
"""

from typing import Iterable
import numpy as np

from . import cache, connectivity, jsonify, utils, validate
from .labels import NodeLabels
from .node import generate_nodes, node as Node
from .tpm import ExplicitTPM, ImplicitTPM
from .state_space import build_state_space


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
        state_space (Optional[tuple[tuple[Union[int, str]]]]):
            Labels for the state space of each node in the network. If ``None``,
            states will be automatically labeled using a zero-based integer
            index per node.
    """

    def __init__(
            self,
            tpm,
            cm=None,
            node_labels=None,
            state_space=None,
            purview_cache=None
    ):
        # Initialize _tpm according to argument type.

        if isinstance(tpm, (np.ndarray, ExplicitTPM)):
            # Validate TPM and convert to state-by-node multidimensional format.
            tpm = ExplicitTPM(tpm, validate=True)

            self._cm, self._cm_hash = self._build_cm(cm, tpm)

            self._node_indices = tuple(range(self.size))
            self._node_labels = NodeLabels(node_labels, self._node_indices)

            self._state_space, _ = build_state_space(
                self._node_labels,
                tpm.shape[:-1],
                state_space
            )

            self._tpm = ImplicitTPM(
                generate_nodes(
                    tpm,
                    self._cm,
                    self._state_space,
                    self._node_indices,
                    self._node_labels
                )
            )

        elif isinstance(tpm, Iterable):
            invalid = [
                i for i in tpm if not isinstance(i, (np.ndarray, ExplicitTPM))
            ]

            if invalid:
                raise TypeError("Invalid set of nodes containing {}.".format(
                    ', '.join(str(i) for i in invalid)
                ))

            tpm = tuple(
                ExplicitTPM(node_tpm, validate=False) for node_tpm in tpm
            )

            shapes = [node.shape for node in tpm]

            self._cm, self._cm_hash = self._build_cm(cm, tpm, shapes)

            self._node_indices = tuple(range(self.size))
            self._node_labels = NodeLabels(node_labels, self._node_indices)

            network_tpm_shape = ImplicitTPM._node_shapes_to_shape(shapes)
            self._state_space, _ = build_state_space(
                self._node_labels,
                network_tpm_shape[:-1],
                state_space
            )

            self._tpm = ImplicitTPM(
                tuple(
                    Node(
                        node_tpm,
                        self._cm,
                        self._state_space,
                        index,
                        node_labels=self._node_labels
                    ).pyphi_accessor
                    for index, node_tpm in zip(self._node_indices, tpm)
                )
            )

        elif isinstance(tpm, ImplicitTPM):
            self._tpm = tpm
            self._cm, self._cm_hash = self._build_cm(cm, self._tpm)
            self._node_indices = tuple(range(self.size))
            self._node_labels = NodeLabels(node_labels, self._node_indices)
            self._state_space, _ = build_state_space(
                self._node_labels,
                self._tpm.shape[:-1],
                state_space
            )

        # FIXME(TPM) initialization from JSON
        elif isinstance(tpm, dict):
            # From JSON.
            self._tpm = ImplicitTPM(tpm["_tpm"])
            self._cm, self._cm_hash = self._build_cm(cm, tpm)
            self._node_indices = tuple(range(self.size))
            self._node_labels = NodeLabels(node_labels, self._node_indices)

        else:
            raise TypeError(f"Invalid TPM of type {type(tpm)}.")

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

    def _build_cm(self, cm, tpm, shapes=None):
        """Convert the passed CM to the proper format, or construct the
        unitary CM if none was provided (explicit TPM), or infer from node TPMs.
        """
        if cm is None:
            if hasattr(tpm, "shape"):
                network_size = tpm.shape[-1]
            else:
                network_size = len(tpm)

            # Explicit TPM without connectivity matrix: assume all are connected.
            if shapes is None:
                cm = np.ones((network_size, network_size), dtype=int)
                utils.np_immutable(cm)
                return (cm, utils.np_hash(cm))

            # ImplicitTPM without connectivity matrix: infer from node TPMs.
            cm = np.zeros((network_size, network_size), dtype=int)

            for i, shape in enumerate(shapes):
                for j in range(len(shapes)):
                    if shape[j] != 1:
                        cm[j][i] = 1

            utils.np_immutable(cm)
            return (cm, utils.np_hash(cm))

        cm = np.array(cm)
        utils.np_immutable(cm)

        # Explicit TPM with connectivity matrix: return.
        if shapes is None:
            return (cm, utils.np_hash(cm))

        # ImplicitTPM with connectivity matrix: validate against node shapes.
        validate.shapes(shapes, cm)

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

    @property
    def state_space(self):
        """tuple[tuple[Union[int, str]]]: Labels for the state space of each node.
        """
        return self._state_space

    @property
    def num_states(self):
        """int: The number of possible states of the network."""
        return np.prod(
            [len(node_states) for node_states in self._state_space]
        )

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
        return self._cm.shape[0]

    def __repr__(self):
        # TODO implement a cleaner repr, similar to analyses objects,
        # distinctions, etc.
        return "Network(\n{},\ncm={},\nnode_labels={},\nstate_space={}\n)".format(
            self.tpm, self.cm, self.node_labels, self.state_space._dict
        )

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

    # TODO(tpm): Immutability in xarray.
    def __hash__(self):
        return hash((hash(self.tpm), self._cm_hash))

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            "tpm": self.tpm,
            "cm": self.cm,
            "size": self.size,
            "node_labels": self.node_labels,
            "state_space": self.state_space,
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

    # TODO(4.0) use generator?
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
