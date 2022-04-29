#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# network.py

"""
Represents the network of interest. This is the primary object of PyPhi and the
context of all |small_phi| and |big_phi| computation.
"""

import numpy as np

from . import cache, config, connectivity, convert, jsonify, utils, validate, node
from .labels import NodeLabels
from .tpm import is_state_by_state
from .__tpm import TPM, SbN


class Network:
    """A network of nodes.

    Represents the network under analysis and holds auxilary data about it.

    Args:
        tpm (np.ndarray): The transition probability matrix of the network.

            The TPM can be provided in any of three forms: **state-by-state**,
            **state-by-node**, or **multidimensional state-by-node** form.
            In the state-by-node forms, row indices must follow the
            little-endian convention (see :ref:`little-endian-convention`). In
            state-by-state form, column indices must also follow the
            little-endian convention.

            If the TPM is given in state-by-node form, it can be either
            2-dimensional, so that ``tpm[i]`` gives the probabilities of each
            node being ON if the previous state is encoded by |i| according to
            the little-endian convention, or in multidimensional form, so that
            ``tpm[(0, 0, 1)]`` gives the probabilities of each node being ON if
            the previous state is |N_0 = 0, N_1 = 0, N_2 = 1|.

            The shape of the 2-dimensional form of a state-by-node TPM must be
            ``(s, n)``, and the shape of the multidimensional form of the TPM
            must be ``[2] * n + [n]``, where ``s`` is the number of states and
            ``n`` is the number of nodes in the network.

    Keyword Args:
        cm (np.ndarray): A square binary adjacency matrix indicating the
            connections between nodes in the network. ``cm[i][j] == 1`` means
            that node |i| is connected to node |j| (see :ref:`cm-conventions`).
            **If no connectivity matrix is given, PyPhi assumes that every node
            is connected to every node (including itself)**.
        p_nodes (list[str]): Human-readable list of names of nodes at time |t-1|
        p_states (list[int]): List of the number of states of each node at time |t-1| (necessary for defining a multi-valued tpm)
        n_nodes (list[str]): Human-readable list of names of nodes at time |t|
        n_states (list[int]): List of the number of states of each node at time |t| (necessary for defining a multi-valued tpm)

    Example:
        In a 3-node network, ``the_network.tpm[(0, 0, 1)]`` gives the
        transition probabilities for each node at |t| given that state at |t-1|
        was |N_0 = 0, N_1 = 0, N_2 = 1|.
    """

    # TODO make tpm also optional when implementing logical network definition
    # TODO node_labels attribute deprecated, but many tests use it so currently keeping it an option
    def __init__(self, tpm, cm=None, p_nodes=None, p_states=None, n_nodes=None, n_states=None, purview_cache=None, node_labels=None):
        if isinstance(tpm, list): 
            self._is_list = True
            if node_labels:
                p_nodes = node_labels
            self._tpm = tpm
            self._cm = self._build_cm_from_list(cm)
            self._node_indices = tuple(range(len(tpm)))
            # User can specify names of each node to be lined up with the list of node TPMs, else default labels are generated
            self._node_labels = NodeLabels(p_nodes, self._node_indices)
            
        else:
            self._is_list = False
            if node_labels:
                p_nodes = node_labels
            if p_states or n_states: # Requires NB state-by-state, could just do only TPM and convert to SbN later?
                self._tpm = TPM(tpm, p_nodes, p_states, n_nodes, n_states)
                #validate.tpm(tpm, check_independence=config.VALIDATE_CONDITIONAL_INDEPENDENCE)
            else:
                self._tpm = SbN(tpm, p_nodes, p_states, n_nodes, n_states)

            self._cm = self._tpm.infer_cm()
            # Convert to list
            tpm_list = [self._tpm.create_node_tpm(index, self._cm) for index in range(len(self._tpm.n_nodes))]

            self._node_indices = tuple(range(self.size))
            self._node_labels = NodeLabels(self._tpm._p_nodes, self.node_indices) # TODO consider using self._tpm._p_nodes instead for more readability?

            self._tpm = tpm_list
            self._is_list = True
        
        self.purview_cache = purview_cache or cache.PurviewCache()
         
        validate.network(self)

    @property
    def tpm(self):
        """np.ndarray: The network's transition probability matrix, in
        multidimensional form.
        """
        return self._tpm

    # TODO Deprecated?
    @staticmethod
    def _build_tpm(tpm):
        """Validate the TPM passed by the user and convert to multidimensional
        form.
        """
        tpm = np.array(tpm)

        validate.tpm(tpm, check_independence=config.VALIDATE_CONDITIONAL_INDEPENDENCE)

        # Convert to multidimensional state-by-node form
        if is_state_by_state(tpm):
            tpm = convert.state_by_state2state_by_node(tpm)
        else:
            tpm = convert.to_multidimensional(tpm)

        utils.np_immutable(tpm)

        return (tpm, utils.np_hash(tpm))

    def _build_cm_from_list(self, cm):
        """Generate the connectivity matrix for a network whose tpm is defined as a 
        list of Node TPMs, by concatenating the individual Node TPM cms.

        Args:
            tpm_list (list[TPM]): List of Node TPMs that define how the network transitions.
        """
        if cm is None:
            cm_list = [TPM.infer_node_cm(node_tpm) for node_tpm in self._tpm] 
            return np.concatenate(cm_list, axis=1)
        else:
            return np.array(cm)
        # return np.ones((self.size, self.size))

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
            # Build cm from TPM method
            cm = self._tpm.infer_cm()
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
        # If list, states can be counted as product of possible transitions of each node
        # If not, use TPM.num_states?
        # if self._is_list:
        num = 1
        for tpm in self._tpm:
            num *= tpm.shape[-1]
        return num
        # else:
        #    return self._tpm.num_states

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
        if self._is_list:
            return len(self._tpm)
        elif isinstance(self.tpm, TPM):
            # TODO Assumes symmetry for now
            return len(self.tpm.p_nodes)
        else:
            return self.tpm.shape[-1]

    def __repr__(self):
        if self._is_list:
            return str([tpm for tpm in self._tpm]) #TODO Consider representations
        else:
            return "Network({}, cm={})".format(self.tpm, self.cm)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this network equals the other object.

        Networks are equal if they have the same TPM and CM.
        """
        if self._is_list:
            return np.all([node_tpm == node_tpm for node_tpm in self._tpm])
        return (
            isinstance(other, Network)
            and np.array_equal(self.tpm, other.tpm)
            and np.array_equal(self.cm, other.cm)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self._tpm_hash, self._cm_hash))

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
