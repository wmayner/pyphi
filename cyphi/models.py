# -*- coding: utf-8 -*-

"""
cyphi.models
~~~~~~~~~~~~

This module contains the primary objects that power CyPhi.

"""

# TODO Optimizations:
# - Preallocation
# - Vectorization
# - Memoization

import numpy as np
from . import utils
from .exceptions import ValidationException


class Network(object):
    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.

    """
    # TODO implement network definition via connectivity_matrix
    def __init__(self, tpm, state):
        """
        Generates and initializes a set of nodes based on a connectivity matrix
        and a transition probability matrix.

        :param tpm: The network's transition probability matrix **in
            state-by-node form**
        :type tpm: ``np.ndarray``
        :param state: An array describing the network's initial state;
            ``state[i]`` gives the state of ``self.nodes[i]``
        :type state: ``np.ndarray``
        """

        # Validate the TPM
        if (tpm.ndim is not 2):
            raise ValidationException(
                "Invalid TPM: A Network's TPM must be in state-by-node form " +
                "and thus 2-dimensional; this TPM is " + str(tpm.ndim) +
                "-dimensional.")
        # TODO extend this hard-coded value to handle more than binary nodes
        if (tpm.size is not (2 ** tpm.shape[1]) * tpm.shape[1]):
            raise ValidationException(
                "Invalid TPM: There must be " + str(2 ** tpm.shape[1]) +
                "[ 2^(number of nodes) * (number of nodes) ] elements in the " +
                " TPM;" + str(tpm.size) + "were given.")

        self.tpm = tpm
        self.state = state
        # The number of nodes in the Network (TPM is in state-by-node form, so
        # number of nodes is given by second dimension)
        self.size = tpm.shape[1]

        # Validate the state
        if (state.size is not self.size):
            raise ValidationException(
                "Invalid state: there must be one entry per node in the " +
                "network; this state has " + str(state.size) + " entries, " +
                "but there are " + self.size + " nodes.")

        # Generate the the powerset of the node indices
        self.powerset = utils.powerset(np.arange(self.size))
        # Generate the nodes
        self.nodes = [Node(self, node_index)
                      for node_index in range(self.size)]
        # TODO handle nonbinary nodes
        self.num_states = 2 ** self.size
        self.uniform_distribution = utils.uniform_distribution(self.size)

    def __repr__(self):
        return "Network(" + repr(self.tpm) + ")"

    def __str__(self):
        return "Network with TPM:\n" + str(self.tpm)

    def __eq__(self, other):
        """
        Two Networks are equal if they have the same TPM.
        """
        return np.array_equal(self.tpm, other.tpm)

    def __ne__(self, other):
        return not self.__eq__(other)


# TODO extend to more complex types of nodes
class Node:
    """A node in a network.

    Contains a TPM for just this node (indexed by network state). The TPM gives
    the probability that this node is on.

    For example, in a 3-node network, ``self.tpm[0][1][0]`` gives the
    probability that this node is on at :math:`t_0` if the state of the network
    is :math:`\{0,1,0\}` at :math:`t_{-1}`.
    """
    def __init__(self, network, index, label=None):
        """
        :param network: The network this node belongs to
        :type network: ``Network``
        :param index: The index of this node in the network's list of nodes
        :type index: ``int``
        :param label: The label for this node, for display purposes. Optional;
            defaults to ``None``.
        :type label: ``str``
        """
        # This node's parent network
        self.network = network
        # This node's index in the network's list of nodes
        self.index = index
        # Label for display
        self.label = label
        # The TPM for just this node
        # (Grab this node's column in the network TPM and reshape it to be
        # indexed by state; i.e., there is a dimension for each node, the size
        # of which is the number of that node's states)
        # TODO number of node states is hardcoded for 2 here; at some point we
        # want to extend to nonbinary nodes
        self.tpm = network.tpm[..., index].reshape([2] * network.size)

    def __repr__(self):
        return "Node(" + ", ".join([repr(self.network),
                                    repr(self.index),
                                    repr(self.label)]) + ")"

    def __str__(self):
        return (self.label if self.label is not None
                else 'n' + str(self.index))

    def __eq__(self, other):
        """
        Two nodes are equal if they belong to the same network and have the
        same index (``tpm`` must be the same in that case, so this method
        doesn't need to check ``tpm`` equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return self.network == other.network and self.index == other.index

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int(str(self.network.__hash__()) + str(self.index))


class Mechanism:
    """A set of nodes, considered as a single mechanism."""
    def __init__(self, nodes, state, tpm=None, MIP=None):
        """
        :param nodes: The nodes in the mechanism
        :type nodes: ``[Node]``
        :param state: The state of the nodes in the mechanism. A dictionary
            where the keys are node indices and the values are node states.
        :type state: ``dict``
        :param tpm: The transition probability matrix for this mechanism
        :type tpm: ``np.ndarray``
        :param MIP: The minimum information partition for this mechanism
        :type MIP:
        """
        self.nodes = nodes
        # The Mechanism belongs to the same network as its nodes
        self.network = nodes[0].network
        self.tpm = tpm
        self.MIP = MIP

    def __eq__(self, other):
        """
        Two mechanisms are equal if they consist of the same set of nodes, in
        the same state.
        """
        return (set(self.nodes) == set(other.nodes) and
                self.state == other.state)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int(str(set(self.nodes).__hash__()) +
                   str(self.state.__hash__()))

    def uc_past_repertoire(self):
        """
        Return the unconstrained past repertoire for this mechanism.
        """

    # TODO calculate unconstrained repertoires here, or in cyphi.compute?
    def cause_repertoire(self, purview):
        """
        Return the cause repertoire of this mechanism over the given purview.

        In the Matlab version's terminology,

        * *Cause repertoire* is "backward repertoire" or "perspective"
        * *Mechanism* is "numerator"
        * *Purview* is "denominator"
        * ``conditioned_tpm`` is ``next_num_node_distribution``

        :param purview: The purview over which to calculate the cause
            repertoire
        :type purview: ``Purview``

        :returns: The cause repertoire of the mechanism over
            the given purview
        :rtype: ``np.ndarray``

        """
        # Return immediately if purview is empty
        if (purview.nodes.size is 0):
            # TODO should this really be None?
            return None

        # If the mechanism is empty, just return the maximum entropy
        # distribution over the purview
        if (self.nodes.size is 0):
            return purview.max_entropy_distribution()

        # Preallocate the mechanism's conditional joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(tuple(2 if node in self.purview.nodes else 1
                                        for node in self.network.nodes))
        # Loop over all nodes in this mechanism, successively taking the
        # product product (with expansion/broadcasting of singleton dimensions)
        # of each individual node's CPT (conditioned on that node's state) in
        # order to get the conditional joint distribution for the whole
        # mechanism (conditioned on the whole mechanism's state). This is the
        # cause repertoire. Normalization of the distribution happens once,
        # after this loop.
        for node in self.nodes:
            # Collapse the dimensions of nodes with fixed states, i.e., the
            # current node and all external nodes (which are treated as fixed
            # boundary conditions)
            conditioned_tpm = node.tpm[self._conditioning_indicies(node)]

            # Marginalize-out the nodes with inputs to this mechanism that
            # aren't in the given purview
            # TODO explicit inputs to nodes (right now each node is implicitly
            # connected to all other nodes, since initializing a Network with a
            # connectivity matrix isn't implemented yet)
            for non_marginal_node in (node for node in self.network.nodes if
                                      node not in self.purview.nodes):
                                      # and node in self.input_nodes):
                conditioned_tpm = self._marginalize_out(non_marginal_node,
                                                        conditioned_tpm)

            # TODO [***] (3-stars means low priority): refactor states to not
            # be indices?

            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product with singleton
            # broadcasting
            accumulated_cjd = np.multiply(accumulated_cjd,
                                          conditioned_tpm)

        # Finally, normalize the distribution to get the true cause repertoire
        return np.divide(accumulated_cjd, np.sum(accumulated_cjd))

        def _marginalize_out(non_marginal_node, conditioned_tpm):
            """Marginalize out a non-marginal node from a conditioned TPM."""
            return np.divide(
                np.sum(conditioned_tpm, non_marginal_node.index),
                conditioned_tpm.shape[non_marginal_node.index])

        def _conditioning_indicies(current_node):
            """
            Return the indices that collapse the dimensions of the current
            node's TPM so that only entries corresponding to fixed states---the
            state of the current node and the states of external nodes---are
            left. When sliced with the returned indices, the current node's TPM
            becomes an intermediate result used in getting the cause
            repertoire; it still needs to be modified by marginalizing out the
            nodes internal to this mechanism but not in the purview, and
            finally by normalizing the whole table.
            """
            # To generate the indices for the CPT for this node we begin by
            # taking all the entries in its TPM (all states of all nodes)...
            conditioning_indicies = [slice(None)] * self.network.size
            # ...except those that don't correspond to the current state of the
            # node in question. States are indices, so we just set the CPT
            # index for that node to its state.
            conditioning_indicies[current_node.index] = \
                self.state[current_node.index]

            # Then we do the same thing for nodes outside this mechanism, which
            # are treated as fixed boundary conditions (so we take only TPM
            # entries corresponding to their past state)
            # TODO vectorize this?
            # TODO unidircut stuff
            for external_node in (node for node in self.network.nodes if
                                  node not in purview.nodes and
                                  node not in self.nodes):
                # TODO [*] (high priority) do we need a different past state?
                past_state = self.network.state
                conditioning_indicies[external_node.index] = \
                    past_state[external_node.index]

            return conditioning_indicies


class Purview:
    """A set of nodes, considered as a purview."""
    def __init__(self, nodes):
        self.network = nodes[0].network
        self.nodes = nodes

    # TODO decorate this with a memoizer
    def max_entropy_distribution(self):
        """
        Get the maximum entropy distribution over this purview (this is
        different from the network's uniform distribution because nodes outside
        the purview are fixed and treated as if they have only 1 state).
        """
        # TODO extend to nonbinary nodes
        max_ent_shape = [2 if node in self.nodes else 1
                         for node in self.network.nodes]
        return np.divide(np.ones(max_ent_shape),
                         np.ufunc.reduce(np.multiply, max_ent_shape))


class Distribution:
    """A wrapper object for a probability distribution.

    **Indexing rules:**

    When the distribution is over the state-space of a set of nodes,
    ``data[i]`` gives the probability of the state represented by the binary
    representation of ``i``.

    For example, for a ``Distribution`` over the state space of a set of nodes
    :math:`\{n_0, n_1, n_2\}`, the probability of the state :math:`\{0, 1, 0\}`
    is given by ``Distribution.data[2]``, since :math:`2_{10} = 010_2`.

    **NOTE:** This is enforced **by convention only**---it is the
    responsibility of the creator of the ``Distribution`` to abide by the
    convention.

    """
    def __init__(self, data):
        """
        :param data: a probability distribution (must sum to 1)
        :param type: ``np.ndarray``
        """
        # Ensure ``data`` represents a probability distribution
        if np.sum(data) != 1.0:
            raise ValidationException(
                "Probabilities in a distribution must sum to 1.")
        self.data = data
