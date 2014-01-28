# -*- coding: utf-8 -*-

"""
cyphi.models
~~~~~~~~~~~~

This module contains the primary objects that power CyPhi.

"""

# TODO use structured arrays to represent sets of nodes, so labels
# are explicit rather than implicit (indices)?

import numpy as np
from . import utils
from .exceptions import ValidationException


class Network(object):
    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.

    """
    # TODO implement network definition via connectivity_matrix
    def __init__(self, tpm):
        """
        Generates and initializes a set of nodes based on a connectivity matrix
        and a transition probability matrix.

        :param tpm: The network's transition probability matrix **in
            state-by-node form**
        :type tpm: ``np.ndarray``
        """

        # Validate the TPM
        if (tpm.ndim is not 2):
            raise ValidationException(
                "Invalid TPM: A Network's TPM must be 2-dimensional; " +
                "this TPM is " + str(tpm.ndim) + "-dimensional.")
        # TODO extend this hard-coded value to handle more than binary nodes
        if (tpm.size is not (2 ** tpm.shape[1]) * tpm.shape[1]):
            raise ValidationException(
                "Invalid TPM: There must be [ 2^(number of nodes) * " +
                "(number of nodes) ] elements in the TPM.")

        self.tpm = tpm
        # The number of nodes in the Network (TPM is in state-by-node form, so
        # number of nodes is given by second dimension)
        self.size = tpm.shape[1]
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

        # TODO number of node states is hardcoded for 2 here; at some point we
        # want to allow for an arbitrary number of states

        # The TPM for just this node
        # (Grab this node's column in the network TPM and reshape it to be
        # indexed by state)
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
        :param nodes: The indices of the nodes in the mechanism
        :type nodes: ``np.ndarray``
        :param state: A dictionary mapping node indices to node states
        :type state: ``dict``
        :param tpm: The TPM for this mechanism
        :type tpm: ``np.ndarray``
        :param MIP:
        :type MIP:
        """
        self.nodes = nodes
        # The Mechanism belongs to the same network as its nodes
        self.network = nodes[0].network
        self.state = state
        # The transition probability matrix for this mechanism
        self.tpm = tpm
        # The minimum information partition
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

        :param purview: The purview over which to calculate the cause
            repertoire
        :type purview: ``Purview``

        :returns: The cause repertoire of the mechanism over
            the given purview
        :rtype: ``Distribution``

        """
        # Return immediately if purview is empty
        if (purview.nodes.size is 0):
            return None

        # If the mechanism is empty, just return the maximum entropy
        # distribution over the purview
        if (self.nodes.size is 0):
            return purview.max_entropy_distribution()

        pass


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
