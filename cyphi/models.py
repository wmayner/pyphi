# -*- coding: utf-8 -*-

"""
cyphi.models
~~~~~~~~~~~~

This module contains the primary objects that power CyPhi.

"""

# TODO Optimizations:
# - Memoization
# - Preallocation
# - Vectorization
# - Cythonize the hotspots
# - Use generators instead of list comprehensions where possible for memory
#   efficiency

import numpy as np
from . import utils
from .exceptions import ValidationException
from itertools import chain


# DEBUG = False
DEBUG = True


def pprint(debug_statement):
    """Conditional print for debugging."""
    if DEBUG:
        print(debug_statement)



class Network:

    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.

    """

    # TODO implement network definition via connectivity_matrix
    def __init__(self, tpm, current_state, past_state):
        """Generate and initialize a set of nodes based on a transition
        probability matrix.

        :param tpm: The network's transition probability matrix **in
            state-by-node form**, so that ``tpm[0][1][0]`` gives the
            probabilities of each node being on if the past state is
            :math:`\{0,1,0\}`. The shape of this TPM should thus be ``(number
            of states for each node) + (number of nodes)``.
        :type tpm: ``np.ndarray``
        :param state: An array describing the network's current state;
            ``state[i]`` gives the state of ``self.nodes[i]``
        :type state: ``np.ndarray``
        :param past_state: An array describing the network's past state;
            ``state[i]`` gives the past state of ``self.nodes[i]``
        :type state: ``np.ndarray``

        """
        # Validate the TPM
        if (tpm.shape[-1] is not len(tpm.shape) - 1):
            raise ValidationException(
                """Invalid TPM: There must be a dimension for each node, each
                one being the size of the corresponding node's state space,
                plus one dimension that is the same size as the network.""")
        # TODO extend to nonbinary nodes
        if (tpm.shape[:-1] != tuple([2] * tpm.shape[-1])):
            raise ValidationException(
                """Invalid TPM: We can only handle binary nodes at the moment.
                Each dimension except the last must be of size 2.""")

        self.tpm = tpm
        self.current_state = current_state
        self.past_state = past_state
        # Make these properties immutable (for hashing)
        self.tpm.flags.writeable = False
        self.current_state.flags.writeable = False
        self.past_state.flags.writeable = False
        # The number of nodes in the Network (TPM is in state-by-node form, so
        # number of nodes is given by the size of the last dimension)
        self.size = tpm.shape[-1]

        # Validate the state
        if (current_state.size is not self.size):
            raise ValidationException(
                "Invalid state: there must be one entry per node in the " +
                "network; this state has " + str(current_state.size) +
                "entries, " + "but there are " + str(self.size) + " nodes.")

        # Generate the the powerset of the node indices
        self.powerset = utils.powerset(np.arange(self.size))
        # Generate the nodes
        self.nodes = [Node(self, node_index)
                      for node_index in range(self.size)]
        # TODO extend to nonbinary nodes
        self.num_states = 2 ** self.size
        self.uniform_distribution = utils.uniform_distribution(self.size)

    def __repr__(self):
        return "Network(" + repr(self.tpm) + ")"

    def __str__(self):
        return "Network(" + str(self.tpm) + ")"

    def __eq__(self, other):
        """Two networks are equal if they have the same TPM, current state, and
        past state."""
        return (np.array_equal(self.tpm, other.tpm) and
                np.array_equal(self.current_state, other.current_state) and
                np.array_equal(self.past_state, other.past_state))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.tpm.tostring(), self.current_state.tostring(),
                    self.past_state.tostring()))


# TODO extend to nonbinary nodes
# TODO refactor to use purely indexes for nodes?
class Node:

    """A node in a network.

    Contains a TPM for just this node (indexed by network state). The TPM gives
    the probability that this node is on.

    For example, in a 3-node network, ``self.tpm[0][1][0]`` gives the
    probability that this node is on at :math:`t_0` if the state of the network
    is :math:`\{0,1,0\}` at :math:`t_{-1}`.

    """

    def __init__(self, network, index, label=None):
        """Initialize a Node.

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
        # The node's transition probability matrix (give probability that node
        # is on)
        # TODO extend to nonbinary nodes
        self.tpm = network.tpm[..., index]
        # Make the TPM immutable (for hashing)
        self.tpm.flags.writeable = False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (self.label if self.label is not None
                else 'n' + str(self.index))

    def __eq__(self, other):
        """Two nodes are equal if they belong to the same network and have the
        same index (``tpm`` must be the same in that case, so this method
        doesn't need to check ``tpm`` equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return self.network == other.network and self.index == other.index

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.network, self.index))


class Mechanism:

    """A set of nodes, considered as a single mechanism."""

    def __init__(self, nodes, current_state, past_state, network):
        """Initialize a Mechanism.

        :param nodes: The nodes in the mechanism
        :type nodes: ``[Node]``
        :param state: An array describing the state of the nodes in the
            mechanism
        :type state: ``np.ndarray``
        :param past_state: An array describing the past state of the nodes in
            the mechanism
        :type past_state: ``np.ndarray``
        :param network: The network the mechanism belongs to
        :type network: ``Network``

        """
        self.nodes = nodes
        self.current_state = current_state
        self.past_state = past_state
        # Make the state and past state immutable (for hashing)
        self.current_state.flags.writeable = False
        self.past_state.flags.writeable = False
        # The Mechanism belongs to the same network as its nodes
        self.network = network

    def __repr__(self):
        return "Mechanism(" + ", ".join([repr(self.nodes),
                                         repr(self.past_state),
                                         repr(self.current_state)]) + ")"

    def __str__(self):
        return "Mechanism(" + str(list(map(str, self.nodes))) + \
            ",\n\tCurrent state: " + str(self.current_state) + \
            ",\n\tPast state: " + str(self.past_state) + ")"

    def __eq__(self, other):
        """Two mechanisms are equal if they consist of the same set of nodes in
        the same state."""
        return (set(self.nodes) == set(other.nodes) and
                self.current_state == other.current_state and
                self.past_state == other.past_state)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((frozenset(self.nodes), self.current_state.tostring(),
                     self.past_state.tostring()))


class Subsystem:

    """A set of nodes in a network.

    Acts as a candidate set for cause/effect repertoire calculation.

    """

    def __init__(self, nodes, current_state, past_state, network):
        """Initialize a Subsystem.

        :param nodes: A list of nodes in this subsystem
        :type nodes: ``[Node]``
        :param current_state: The current state of this subsystem
        :type current_state: ``np.ndarray``
        :param past_state: The past state of this subsystem
        :type past_state: ``np.ndarray``
        :param network: The network the subsystem is part of
        :type network: ``Network``

        """
        self.nodes = nodes

        self.current_state = current_state
        self.past_state = past_state
        # Make the state and past state immutable (for hashing)
        self.current_state.flags.writeable = False
        self.past_state.flags.writeable = False

        self.network = network

        # Nodes outside the subsystem will be treated as fixed boundary
        # conditions in cause/effect repertoire calculations
        self.external_nodes = set(network.nodes) - set(nodes)

    def __repr__(self):
        return "Subsystem(" + ", ".join([repr(self.nodes),
                                         repr(self.past_state),
                                         repr(self.current_state)]) + ")"

    def __str__(self):
        return "Subsystem([" + str(list(map(str, self.nodes))) + "]" + \
            ", " + str(self.current_state) + ", " + str(self.past_state) + \
            ", " + str(self.network) + ")"

    def __eq__(self, other):
        """
        Two subsystems are equal if their sets of nodes, current and past
        states, and networks are equal.
        """
        return (set(self.nodes) == set(other.nodes) and self.past_state ==
                other.past_state and self.current_state == other.past_state and
                self.network == other.network)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((frozenset(self.nodes), self.current_state.tostring(),
                     self.past_state.tostring(), self.network))

    # TODO finish this? is it needed?
    def uc_past_repertoire(self):
        """Return the unconstrained past repertoire for this mechanism."""

    # TODO move to utils?
    def combine_cjd_with_external_max_ent(self, mechanism_cjd):
        """Return a cause or effect repertoire given a mechanism's CJD and a
        purview.

        Combines a conditional joint distribution for a mechanism (an
        intermediate result in the calculation of cause and effect repertoires)
        with the maximum entropy distribution for the nodes outside of that
        mechanism, resulting in a distribution (the cause or effect repertoire)
        over the entire network.

        :param mechanism_cjd: The mechanism's conditional joint distribution
        :type mechanism_cjd: ``np.ndarray``
        :param purview: The purview over which the cause or effect repertoire
            is being calculated.
        :type purview: ``Subsystem``
        """
        print("===================combine============")
        print("External nodes:", self.external_nodes)
        print("===================combine============")
        return np.multiply(
            utils.max_entropy_distribution(self.external_nodes, self.network),
            mechanism_cjd)

    def cause_repertoire(self, mechanism, purview):
        """Return the cause repertoire of this mechanism over the given
        purview.

        In the Matlab version's terminology,

        * *Cause repertoire* is "backward repertoire"
        * *Mechanism* is "numerator"
        * *Purview* is "denominator"
        * ``conditioned_tpm`` is ``next_num_node_distribution``
        * ``accumulated_cjd`` is ``numerator_conditional_joint``

        :param purview: The purview over which to calculate the cause
            repertoire
        :type purview: ``Subsystem``

        :returns: The cause repertoire of the mechanism over the given purview
        :rtype: ``np.ndarray``

        """
        # Return immediately if purview is empty
        if (len(purview) is 0):
            # TODO should this really be an empty array?
            return np.array([])

        # If the mechanism is empty, just return the maximum entropy
        # distribution over the purview
        if (len(mechanism) is 0):
            return utils.max_entropy_distribution(purview, self.network)

        # Preallocate the mechanism's conditional joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(tuple(2 if node in purview else 1
                                        for node in self.network.nodes))
        # Loop over all nodes in this mechanism, successively taking the
        # product (with expansion/broadcasting of singleton dimensions) of each
        # individual node's CPT (conditioned on that node's state) in order to
        # get the conditional joint distribution for the whole mechanism
        # (conditioned on the whole mechanism's state). After normalization,
        # this is the cause repertoire. Normalization happens after this loop.
        future_nodes = mechanism
        past_nodes = purview
        for node in future_nodes:
            # TODO extend to nonbinary nodes
            # We're conditioning on this node's state, so take the
            # probabilities that correspond to that state (The TPM subtracted
            # from 1 gives the probability that the node is off).
            conditioned_tpm = (node.tpm if self.current_state[node.index] == 1
                               else 1 - node.tpm)
            # Marginalize-out the nodes with inputs to this mechanism that
            # aren't in the given purview
            # TODO explicit inputs to nodes (right now each node is implicitly
            # connected to all other nodes, since initializing a Network with a
            # connectivity matrix isn't implemented yet)
            for non_purview_input in set(self.network.nodes) - set(past_nodes):
                                      # TODO add this when inputs are
                                      # implemented:
                                      # and node in self.input_nodes):
                # If the non-purview input node is part of the candidate
                # system, we marginalize it out of the current node's CPT.
                if non_purview_input in self.nodes:
                    conditioned_tpm = utils.marginalize_out(non_purview_input,
                                                            conditioned_tpm)
                # Now we condition the CPT on the past states of nodes outside
                # the candidate system, which we treat as fixed boundary
                # conditions. We collapse the dimensions corresponding to the
                # fixed nodes so they contain only the probabilities that
                # correspond to their past states.
                elif non_purview_input not in self.nodes:
                    past_conditioning_indices = \
                        [slice(None)] * self.network.size
                    past_conditioning_indices[non_purview_input.index] = \
                        [self.past_state[non_purview_input.index]]
                    conditioned_tpm = \
                        conditioned_tpm[past_conditioning_indices]
            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product (with singleton
            # broadcasting)
            accumulated_cjd = np.multiply(accumulated_cjd,
                                          conditioned_tpm)

        # Finally, normalize by the marginal probability of the past state to
        # get the mechanism's CJD
        accumulated_cjd_sum = np.sum(accumulated_cjd)
        # Don't divide by zero
        if accumulated_cjd_sum != 0:
            accumulated_cjd = (np.divide(accumulated_cjd,
                                         np.sum(accumulated_cjd)))

        # Note that we're not returning a distribution over all the nodes in
        # the network, only a distribution over the nodes in the purview. This
        # is because we never actually need to compare proper cause/effect
        # repertoires, which are distributions over the whole network; we need
        # only compare the purview-repertoires with each other, since cut vs.
        # whole comparisons are only ever done over the same purview.
        return accumulated_cjd

    def effect_repertoire(self, mechanism, purview):
        """Return the effect repertoire of this mechanism over the given
        purview.

        In the Matlab version's terminology,

        * *Effect repertoire* is "forward repertoire"
        * *Mechanism* is "numerator"
        * *Purview* is "denominator"
        * ``conditioned_tpm`` is ``next_denom_node_distribution``
        * ``accumulated_cjd`` is ``denom_conditional_joint``

        :param purview: The purview over which to calculate the effect
            repertoire
        :type purview: ``Subsystem``

        :returns: The effect repertoire of the mechanism over the given purview
        :rtype: ``np.ndarray``

        """
        # Preallocate the purview's joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(tuple([1] * self.network.size +
                                        [2 if node in purview else 1
                                         for node in self.network.nodes]))
        # Loop over all nodes in the purview, successively taking the product
        # (with 'expansion'/'broadcasting' of singleton dimensions) of each
        # individual node's TPM in order to get the joint distribution for the
        # whole purview. After conditioning on the mechanism's state and that
        # of external nodes, this will be the effect repertoire as a
        # distribution over the purview.
        future_nodes = purview
        past_nodes = mechanism
        for node in future_nodes:
            # Unlike in calculating the cause repertoire, here the TPM is not
            # conditioned yet. `tpm` is an array with twice as many dimensions
            # as the network has nodes. For example, in a network with three
            # nodes {n0, n1, n2}, the CPT for node n1 would have shape
            # (2,2,2,1,2,1). The CPT for the node being off would be given by
            # `tpm[:,:,:,0,0,0]`, and the CPT for the node being on would be
            # given by `tpm[:,:,:,0,1,0]`. The second half of the shape is for
            # indexing based on the current node's state, and the first half of
            # the shape is the CPT indexed by network state, so that the
            # overall CPT can be broadcast over the `accumulated_cjd` and then
            # later conditioned by indexing.

            # TODO extend to nonbinary nodes
            # Allocate the TPM
            tpm = np.zeros([2] * self.network.size +
                           [2 if i is node.index else 1 for i in
                            range(self.network.size)])
            tpm_off_indices = [slice(None)] * self.network.size + \
                [0] * self.network.size
            # Insert the TPM for the node being off
            tpm[tpm_off_indices] = 1 - node.tpm
            # Insert the TPM for the node being on
            tpm_on_indices = [slice(None)] * self.network.size + \
                [1 if i == node.index else 0 for i in
                 range(self.network.size)]
            tpm[tpm_on_indices] = node.tpm

            # Marginalize-out the subsystem nodes with inputs to the purview
            # that aren't in the mechanism
            # TODO explicit inputs to nodes (right now each node is implicitly
            # connected to all other nodes, since initializing a Network with a
            # connectivity matrix isn't implemented yet)
            for non_past_input in set(self.nodes) - set(past_nodes):
                                   # TODO add this when inputs are implemented:
                                   # and node in self.input_nodes):
                tpm = utils.marginalize_out(non_past_input, tpm)

            # Incorporate this node's CPT into the future_nodes' conditional
            # joint distribution by taking the product (with singleton
            # broadcasting)
            accumulated_cjd = np.multiply(accumulated_cjd, tpm)

        # Now we condition on the state of the past nodes and the external
        # nodes (by collapsing the CJD onto those states).

        # Initialize the conditioning indices, taking the slices as singleton
        # lists-of-lists for later flattening with `chain`.
        conditioning_indices = [[slice(None)]] * self.network.size
        for node in set(past_nodes) | set(self.external_nodes):
            # Preserve singleton dimensions with `np.newaxis`
            conditioning_indices[node.index] = [self.current_state[node.index],
                                                np.newaxis]
        # Flatten the indices
        conditioning_indices = list(chain.from_iterable(conditioning_indices))

        # Obtain the actual conditioned distribution by indexing with the
        # conditioning indices
        accumulated_cjd = accumulated_cjd[tuple(conditioning_indices)]
        # The distribution still has twice as many dimensions as the network
        # has nodes, with the first half of the shape now all singleton
        # dimesnions, so we reshape to eliminate those singleton dimensions
        # (the second half of the shape may also contain singleton dimensions,
        # depending on how many nodes are in the purview).
        accumulated_cjd = accumulated_cjd.reshape(
            accumulated_cjd.shape[self.network.size:2 * self.network.size])

        # Note that we're not returning a distribution over all the nodes in
        # the network, only a distribution over the nodes in the purview. This
        # is because we never actually need to compare proper cause/effect
        # repertoires, which are distributions over the whole network; we need
        # only compare the purview-repertoires with each other, since cut vs.
        # whole comparisons are only ever done over the same purview.
        return accumulated_cjd
