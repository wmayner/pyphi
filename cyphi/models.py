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


# DEBUG = False
DEBUG = True


def pprint(debug_statement):
    """Conditional print for debugging"""
    if DEBUG:
        print(debug_statement)


class Network(object):
    """A network of elements.

    Represents the network we're analyzing and holds auxilary data about it.
    """
    # TODO implement network definition via connectivity_matrix
    def __init__(self, tpm, state, past_state):
        """
        Generates and initializes a set of nodes based on a transition
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
        self.state = state
        self.past_state = past_state
        # Make these properties immutable (for hashing)
        self.tpm.flags.writeable = False
        self.state.flags.writeable = False
        self.past_state.flags.writeable = False
        # The number of nodes in the Network (TPM is in state-by-node form, so
        # number of nodes is given by the size of the last dimension)
        self.size = tpm.shape[-1]

        # Validate the state
        if (state.size is not self.size):
            raise ValidationException(
                "Invalid state: there must be one entry per node in the " +
                "network; this state has " + str(state.size) + " entries, " +
                "but there are " + str(self.size) + " nodes.")

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
        """
        Two Networks are equal if they have the same TPM, current state, and
        past state.
        """
        return (np.array_equal(self.tpm, other.tpm) and
                np.array_equal(self.state, other.state) and
                np.array_equal(self.past_state, other.past_state))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self, other):
        return hash(self.tpm.tostring() + self.state.tostring() +
                    self.past_state.tostring())


# TODO extend to nonbinary nodes
class Node:
    """
    A node in a network.

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
        self.tpm = network.tpm[..., index]
        # Make the TPM immutable (for hashing)
        self.tpm.flags.writeable = False

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
        return int(str(hash(self.network)) + str(hash(self.index)))


class Mechanism:
    """A set of nodes, considered as a single mechanism."""
    def __init__(self, nodes, state, past_state, network):
        """
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
        self.state = state
        self.past_state = past_state
        # Make the state and past state immutable (for hashing)
        self.state.flags.writeable = False
        self.past_state.flags.writeable = False
        # The Mechanism belongs to the same network as its nodes
        self.network = network

    def __eq__(self, other):
        """
        Two mechanisms are equal if they consist of the same set of nodes in
        the same state.
        """
        return (set(self.nodes) == set(other.nodes) and
                self.state == other.state and
                self.past_state == other.past_state)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int(str(hash(frozenset(self.nodes))) + self.state.tostring +
                   self.past_state.tostring)

    def uc_past_repertoire(self):
        """
        Return the unconstrained past repertoire for this mechanism.
        """

    # Cause repertoire helpers
    #======================================================================

    # TODO move to utils?
    @staticmethod
    def _marginalize_out(node, tpm):
        """
        Marginalize out a node from a TPM.

        The TPM must be indexed by individual node state.

        :param node: The node to be marginalized out
        :type node: ``Node``
        :param tpm: The tpm to marginalize the node out of
        :type tpm: ``np.ndarray``

        :returns: The TPM after marginalizing out the node
        :rtype: ``np.ndarray``
        """
        # Preserve number of dimensions so node indices still index into
        # the proper axis of the returned distribution
        prenormalized = np.expand_dims(np.sum(tpm, node.index), node.index)
        # Normalize the distribution by number of states
        return np.divide(prenormalized, tpm.shape[node.index])

    #======================================================================

    def cause_repertoire(self, purview):
        """
        Return the cause repertoire of this mechanism over the given purview.

        In the Matlab version's terminology,

        * *Cause repertoire* is "backward repertoire"
        * *Mechanism* is "numerator"
        * *Purview* is "denominator"
        * ``conditioned_tpm`` is ``next_num_node_distribution``

        :param purview: The purview over which to calculate the cause
            repertoire
        :type purview: ``Subsystem``

        :returns: The cause repertoire of the mechanism over the given purview
        :rtype: ``np.ndarray``
        """
        # Return immediately if purview is empty
        if (len(purview.nodes) is 0):
            # TODO should this really be None?
            return None

        # If the mechanism is empty, just return the maximum entropy
        # distribution over the purview
        if (len(self.nodes) is 0):
            return purview.max_entropy_distribution()

        # Preallocate the mechanism's conditional joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(tuple(2 if node in purview.nodes else 1
                                        for node in self.network.nodes))

        pprint('accumulated_cjd initial shape:')
        pprint(accumulated_cjd.shape)
        pprint('current mechanism:')
        pprint(list(map(str, self.nodes)))
        pprint('purview:')
        pprint(str(purview))

        # Loop over all nodes in this mechanism, successively taking the
        # product (with expansion/broadcasting of singleton dimensions) of each
        # individual node's CPT (conditioned on that node's state) in order to
        # get the conditional joint distribution for the whole mechanism
        # (conditioned on the whole mechanism's state). After normalization,
        # this is the cause repertoire. Normalization happens after this loop.
        for node in self.nodes:
            pprint('  current node:')
            pprint(str(node))
            pprint('  current node tpm:')
            pprint(node.tpm)
            pprint('  current node tpm shape')
            pprint(node.tpm.shape)

            # TODO extend to nonbinary nodes
            # We're conditioning on this node's state, so take the
            # probabilities that correspond to that state (The TPM subtracted
            # from 1 gives the probability that the node is off).
            conditioned_tpm = (node.tpm if self.state[node.index] == 1 else
                               1 - node.tpm)

            pprint("  initial conditioned tpm shape:")
            pprint(conditioned_tpm.shape)
            pprint("  initial conditioned tpm:")
            pprint(conditioned_tpm)
            pprint("  Looping over external nodes…")
            pprint(str(list(map(str, (node for node in self.network.nodes if
                                      node not in purview.nodes)))))

            # Marginalize-out the nodes with inputs to this mechanism that
            # aren't in the given purview
            # TODO explicit inputs to nodes (right now each node is implicitly
            # connected to all other nodes, since initializing a Network with a
            # connectivity matrix isn't implemented yet)
            for non_purview_input in (node for node in self.network.nodes if
                                      node not in purview.nodes):
                                      # TODO add this when inputs are
                                      # implemented:
                                      # and node in self.input_nodes):
                # If the non-purview input node is part of this mechanism, we
                # marginalize it out of the current node's CPT.
                if non_purview_input in self.nodes:
                    pprint("-----------------------------------------------")
                    pprint('    external_node:')
                    pprint(str(non_purview_input))
                    pprint('    conditioned_tpm pre-marginalize:')
                    pprint(str(conditioned_tpm))
                    pprint('    shape:')
                    pprint(conditioned_tpm.shape)
                    pprint('    ##### MARGINALIZING ######')
                    conditioned_tpm = self._marginalize_out(non_purview_input,
                                                            conditioned_tpm)
                    pprint('    conditioned_tpm post-marginalize:')
                    pprint(str(conditioned_tpm))
                    pprint('    shape:')
                    pprint(conditioned_tpm.shape)
                # Now we condition the CPT on the past states of nodes outside
                # this mechanism, which we treat as fixed boundary conditions.
                # We collapse the dimensions corresponding to the fixed nodes
                # so they contain only the probabilities that correspond to
                # their past states.
                elif non_purview_input not in self.nodes:
                    past_conditioning_indices = \
                        [slice(None)] * self.network.size
                    past_conditioning_indices[non_purview_input.index] = \
                        [self.past_state[non_purview_input.index]]
                    conditioned_tpm = \
                        conditioned_tpm[past_conditioning_indices]

            pprint("=========================================================")

            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product (with singleton
            # broadcasting)
            pprint("    Multiplying distributions with broadcasting…")
            pprint('    accumulated_cjd:')
            pprint(str(accumulated_cjd))
            pprint('    conditioned_tpm:')
            pprint(str(conditioned_tpm))
            accumulated_cjd = np.multiply(accumulated_cjd,
                                          conditioned_tpm)

        # Finally, normalize by the marginal probability of the past state to
        # get the mechanism's CJD
        accumulated_cjd_sum = np.sum(accumulated_cjd)
        # Don't divide by zero
        if accumulated_cjd_sum != 0:
            mechanism_cjd = (np.divide(accumulated_cjd,
                                       np.sum(accumulated_cjd)))
        else:
            mechanism_cjd = accumulated_cjd

        pprint("mechanism_cjd:")
        pprint(mechanism_cjd)
        pprint(mechanism_cjd.shape)

        # Now we need to combine the mechanism's CJD with the maximum entropy
        # distribution for the non-mechanism nodes. The resulting product will
        # gives the actual cause repertoire, a distribution over all nodes in
        # the network.
        external_nodes = Subsystem([node for node in self.network.nodes if
                                    node not in self.nodes and
                                    node not in purview.nodes],
                                   self.network)

        pprint(str(external_nodes))
        pprint(str(self.nodes))
        pprint(str(list(map(str, purview.nodes))))

        cause_repertoire = np.multiply(
            external_nodes.max_entropy_distribution(),
            mechanism_cjd)

        pprint("external nodes maxent dist")
        pprint(external_nodes.max_entropy_distribution())
        pprint(external_nodes.max_entropy_distribution().shape)

        pprint("RETURNING:")
        pprint(cause_repertoire)
        pprint(cause_repertoire.shape)
        return cause_repertoire

    def effect_repertoire(self, purview):
        # Switch roles of mechanism and purview
        mechanism = Mechanism(purview.nodes, self.state, self.past_state,
                              self.network)
        cr = mechanism.cause_repertoire(Subsystem(self.nodes, self.network))
        return cr


class Subsystem:
    """A subset of nodes in a network."""
    def __init__(self, nodes, network):
        """
        :param nodes: A list of nodes in this subsystem
        :type nodes: ``[Node]``
        """
        self.nodes = nodes
        self.network = network

    def __repr__(self):
        return "Subsystem(" + repr(self.nodes) + ")"

    def __str__(self):
        return "Subsystem(" + str(list(map(str, self.nodes))) + ")"

    def __eq__(self, other):
        """Two subsystems are equal if their sets of nodes are equal."""
        return (set(self.nodes) == set(other.nodes))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(frozenset(self.nodes))

    # TODO memoize this
    def max_entropy_distribution(self):
        """
        Return the maximum entropy distribution over this subsystem.

        This is different from the network's uniform distribution because nodes
        outside the are fixed and treated as if they have only 1 state.

        :returns: The maximum entropy distribution over this subsystem
        :rtype: ``np.ndarray``
        """
        # TODO extend to nonbinary nodes
        max_ent_shape = [2 if node in self.nodes else 1
                         for node in self.network.nodes]
        return np.divide(np.ones(max_ent_shape),
                         np.ufunc.reduce(np.multiply, max_ent_shape))
