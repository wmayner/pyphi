#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import functools


# TODO extend to nonbinary nodes
# TODO? refactor to use purely indexes for nodes
@functools.total_ordering
class Node(object):

    """A node in a network.

    Attributes:
        network (Network): The network the node belongs to.
        index (int): The node's index in the network's list of nodes.
        label (str): An optional label for the node.
        inputs (list(Node)): A list of nodes that have connections to this
            node.
        tpm (np.ndarray): The TPM for this node. ``this_node.tpm[0]`` and
            ``this_node.tpm[1]`` gives the probability tables that this node is
            off and on, respectively, indexed by network state.

    Examples:
        In a 3-node network, ``self.tpm[0][(0, 1, 0)]`` gives the probability
        that this node is off at |t_0| if the state of the network is |0,1,0|
        at |t_{-1}|.
    """

    def __init__(self, network, index, label=None):
        # This node's parent network
        self.network = network
        # This node's index in the network's list of nodes
        self.index = index
        # Label for display
        self.label = label

        # This will hold the list of nodes that have connections to this node.
        # It can only be generated after the network to which this node belongs
        # has finished initializing, so it's set to None for now.
        self._inputs = None
        # Get indices of the inputs
        if self.network.connectivity_matrix is not None:
            # If a connectivity matrix was provided, store the indices of nodes
            # that connect to this node
            self._input_indices = np.array(
                [i for i in range(self.network.size) if
                 self.network.connectivity_matrix[i][self.index]])
        else:
            # If no connectivity matrix was provided, assume all nodes connect
            # to all nodes
            self._input_indices = tuple(range(self.network.size))

        # Generate the node's TPM
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO! document tpm generation
        tpm_on = self.network.tpm[..., self.index]
        tpm_off = 1 - tpm_on
        # Marginalize-out non-input nodes.
        for index in range(self.network.size):
            if index not in self._input_indices:
                # TODO extend to nonbinary nodes
                tpm_on = tpm_on.sum(index, keepdims=True) / 2
                tpm_off = tpm_off.sum(index, keepdims=True) / 2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store the generated TPM
        self.tpm = np.array([tpm_off, tpm_on])
        # Make it immutable (for hashing)
        self.tpm.flags.writeable = False

        # Only compute hash once
        self._hash = hash((self.network, self.index))

    # ``inputs`` must be a property because at the time of node
    # creation, the network doesn't have a list of Node objects yet, only a
    # size (and thus a range of node indices); we want the node's inputs to be
    # a list of actual Node objects, so we defer access to this list of Nodes
    # until it is created.
    def get_inputs(self):
        if self._inputs:
            return self._inputs
        else:
            return set([node for node in self.network.nodes if node.index in
                        self._input_indices])
    inputs = property(get_inputs,
                      "The set of nodes with connections to this node.")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (self.label if self.label is not None
                else 'n' + str(self.index))

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same network and have the
        same index (``tpm`` must be the same in that case, so this method
        doesn't need to check ``tpm`` equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return ((self.index == other.index and self.network == other.network)
                if isinstance(other, type(self)) else False)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        return self.index < other.index
