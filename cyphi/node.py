#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# TODO extend to nonbinary nodes
# TODO? refactor to use purely indexes for nodes
class Node:

    """A node in a network.

    Contains a TPM for just this node (indexed by network state). The TPM gives
    the probability that this node is on.

    For example, in a 3-node network, ``self.tpm[0][1][0]`` gives the
    probability that this node is on at |t_0| if the state of the network
    is |0,1,0| at |t_{-1}|.
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

        # TODO test
        if self.network.connectivity_matrix != None:
            # If a connectivity matrix was provided, store the indices of nodes
            # that connect to this node
            self._input_indices = np.array(
                [index for index in range(self.network.size) if
                 self.network.connectivity_matrix[index][self.index]])
        else:
            # If no connectivity matrix was provided, assume all nodes connect
            # to all nodes
            self._input_indices = tuple(range(self.network.size))

        # The node's conditional transition probability matrix (gives
        # probability that node is on)
        # TODO extend to nonbinary nodes
        self.tpm = network.tpm[..., index]

        # Make the TPM immutable (for hashing)
        self.tpm.flags.writeable = False

    # ``inputs`` must be a property because at the time of node creation, the
    # network doesn't have a list of nodes yet (so we can only store the input
    # indices during initialization, rather than a set of actual nodes, which
    # is more convenient)
    def getinputs(self):
        return set([node for node in self.network.nodes if node.index in
                    self._input_indices])
    inputs = property(getinputs, "The set of nodes with connections to this node.")

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
        return self.index == other.index and self.network == other.network

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.network, self.index))
