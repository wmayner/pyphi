#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# node.py

"""
Represents a node in a subsystem. Each node has a unique index, its position
in the network's list of nodes.
"""

import functools

import numpy as np

from . import utils


# TODO extend to nonbinary nodes
@functools.total_ordering
class Node:
    """A node in a subsystem.

    Attributes:
        subsystem (Subsystem):
            The subsystem the node belongs to.
        index (int):
            The node's index in the network.
        network (Network):
            The network the node belongs to.
        label (str):
            An optional label for the node.
        state (int):
            The state of this node.
    """

    def __init__(self, subsystem, index, indices=None, label=None):
        # This node's parent subsystem.
        self.subsystem = subsystem
        # This node's index in the list of nodes.
        self.index = index
        # This node's parent network.
        self.network = subsystem.network

        # Label for display.
        if label is None:
            label = 'n' + str(index)
        self.label = label

        # State of this node.
        self.state = self.subsystem.state[self.index]
        # Get indices of the inputs.
        self._input_indices = utils.get_inputs_from_cm(
            self.index, subsystem.cm)
        self._output_indices = utils.get_outputs_from_cm(
            self.index, subsystem.cm)

        # Generate the node's TPM.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We begin by getting the part of the subsystem's TPM that gives just
        # the state of this node. This part is still indexed by network state,
        # but its last dimension will be gone, since now there's just a single
        # scalar value (this node's state) rather than a state-vector for all
        # the network nodes.
        tpm_on = self.subsystem.tpm[..., self.index]
        # Get the TPM that gives the probability of the node being off, rather
        # than on.
        tpm_off = 1 - tpm_on

        # Subsystem indices to generate TPM from
        if indices is None:
            indices = subsystem.node_indices

        for i in indices:
            # TODO extend to nonbinary nodes
            # Marginalize out non-input nodes that are in the subsystem, since
            # the external nodes have already been dealt with as boundary
            # conditions in the subsystem's TPM.
            if i not in self._input_indices:
                tpm_on = tpm_on.sum(i, keepdims=True) / 2
                tpm_off = tpm_off.sum(i, keepdims=True) / 2

        # Combine the on- and off-TPM.
        self.tpm = np.array([tpm_off, tpm_on])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Make the TPM immutable (for hashing).
        self.tpm.flags.writeable = False

        # Only compute the hash once.
        self._hash = hash((self.index, self.subsystem))

        # Deferred properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ``inputs`` and ``outputs`` must be properties because at
        # the time of node creation, the subsystem doesn't have a list of Node
        # objects yet, only a size (and thus a range of node indices). So, we
        # defer construction until the properties are needed.
        self._inputs = None
        self._outputs = None

    @property
    def input_indices(self):
        """The indices of nodes which connect to this node."""
        return self._input_indices

    @property
    def output_indices(self):
        """The indices of nodes that this node connects to."""
        return self._output_indices

    @property
    def inputs(self):
        """The set of nodes with connections to this node."""
        if self._inputs is None:
            self._inputs = [node for node in self.subsystem.nodes if
                            node.index in self._input_indices]
        return self._inputs

    @property
    def outputs(self):
        """The set of nodes this node has connections to."""
        if self._outputs is None:
            self._outputs = [node for node in self.subsystem.nodes if
                             node.index in self._output_indices]
        return self._outputs

    def __repr__(self):
        return self.label

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same subsystem and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (self.index == other.index and
                self.subsystem == other.subsystem)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        return self._hash

    # TODO do we need more than the index?
    def to_json(self):
        return self.index


# TODO: rework MacroSubsystem to not need the indices arg
def generate_nodes(subsystem, indices=None, labels=False):
    """Generate the |Node| objects for these indices.

    Args:
        subsystem (Subsystem): The subsystem for which nodes are being
            generated.

    Keyword Args:
        indices (tuple(int)): Used by |MacroSubsystem| to force generation to
            use certain indices.
        labels (boolean): If True, nodes will be labeled with the labels of the
            network. (This is also used by macro systems to keep labels from
            being mixed up when many micro elements are combined into one macro
            element.)

    Returns:
        tuple(Node): The nodes of the |Subsystem|.
    """
    if indices is None:
        indices = subsystem.node_indices

    if labels is True:
        labels = subsystem.network.indices2labels(indices)
    else:
        labels = [None] * len(indices)

    return tuple(Node(subsystem, index, indices=indices, label=label)
                 for index, label in zip(indices, labels))


def expand_node_tpm(tpm):
    """Broadcast a node TPM over the full network.

    This is different from broadcasting the TPM of a full system since the last
    dimension (containing the state of the node) is unitary -- not a state-
    tuple.
    """
    uc = np.ones([2 for node in tpm.shape])
    return uc * tpm
