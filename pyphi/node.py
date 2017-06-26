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
from .connectivity import get_inputs_from_cm, get_outputs_from_cm
from .tpm import marginalize_out, tpm_indices


# TODO extend to nonbinary nodes
@functools.total_ordering
class Node:
    """A node in a subsystem.

    Attributes:
        tpm (np.ndarray):
            The TPM of the subsystem.
        cm (np.ndarray):
            The CM of the subsystem.
        index (int):
            The node's index in the network.
        state (int):
            The state of this node.
        label (str):
            An optional label for the node.
    """

    def __init__(self, tpm, cm, index, state, label):

        # This node's index in the list of nodes.
        self.index = index

        # Label for display.
        self.label = label

        # State of this node.
        self.state = state

        # Get indices of the inputs.
        self._input_indices = get_inputs_from_cm(self.index, cm)
        self._output_indices = get_outputs_from_cm(self.index, cm)

        # Generate the node's TPM.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We begin by getting the part of the subsystem's TPM that gives just
        # the state of this node. This part is still indexed by network state,
        # but its last dimension will be gone, since now there's just a single
        # scalar value (this node's state) rather than a state-vector for all
        # the network nodes.
        tpm_on = tpm[..., self.index]

        # TODO extend to nonbinary nodes
        # Marginalize out non-input nodes that are in the subsystem, since the
        # external nodes have already been dealt with as boundary conditions in
        # the subsystem's TPM.
        non_input_indices = set(tpm_indices(tpm)) - set(self._input_indices)
        tpm_on = marginalize_out(non_input_indices, tpm_on)

        # Get the TPM that gives the probability of the node being off, rather
        # than on.
        tpm_off = 1 - tpm_on

        # Combine the on- and off-TPM.
        self.tpm = np.array([tpm_off, tpm_on])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Make the TPM immutable (for hashing).
        utils.np_immutable(self.tpm)

        # Only compute the hash once.
        self._hash = hash((index, utils.np_hash(self.tpm), self.state,
                           self._input_indices, self._output_indices))

        # Deferred properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ``inputs`` and ``outputs`` must be properties because at the time of
        # node creation, the subsystem doesn't have a list of Node objects yet,
        # only a size (and thus a range of node indices). So, we defer
        # construction until the properties are needed.
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
        return self._inputs

    @property
    def outputs(self):
        """The set of nodes this node has connections to."""
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
                np.array_equal(self.tpm, other.tpm) and
                self.state == other.state and
                self.input_indices == other.input_indices and
                self.output_indices == other.output_indices)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        return self._hash

    # TODO do we need more than the index?
    def to_json(self):
        return self.index


def default_label(index):
    """Default label for a node."""
    return "n{}".format(index)


def default_labels(indices):
    """Default labels for serveral nodes."""
    return tuple(default_label(i) for i in indices)


def generate_nodes(tpm, cm, network_state, labels=None):
    """Generate |Node| objects for a subsystem.

    Args:
        tpm (np.ndarray): The system's TPM
        cm (np.ndarray): The corresponding CM.
        network_state (tuple): The state of the network.

    Keyword Args:
        labels (tuple[str]): Textual labels for each node.

    Returns:
        tuple[|Node|]: The nodes of the system.
    """

    # Indices in the TPM
    indices = tpm_indices(tpm)

    if labels is None:
        labels = default_labels(indices)
    else:
        assert len(labels) == len(indices)

    node_state = utils.state_of(indices, network_state)

    nodes = tuple(Node(tpm, cm, index, state, label=label)
                  for index, state, label in zip(indices, node_state, labels))

    # Finalize inputs and outputs
    for node in nodes:
        node._inputs = tuple(
            n for n in nodes if n.index in node._input_indices)
        node._outputs = tuple(
            n for n in nodes if n.index in node._output_indices)

    return nodes


def expand_node_tpm(tpm):
    """Broadcast a node TPM over the full network.

    This is different from broadcasting the TPM of a full system since the last
    dimension (containing the state of the node) is unitary -- not a state-
    tuple.
    """
    uc = np.ones([2 for node in tpm.shape])
    return uc * tpm
