#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# node.py

"""
Represents a node in a network. Each node has a unique index, its position in
the network's list of nodes.
"""

import functools

import numpy as np

from . import utils
from .connectivity import get_inputs_from_cm, get_outputs_from_cm
from .labels import NodeLabels
from .tpm import marginalize_out, tpm_indices


# TODO extend to nonbinary nodes
@functools.total_ordering
class Node:
    """A node in a subsystem.

    Args:
        tpm (np.ndarray): The TPM of the subsystem.
        cm (np.ndarray): The CM of the subsystem.
        index (int): The node's index in the network.
        state (int): The state of this node.
        node_labels (|NodeLabels|): Labels for these nodes.

    Attributes:
        tpm (np.ndarray): The node TPM is a 2^(n_inputs)-by-2 matrix, where
            node.tpm[i][j] gives the marginal probability that the node is in
            state j at t+1 if the state of its inputs is i at t. If the node is
            a single element with a cut selfloop, (i.e. it has no inputs), the
            tpm is simply its unconstrained effect repertoire.
    """

    def __init__(self, tpm, cm, index, state, node_labels):

        # This node's index in the list of nodes.
        self.index = index

        # State of this node.
        self.state = state

        # Node labels used in the system
        self.node_labels = node_labels

        # Get indices of the inputs.
        self._inputs = frozenset(get_inputs_from_cm(self.index, cm))
        self._outputs = frozenset(get_outputs_from_cm(self.index, cm))

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
        non_inputs = set(tpm_indices(tpm)) - self._inputs
        tpm_on = marginalize_out(non_inputs, tpm_on)

        # Get the TPM that gives the probability of the node being off, rather
        # than on.
        tpm_off = 1 - tpm_on

        # Combine the on- and off-TPM so that the first dimension is indexed by
        # the state of the node's inputs at t, and the last dimension is
        # indexed by the node's state at t+1. This representation makes it easy
        # to condition on the node state.
        self.tpm = np.stack([tpm_off, tpm_on], axis=-1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Make the TPM immutable (for hashing).
        utils.np_immutable(self.tpm)

        # Only compute the hash once.
        self._hash = hash((index, utils.np_hash(self.tpm), self.state,
                           self._inputs, self._outputs))

    @property
    def tpm_off(self):
        """The TPM of this node containing only the 'OFF' probabilities."""
        return self.tpm[..., 0]

    @property
    def tpm_on(self):
        """The TPM of this node containing only the 'ON' probabilities."""
        return self.tpm[..., 1]

    @property
    def inputs(self):
        """The set of nodes with connections to this node."""
        return self._inputs

    @property
    def outputs(self):
        """The set of nodes this node has connections to."""
        return self._outputs

    @property
    def label(self):
        """The textual label for this node."""
        return self.node_labels[self.index]

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
                self.inputs == other.inputs and
                self.outputs == other.outputs)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.index < other.index

    def __hash__(self):
        return self._hash

    # TODO do we need more than the index?
    def to_json(self):
        """Return a JSON-serializable representation."""
        return self.index


def generate_nodes(tpm, cm, network_state, indices, node_labels=None):
    """Generate |Node| objects for a subsystem.

    Args:
        tpm (np.ndarray): The system's TPM
        cm (np.ndarray): The corresponding CM.
        network_state (tuple): The state of the network.
        indices (tuple[int]): Indices to generate nodes for.

    Keyword Args:
        node_labels (|NodeLabels|): Textual labels for each node.

    Returns:
        tuple[Node]: The nodes of the system.
    """
    if node_labels is None:
        node_labels = NodeLabels(None, indices)

    node_state = utils.state_of(indices, network_state)

    return tuple(Node(tpm, cm, index, state, node_labels)
                 for index, state in zip(indices, node_state))


def expand_node_tpm(tpm):
    """Broadcast a node TPM over the full network.

    This is different from broadcasting the TPM of a full system since the last
    dimension (containing the state of the node) contains only the probability
    of *this* node being on, rather than the probabilities for each node.
    """
    uc = np.ones([2 for node in tpm.shape])
    return uc * tpm
