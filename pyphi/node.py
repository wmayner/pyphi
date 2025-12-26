# node.py
"""Represents a node in a network."""

import functools

import numpy as np

from . import utils
from .connectivity import get_inputs_from_cm, get_outputs_from_cm
from .labels import NodeLabels
from .tpm import ExplicitTPM


# TODO extend to nonbinary nodes
@functools.total_ordering
class Node:
    """A node in a subsystem.

    Args:
        cause_tpm (ExplicitTPM): The cause (backward) TPM of the subsystem.
        effect_tpm (ExplicitTPM): The effect (forward) TPM of the subsystem.
        cm (np.ndarray): The CM of the subsystem.
        index (int): The node's index in the network.
        state (int): The state of this node.
        node_labels (|NodeLabels|): Labels for these nodes.

    Attributes:
        cause_tpm (ExplicitTPM),
        effect_tpm (ExplicitTPM): The node TPM is an array with shape ``(2,)*(n + 1)``,
            where ``n`` is the size of the |Network|. The first ``n``
            dimensions correspond to each node in the system. Dimensions
            corresponding to nodes that provide input to this node are of size
            2, while those that do not correspond to inputs are of size 1, so
            that the TPM has |2^m x 2| elements where |m| is the number of
            inputs. The last dimension corresponds to the state of the node in
            the next timestep, so that ``node.tpm[..., 0]`` gives probabilities
            that the node will be 'OFF' and ``node.tpm[..., 1]`` gives
            probabilities that the node will be 'ON'.
    """

    def __init__(self, cause_tpm, effect_tpm, cm, index, state, node_labels):
        # This node's index in the list of nodes.
        self.index = index

        # State of this node.
        self.state = state

        # Node labels used in the system
        self.node_labels = node_labels

        # Get indices of the inputs.
        self._inputs = frozenset(get_inputs_from_cm(self.index, cm))
        self._outputs = frozenset(get_outputs_from_cm(self.index, cm))

        # Generate the node's TPMs.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We begin by getting the part of the subsystem's TPM that gives just
        # the state of this node. This part is still indexed by network state,
        # but its last dimension will be gone, since now there's just a single
        # scalar value (this node's state) rather than a state-vector for all
        # the network nodes.
        cause_tpm_on = cause_tpm[..., self.index]
        effect_tpm_on = effect_tpm[..., self.index]

        # TODO extend to nonbinary nodes
        # Marginalize out non-input nodes that are in the subsystem, since the
        # external nodes have already been dealt with as boundary conditions in
        # the subsystem's TPM.

        # TODO use names rather than indices
        cause_non_inputs = set(cause_tpm.tpm_indices()) - self._inputs
        cause_tpm_on = cause_tpm_on.marginalize_out(cause_non_inputs).tpm

        effect_non_inputs = set(effect_tpm.tpm_indices()) - self._inputs
        effect_tpm_on = effect_tpm_on.marginalize_out(effect_non_inputs).tpm

        # Get the TPM that gives the probability of the node being off, rather
        # than on.
        cause_tpm_off = 1 - cause_tpm_on
        effect_tpm_off = 1 - effect_tpm_on

        # Combine the on- and off-TPM so that the first dimension is indexed by
        # the state of the node's inputs at t, and the last dimension is
        # indexed by the node's state at t+1. This representation makes it easy
        # to condition on the node state.
        self.cause_tpm = ExplicitTPM(
            np.stack([cause_tpm_off, cause_tpm_on], axis=-1),
        )
        self.effect_tpm = ExplicitTPM(
            np.stack([effect_tpm_off, effect_tpm_on], axis=-1),
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Only compute the hash once.
        self._hash = hash(
            (
                index,
                hash(self.cause_tpm),
                hash(self.effect_tpm),
                self.state,
                self._inputs,
                self._outputs,
            )
        )

    @property
    def cause_tpm_off(self):
        """The cause (backward) TPM of this node containing only the 'OFF' probabilities."""
        return self.cause_tpm[..., 0]

    @property
    def effect_tpm_off(self):
        """The effect (forward) TPM of this node containing only the 'OFF' probabilities."""
        return self.effect_tpm[..., 0]

    @property
    def cause_tpm_on(self):
        """The cause (backward) TPM of this node containing only the 'ON' probabilities."""
        return self.cause_tpm[..., 1]

    @property
    def effect_tpm_on(self):
        """The effect (forward) TPM of this node containing only the 'ON' probabilities."""
        return self.effect_tpm[..., 1]

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
        return (
            self.index == other.index
            and self.cause_tpm.array_equal(other.cause_tpm)
            and self.effect_tpm.array_equal(other.effect_tpm)
            and self.state == other.state
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

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


def generate_nodes(cause_tpm, effect_tpm, cm, network_state, indices, node_labels=None):
    """Generate |Node| objects for a subsystem.

    Args:
        cause_tpm (ExplicitTPM): The system's cause (backward) TPM
        effect_tpm (ExplicitTPM): The system's effect (forward) TPM
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

    return tuple(
        Node(cause_tpm, effect_tpm, cm, index, state, node_labels)
        for index, state in zip(indices, node_state)
    )


def expand_node_tpm(tpm):
    """Broadcast a node TPM over the full network.

    Args:
        tpm (ExplicitTPM): The node TPM to expand.

    This is different from broadcasting the TPM of a full system since the last
    dimension (containing the state of the node) contains only the probability
    of *this* node being on, rather than the probabilities for each node.
    """
    uc = ExplicitTPM(np.ones([2 for node in tpm.shape]))
    return uc * tpm
