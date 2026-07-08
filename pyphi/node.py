# node.py
"""Represents a node in a substrate."""

import functools

import numpy as np

from . import utils
from .connectivity import get_inputs_from_cm
from .connectivity import get_outputs_from_cm
from .core.tpm._node_ops import marginalize_out
from .display import Description
from .display import Displayable
from .labels import NodeLabels
from .models.pandas import ToPandasMixin


@functools.total_ordering
class Node(Displayable, ToPandasMixin):
    """A node in a system.

    Parameters
    ----------
    cause_marginal : pyphi.core.tpm.marginalization.CauseMarginals
        Per-system-unit cause factors; this node reads its own factor via
        ``cause_marginal.factor(index)``.
    effect_marginal : pyphi.core.tpm.factored.FactoredTPM
        The system's effect (forward) TPM in per-node-factored form; this
        node reads its own factor via ``effect_marginal.factor(index)``.
    cm : numpy.ndarray
        The connectivity matrix of the system.
    index : int
        The node's index in the substrate.
    state : int
        The state of this node.
    node_labels : :class:`~pyphi.labels.NodeLabels`
        Labels for these nodes.

    Attributes
    ----------
    cause_marginal : numpy.ndarray
    effect_marginal : numpy.ndarray
        The node's marginal cause and effect TPMs. Each is indexed by the
        states of this node's inputs: an input dimension has the size of that
        input's alphabet, while a non-input dimension is collapsed to size 1
        (so a node with ``m`` binary inputs contributes a 2^m × 2 marginal).
        The trailing axis holds this node's own state distribution — its cause
        (previous-timestep) state for ``cause_marginal`` and its effect
        (next-timestep) state for ``effect_marginal`` — so that ``[..., 0]``
        gives the probabilities that the node is OFF and ``[..., 1]`` that it
        is ON.
    """

    def __init__(self, cause_marginal, effect_marginal, cm, index, state, node_labels):
        # This node's index in the list of nodes.
        self.index = index

        # State of this node.
        self.state = state

        # Node labels used in the system
        self.node_labels = node_labels

        # Get indices of the inputs.
        self._inputs = frozenset(get_inputs_from_cm(self.index, cm))
        self._outputs = frozenset(get_outputs_from_cm(self.index, cm))

        # Generate the node's marginal TPMs.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Each per-unit factor has shape (*alphabet_sizes, k_i): the leading
        # axes index the substrate units' state at |t|, the trailing axis this
        # unit's own state. Marginalizing out the units that are not inputs to
        # this node collapses their axes (and this node's own previous-state
        # axis) to size 1, for any per-node alphabet size.
        cause_factor = cause_marginal.factor(self.index)
        cause_non_inputs = set(range(cause_factor.ndim - 1)) - self._inputs
        self.cause_marginal = marginalize_out(cause_factor, cause_non_inputs)

        effect_factor = effect_marginal.factor(self.index)
        effect_non_inputs = set(range(effect_factor.ndim - 1)) - self._inputs
        self.effect_marginal = marginalize_out(effect_factor, effect_non_inputs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Only compute the hash once.
        self._hash = hash(
            (
                index,
                utils.np_hash(self.cause_marginal),
                utils.np_hash(self.effect_marginal),
                self.state,
                self._inputs,
                self._outputs,
            )
        )

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

    def _pandas_record(self):
        return {"node": self.index, "label": self.label, "state": self.state}

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        return Description(title="Node", compact=self.label)

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they belong to the same system and have the
        same index (their TPMs must be the same in that case, so this method
        doesn't need to check TPM equality).

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            self.index == other.index
            and np.array_equal(self.cause_marginal, other.cause_marginal)
            and np.array_equal(self.effect_marginal, other.effect_marginal)
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


def generate_nodes(
    cause_marginal, effect_marginal, cm, substrate_state, indices, node_labels=None
):
    """Generate :class:`Node` objects for a system.

    Parameters
    ----------
    cause_marginal : pyphi.core.tpm.marginalization.CauseMarginals
        Per-system-unit cause factors; each node reads its own factor via
        ``cause_marginal.factor(index)``.
    effect_marginal : pyphi.core.tpm.factored.FactoredTPM
        The system's effect (forward) TPM in per-node-factored form; each
        node reads its own factor via ``effect_marginal.factor(index)``.
    cm : numpy.ndarray
        The corresponding connectivity matrix.
    substrate_state : tuple
        The state of the substrate.
    indices : tuple[int, ...]
        Indices to generate nodes for.
    node_labels : :class:`~pyphi.labels.NodeLabels`, optional
        Textual labels for each node. If ``None``, default labels are
        generated for ``indices``.

    Returns
    -------
    tuple[Node, ...]
        The nodes of the system.
    """
    if node_labels is None:
        node_labels = NodeLabels(None, indices)

    node_state = utils.state_of(indices, substrate_state)

    return tuple(
        Node(cause_marginal, effect_marginal, cm, index, state, node_labels)
        for index, state in zip(indices, node_state, strict=False)
    )
