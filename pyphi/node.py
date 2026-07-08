# node.py
"""Represents a node in a substrate."""

import functools

import numpy as np

from . import utils
from .connectivity import get_inputs_from_cm
from .connectivity import get_outputs_from_cm
from .core.tpm.joint_distribution import JointTPM
from .display import Description
from .display import Displayable
from .labels import NodeLabels
from .models.pandas import ToPandasMixin


# TODO extend to nonbinary nodes
@functools.total_ordering
class Node(Displayable, ToPandasMixin):
    """A node in a system.

    Parameters
    ----------
    cause_marginal : pyphi.core.tpm.marginalization.CauseMarginals
        Per-system-unit cause factors; this node reads its own factor via
        ``cause_marginal.factor(index)``.
    effect_marginal : FactoredTPM or JointTPM
        The system's effect (forward) TPM. A
        :class:`~pyphi.core.tpm.factored.FactoredTPM` (the per-node-factored
        form, including k-ary substrates) or a
        :class:`~pyphi.core.tpm.joint_distribution.JointTPM` in binary
        state-by-node form.
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
    cause_marginal : pyphi.core.tpm.joint_distribution.JointTPM
    effect_marginal : pyphi.core.tpm.joint_distribution.JointTPM
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

        # Generate the node's TPMs.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # We begin by getting the part of the system's TPM that gives just
        # the state of this node. This part is still indexed by substrate state,
        # but its last dimension will be gone, since now there's just a single
        # scalar value (this node's state) rather than a state-vector for all
        # the substrate nodes.

        # Cause: use the per-unit factor accessor to obtain shape
        # (*alphabet_sizes, k_i) with the per-state conditional along the
        # trailing axis, then marginalize out substrate nodes that are not
        # inputs to this node. The substrate-unit axes are the leading
        # ``ndim - 1`` dimensions; deriving non-inputs from those collapses the
        # node's own previous-state dimension and any non-input dimension to
        # size 1 for every per-node alphabet size.
        cause_factor = JointTPM(cause_marginal.factor(self.index))
        cause_non_inputs = set(range(cause_factor.ndim - 1)) - self._inputs
        self.cause_marginal = cause_factor.marginalize_out(cause_non_inputs)

        # Extract the per-node forward factor. A FactoredTPM exposes
        # factor(index) as the full per-node conditional (*alpha, k_i); a
        # state-by-node ndarray exposes the per-node on-probability as
        # effect_marginal[..., index].
        from .core.tpm.factored import FactoredTPM as _FactoredTPM

        if isinstance(effect_marginal, _FactoredTPM):
            # k-ary path: extract per-node factor as JointTPM for uniform
            # downstream handling (condition_tpm, marginalize_out, reshape).
            node_factor = JointTPM(effect_marginal.factor(self.index))
            effect_non_inputs = set(range(node_factor.ndim - 1)) - self._inputs
            self.effect_marginal = node_factor.marginalize_out(effect_non_inputs)
        else:
            # Binary path: legacy SBN-form ndarray.
            effect_marginal_on = effect_marginal[..., self.index]
            # Marginalize out non-input nodes that are in the system, since the
            # external nodes have already been dealt with as boundary conditions
            # in the system's TPM.
            effect_non_inputs = set(effect_marginal.tpm_indices()) - self._inputs
            effect_marginal_on = effect_marginal_on.marginalize_out(
                effect_non_inputs
            ).tpm

            # Get the TPM that gives the probability of the node being off,
            # rather than on.
            effect_marginal_off = 1 - effect_marginal_on

            # Combine the on- and off-TPM so that the first dimension is indexed
            # by the state of the node's inputs at t, and the last dimension is
            # indexed by the node's state at t+1.
            self.effect_marginal = JointTPM(
                np.stack([effect_marginal_off, effect_marginal_on], axis=-1),
            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Only compute the hash once.
        self._hash = hash(
            (
                index,
                hash(self.cause_marginal),
                hash(self.effect_marginal),
                self.state,
                self._inputs,
                self._outputs,
            )
        )

    @property
    def cause_marginal_off(self):
        """Cause (backward) TPM of this node with only 'OFF' probabilities."""
        return self.cause_marginal[..., 0]

    @property
    def effect_marginal_off(self):
        """Effect (forward) TPM of this node with only 'OFF' probabilities."""
        return self.effect_marginal[..., 0]

    @property
    def cause_marginal_on(self):
        """Cause (backward) TPM of this node with only 'ON' probabilities."""
        return self.cause_marginal[..., 1]

    @property
    def effect_marginal_on(self):
        """Effect (forward) TPM of this node with only 'ON' probabilities."""
        return self.effect_marginal[..., 1]

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
            and self.cause_marginal.array_equal(other.cause_marginal)
            and self.effect_marginal.array_equal(other.effect_marginal)
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
    effect_marginal : FactoredTPM or JointTPM
        The system's effect (forward) TPM. A
        :class:`~pyphi.core.tpm.factored.FactoredTPM` (the per-node-factored
        form, including k-ary substrates) or a
        :class:`~pyphi.core.tpm.joint_distribution.JointTPM` in binary
        state-by-node form.
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


def expand_node_tpm(tpm):
    """Broadcast a node TPM over the full substrate.

    This differs from broadcasting the TPM of a full system: the last
    dimension (the state of the node) holds only the distribution of *this*
    node, rather than the states of every node.

    Parameters
    ----------
    tpm : pyphi.core.tpm.joint_distribution.JointTPM
        The node TPM to expand.
    """
    uc = JointTPM(np.ones([2 for node in tpm.shape]))
    return uc * tpm
