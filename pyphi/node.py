# node.py

"""Represents a node in a network."""

import functools

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

# TODO rework circular dependency between node.py and tpm.py, instead
# of importing all of pyphi.tpm and relying on late binding of pyphi.tpm.<NAME>
# to avoid the circular import error.
import pyphi.tpm
from .connectivity import get_inputs_from_cm, get_outputs_from_cm
from .labels import NodeLabels
from .utils import state_of


@xr.register_dataarray_accessor("node")
@functools.total_ordering
class Node:
    """A node in a subsystem.

    Args:
        effect_dataarray (xr.DataArray): the xarray DataArray for the effect TPM.

    Keyword Args:
        cause_dataarray (xr.DataArray): the xarray DataArray for the cause TPM.

    Attributes:
        index (int): The node's index in the network.
        label (str): The textual label for this node.
        cause_dataarray (xr.DataArray): the xarray DataArray for the cause TPM.
        effect_dataarray (xr.DataArray): the xarray DataArray for the effect TPM.
        cause_tpm (|ExplicitTPM|),
        effect_tpm (|ExplicitTPM|): The node TPM is an array with |i + 1|
            dimensions, The first ``i`` dimensions correspond to the inputs to
            the |Node|, and are of size > 1 (the possible states of the
            input). The last dimension encodes the state of the node in the next
            timestep, so that ``node.tpm[..., 0]`` gives probabilities that the
            node will be 'OFF' and ``node.tpm[..., 1]`` gives probabilities that
            the node will be 'ON'.
        inputs (frozenset): The set of nodes which send connections to this node.
        outputs (frozenset): The set of nodes this node sends connections to.
            inhabit.
        state (int): The current state of this node.
        shape (Tuple[int]): The expanded shape of this node's TPM.
    """

    def __init__(
        self,
        effect_dataarray: xr.DataArray,
        cause_dataarray: Optional[xr.DataArray] = None,
    ):
        self._index = effect_dataarray.attrs["index"]

        # Node labels used in the system
        self._node_labels = effect_dataarray.attrs["node_labels"]

        self._inputs = effect_dataarray.attrs["inputs"]
        self._outputs = effect_dataarray.attrs["outputs"]

        self._cause_dataarray = cause_dataarray
        self._cause_tpm = (
            self._cause_dataarray.data if cause_dataarray is not None else None
        )

        self._effect_dataarray = effect_dataarray
        self._effect_tpm = self._effect_dataarray.data

        # (Optional) current state of this node.
        self.state = effect_dataarray.attrs["state"]

        # Only compute the hash once.
        self._hash = hash(
            (
                self.index,
                hash(pyphi.tpm.ExplicitTPM(self.cause_tpm)),
                hash(pyphi.tpm.ExplicitTPM(self.effect_tpm)),
                self._inputs,
                self._outputs,
                self.state
            )
        )

    @property
    def index(self):
        """int: The node's index in the network."""
        return self._index

    @property
    def label(self):
        """str: The textual label for this node."""
        return self._node_labels[self.index]

    @property
    def cause_dataarray(self):
        """|xr.DataArray|: The cause xarray DataArray for this node."""
        return self._cause_dataarray

    @property
    def effect_dataarray(self):
        """|xr.DataArray|: The effect xarray DataArray for this node."""
        return self._effect_dataarray

    @property
    def cause_tpm(self):
        """|ExplicitTPM|: The TPM of this node."""
        return self._cause_tpm

    @property
    def effect_tpm(self):
        """|ExplicitTPM|: The TPM of this node."""
        return self._effect_tpm

    @property
    def inputs(self):
        """frozenset: The set of nodes with connections to this node."""
        return self._inputs

    @property
    def outputs(self):
        """frozenset: The set of nodes this node has connections to."""
        return self._outputs

    @property
    def state(self):
        """Optional[int]: The current state of this node."""
        return self._state

    @state.setter
    def state(self, value):
        state_space = self.effect_dataarray.coords["Pr"].data
        if value not in (*state_space, None):
            raise ValueError(
                f"Invalid node state. Possible states are {state_space}."
            )

        self._state = value

    @property
    def shape(self):
        """Tuple[int]: The expanded shape of this node's TPM."""
        squeezed_shape = self.effect_tpm.shape
        # A full shape prototype with as many dims as network nodes + 1.
        shape = np.ones(len(self._node_labels) + 1, dtype=int)
        shape[[*self._inputs, -1]] = squeezed_shape
        return tuple(shape)

    def project_index(self, index: Mapping[str, Any]) -> Mapping[str, Any]:
        """Convert absolute |ImplicitTPM| index to a valid one relative to this
        node.

        Args:
            index (Any): The index as provided by ImplicitTPM.__getitem__().

        Returns (Mapping): The dictionary-style index but devoid of dimensions
            missing from this node.
        """
        dimensions = self._effect_dataarray.dims
        new_index = {dim: idx for dim, idx in index.items() if dim in dimensions}
        return new_index

    def __repr__(self):
        return self.label

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they have the same index, the same inputs and
        outputs, the same TPMs and the same state.

        Labels are for display only, so two equal nodes may have different
        labels.

        """
        return (
            self.index == other.index and
            self.cause_tpm.array_equal(other.tpm) and
            self.effect_tpm.array_equal(other.tpm) and
            self.inputs == other.inputs and
            self.outputs == other.outputs and
            self.state == other.state
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


def generate_node(
        effect_tpm: pyphi.tpm.ExplicitTPM,
        cm: np.ndarray,
        index: int,
        node_labels: NodeLabels,
        cause_tpm: Optional[pyphi.tpm.ExplicitTPM] = None,
        state: Optional[int] = None,
        uncut_cm: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """
    Instantiate a node TPM DataArray.

    Args:
        effect_tpm (ExplicitTPM): The effect TPM of this node.
        cm (np.ndarray): The CM of the network.
        index (int): The node's index in the network.
        node_labels (NodeLabels): Textual labels for each node in the network.

    Keyword Args:
        cause_tpm (Optional[ExplicitTPM]): The cause TPM of this node.
        state (Optional[int]): The state of this node.
        uncut_cm (Optional[np.ndarray]): The original CM of the network.

    Returns:
        xr.DataArray: The node in question.
    """
    # Get indices of the inputs and outputs.
    inputs = get_inputs_from_cm(index, cm)
    outputs = get_outputs_from_cm(index, cm)

    if uncut_cm is not None:
        # Marginalize out non-input nodes (required by cut Subsystems).
        original_inputs = get_inputs_from_cm(index, uncut_cm)
        cut_inputs = set(original_inputs) - set(inputs)
        cut_indices = np.where(
            np.isin(original_inputs, list(cut_inputs))
        )[0]
        effect_tpm = effect_tpm.marginalize_out(cut_indices.tolist()).squeeze()
        cause_tpm = cause_tpm.marginalize_out(cut_indices.tolist()).squeeze()

    # Dimensions are the names of this node's inputs plus the last dimension
    # with the probability for each state of this node in the next timestep.
    dimensions = node_labels.indices2labels(inputs) + ("Pr",)

    # The possible states for each dimension.
    coordinates = tuple(range(dim) for dim in effect_tpm.shape)

    attributes = {
        "index": index,
        "node_labels": node_labels,
        "cm": cm,
        "inputs": frozenset(inputs),
        "outputs": frozenset(outputs),
        "state": state,
    }

    cause_dataarray = xr.DataArray(
        name=node_labels[index],
        data=cause_tpm,
        dims=dimensions,
        coords=coordinates,
        attrs=attributes,
    ) if cause_tpm is not None else None

    effect_dataarray = xr.DataArray(
        name=node_labels[index],
        data=effect_tpm,
        dims=dimensions,
        coords=coordinates,
        attrs=attributes,
    )

    return Node(effect_dataarray, cause_dataarray)


def generate_nodes(
        network_tpm,
        cm: np.ndarray,
        indices: Tuple[int],
        node_labels: NodeLabels,
        network_state: Optional[Tuple[int]] = None,
) -> Tuple[xr.DataArray]:
    """Generate |Node| objects out of a binary network |ExplicitTPM|.

    Args:
        network_tpm (|ExplicitTPM, ImplicitTPM|): The system's TPM.
        cm (np.ndarray): The CM of the network.
        indices (Tuple[int]): Indices to generate nodes for.
        node_labels (NodeLabels): Textual labels for each node.

    Keyword Args:
        network_state (Optional[Tuple[int, str]]): The state of the network.

    Returns:
        Tuple[xr.DataArray]: The nodes of the system.
    """
    if isinstance(network_tpm, pyphi.tpm.ImplicitTPM):
        network_tpm = pyphi.tpm.reconstitute_tpm(network_tpm)

    if network_state is None:
        network_state = (None,) * cm.shape[0]

    node_state = state_of(indices, network_state)

    nodes = []

    for index, state in zip(indices, node_state):
        # We begin by getting the part of the subsystem's TPM that gives just
        # the state of this node. This part is still indexed by network state,
        # but its last dimension will be gone, since now there's just a single
        # scalar value (this node's state) rather than a state-vector for all
        # the network nodes.
        tpm_on = network_tpm[..., index]

        # Get the TPM that gives the probability of the node being off, rather
        # than on.
        tpm_off = 1 - tpm_on

        # Combine the on- and off-TPM so that the first dimension is indexed by
        # the state of the node's inputs at t, and the last dimension is
        # indexed by the node's state at t+1. This representation makes it easy
        # to condition on the node state.
        node_tpm = pyphi.tpm.ExplicitTPM(
            np.stack([np.asarray(tpm_off), np.asarray(tpm_on)], axis=-1)
        )

        # Marginalize out non-input nodes (network_tpm is |ExplicitTPM|).
        inputs = get_inputs_from_cm(index, cm)
        non_inputs = frozenset(node_tpm.tpm_indices()) - frozenset(inputs)
        node_tpm = node_tpm.marginalize_out(non_inputs).squeeze()

        node = generate_node(node_tpm, cm, index, node_labels, state=state)
        nodes.append(node)

    return tuple(nodes)


# TODO: nonbinary nodes
def expand_node_tpm(tpm):
    """Broadcast a node TPM over the full network.

    Args:
        tpm (|ExplicitTPM|): The node TPM to expand.

    This is different from broadcasting the TPM of a full system since the last
    dimension (containing the state of the node) contains only the probability
    of *this* node being on, rather than the probabilities for each node.
    """
    uc = pyphi.tpm.ExplicitTPM(np.ones([2 for node in tpm.shape]))
    return uc * tpm
