# node.py
"""Represents a node in a network."""

import functools

from typing import Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import xarray as xr

# TODO rework circular dependency between node.py and tpm.py, instead
# of importing all of pyphi.tpm and relying on late binding of pyphi.tpm.<NAME>
# to avoid the circular import error.
import pyphi.tpm

from .connectivity import get_inputs_from_cm, get_outputs_from_cm
from .state_space import (
    dimension_labels,
    build_state_space,
    SINGLETON_COORDINATE,
)
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
        node_labels (Tuple[str]): The textual labels for the nodes in the network.
        cause_dataarray (xr.DataArray): the xarray DataArray for the cause TPM.
        effect_dataarray (xr.DataArray): the xarray DataArray for the effect TPM.
        cause_tpm (|ExplicitTPM|),
        effect_tpm (|ExplicitTPM|): The node TPM is an array with |n + 1| dimensions,
            where ``n`` is the size of the |Network|. The first ``n`` dimensions
            correspond to each node in the system. Dimensions corresponding to
            nodes that provide input to this node are of size > 1, while those
            that do not correspond to inputs are of size 1. The last dimension
            encodes the state of the node in the next timestep, so that
            ``node.tpm[..., 0]`` gives probabilities that the node will be 'OFF'
            and ``node.tpm[..., 1]`` gives probabilities that the node will be
            'ON'.
        inputs (frozenset): The set of nodes which send connections to this node.
        outputs (frozenset): The set of nodes this node sends connections to.
        state_space (Tuple[Union[int, str]]): The space of states this node can
            inhabit.
        state (Optional[Union[int, str]]): The current state of this node.
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

        if cause_dataarray is None:
            self._cause_dataarray = None
            self._cause_tpm = None
        else:
            self._cause_dataarray = cause_dataarray
            self._cause_tpm = cause_dataarray.data

        self._effect_dataarray = effect_dataarray
        self._effect_tpm = self._effect_dataarray.data

        self.state_space = effect_dataarray.attrs["state_space"]

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
                self.state_space,
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
    def state_space(self):
        """Tuple[Union[int, str]]: The space of states this node can inhabit."""
        return self._state_space

    @state_space.setter
    def state_space(self, value):
        _state_space = tuple(value)

        if len(set(_state_space)) < len(_state_space):
            raise ValueError(
                "Invalid node state space tuple. Repeated states are ambiguous."
            )

        if len(_state_space) < 2:
            raise ValueError(
                "Invalid node state space with less than 2 states."
            )

        self._state_space = _state_space

    @property
    def state(self):
        """Optional[Union[int, str]]: The current state of this node."""
        return self._state

    @state.setter
    def state(self, value):
        if value not in (*self.state_space, None):
            raise ValueError(
                f"Invalid node state. Possible states are {self.state_space}."
            )

        self._state = value

    def project_index(self, index, preserve_singletons=False):
        """Convert absolute TPM index to a valid index relative to this node."""

        # Supported index coordinates (in the right dimension order)
        # respective to this node, to be used like an AND mask, with
        # `singleton_coordinate` acting like 0.
        dimensions = self._effect_dataarray.dims
        coordinates = self._effect_dataarray.coords

        support = {dim: tuple(coordinates[dim].values) for dim in dimensions}

        if isinstance(index, dict):
            singleton_coordinate = (
                [SINGLETON_COORDINATE] if preserve_singletons
                else SINGLETON_COORDINATE
            )

            try:
                # Convert potential int dimension indices to common currency of
                # string dimension labels.
                keys = [
                    k if isinstance(k, str) else dimensions[k]
                    for k in index.keys()
                ]

                projected_index = {
                    key: value if support[key] != (SINGLETON_COORDINATE,)
                    else singleton_coordinate
                    for key, value in zip(keys, index.values())
                }

            except KeyError as e:
                raise ValueError(
                    "Dimension {} does not exist. Expected one or more of: "
                    "{}.".format(e, dimensions)
                ) from e

            return projected_index

        # Assume regular index otherwise.

        if not isinstance(index, tuple):
            # Index is a single int, slice, ellipsis, etc. Make it
            # amenable to zip().
            index = (index,)

        index_support_map = zip(index, support.values())
        singleton_coordinate = [0] if preserve_singletons else 0
        projected_index = tuple(
            i if support != (SINGLETON_COORDINATE,)
            else singleton_coordinate
            for i, support in index_support_map
        )

        return projected_index

    # def __getitem__(self, index):
    #     return self._dataarray[index].node

    def __repr__(self):
        return self.label

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        """Return whether this node equals the other object.

        Two nodes are equal if they have the same index, the same
        inputs and outputs, the same TPMs, the same state_space and the
        same state.

        Labels are for display only, so two equal nodes may have different
        labels.
        """
        return (
            self.index == other.index and
            self.cause_tpm.array_equal(other.tpm) and
            self.effect_tpm.array_equal(other.tpm) and
            self.inputs == other.inputs and
            self.outputs == other.outputs and
            self.state_space == other.state_space and
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
        network_state_space: Mapping[str, Tuple[Union[int, str]]],
        index: int,
        node_labels: Iterable[str],
        cause_tpm: Optional[pyphi.tpm.ExplicitTPM] = None,
        state: Optional[Union[int, str]] = None,
) -> xr.DataArray:
    """
    Instantiate a node TPM DataArray.

    Args:
        effect_tpm (ExplicitTPM): The effect TPM of this node.
        cm (np.ndarray): The CM of the network.
        network_state_space (Mapping[str, Tuple[Union[int, str]]]):
            Labels for the state space of each node in the network.
        index (int): The node's index in the network.
        node_labels (Iterable[str]): Textual labels for each node in the network.

    Keyword Args:
        cause_tpm (ExplicitTPM): The cause TPM of this node.
        state (Optional[Union[int, str]]): The state of this node.

    Returns:
        xr.DataArray: The node in question.
    """
    # Get indices of the inputs and outputs.
    inputs = frozenset(get_inputs_from_cm(index, cm))
    outputs = frozenset(get_outputs_from_cm(index, cm))

    # Marginalize out non-input nodes.
    effect_non_inputs = set(effect_tpm.tpm_indices()) - inputs
    effect_tpm = effect_tpm.marginalize_out(effect_non_inputs)

    if cause_tpm is not None:
        cause_non_inputs = set(cause_tpm.tpm_indices()) - inputs
        cause_tpm = cause_tpm.marginalize_out(cause_non_inputs)

    # Dimensions are the names of this node's parents (whose state this node's
    # TPM can be conditioned on), plus the last dimension with the probability
    # for each possible state of this node in the next timestep.
    dimensions = dimension_labels(node_labels)

    # Compute the relevant state labels (coordinates in xarray terminology) from
    # the perspective of this node and its direct inputs.
    node_states = [network_state_space[dim] for dim in dimensions[:-1]]
    input_coordinates, _ = build_state_space(
        node_labels,
        effect_tpm.shape[:-1],
        node_states,
        singleton_state_space=(SINGLETON_COORDINATE,),
    )

    node_state_space = network_state_space[dimensions[index]]

    coordinates = {**input_coordinates, dimensions[-1]: node_state_space}

    cause_dataarray = xr.DataArray(
        name=node_labels[index],
        data=cause_tpm,
        dims=dimensions,
        coords=coordinates,
        attrs={
            "index": index,
            "node_labels": node_labels,
            "cm": cm,
            "inputs": inputs,
            "outputs": outputs,
            "state_space": tuple(node_state_space),
            "state": state,
            "network_state_space": network_state_space
        }
    ) if cause_tpm is not None else None

    effect_dataarray = xr.DataArray(
        name=node_labels[index],
        data=effect_tpm,
        dims=dimensions,
        coords=coordinates,
        attrs={
            "index": index,
            "node_labels": node_labels,
            "cm": cm,
            "inputs": inputs,
            "outputs": outputs,
            "state_space": tuple(node_state_space),
            "state": state,
            "network_state_space": network_state_space
        }
    )

    return Node(effect_dataarray, cause_dataarray)


def generate_nodes(
        network_tpm,
        cm: np.ndarray,
        state_space: Mapping[str, Tuple[Union[int, str]]],
        indices: Tuple[int],
        node_labels: Tuple[str],
        network_state: Optional[Tuple[Union[int, str]]] = None,
) -> Tuple[xr.DataArray]:
    """Generate |Node| objects out of a binary network |TPM|.

    Args:
        network_tpm (|ExplicitTPM, ImplicitTPM|): The system's TPM.
        cm (np.ndarray): The CM of the network.
        state_space (Mapping[str, Tuple[Union[int, str]]]): Labels
            for the state space of each node in the network.
        indices (Tuple[int]): Indices to generate nodes for.
        node_labels (Optional[Tuple[str]]): Textual labels for each node.

    Keyword Args:
        network_state (Optional[Tuple[Union[int, str]]]): The state of
            the network.

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

        nodes.append(
            generate_node(
                node_tpm,
                cm,
                state_space,
                index,
                node_labels,
                cause_tpm=None,
                state=state,
            )
        )

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
