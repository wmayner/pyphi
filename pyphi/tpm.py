#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tpm.py

"""
Provides the ExplicitTPM and related classes.
"""

import math
import functools
from itertools import chain
from typing import Iterable, Mapping, Optional, Set, Tuple

import numpy as np

from . import convert, distribution, data_structures, exceptions
from .connectivity import subadjacency
from .conf import config
from .constants import OFF, ON
from .data_structures import FrozenMap
from .node import node as Node
from .utils import all_states, eq, np_hash, np_immutable

class TPM:
    """TPM interface for derived classes."""

    _ERROR_MSG_PROBABILITY_IMAGE = (
        "Invalid TPM: probabilities must be in the interval [0, 1]."
    )

    _ERROR_MSG_PROBABILITY_SUM = "Invalid TPM: probabilities must sum to 1."

    def validate(self, check_independence=True):
        raise NotImplementedError

    def to_multidimensional_state_by_node(self):
        raise NotImplementedError

    def conditionally_independent(self):
        raise NotImplementedError

    def condition_tpm(self, condition):
        raise NotImplementedError

    def marginalize_out(self, node_indices):
        raise NotImplementedError

    def is_deterministic(self):
        raise NotImplementedError

    def is_state_by_state(self):
        raise NotImplementedError

    def remove_singleton_dimensions(self):
        raise NotImplementedError

    def expand_tpm(self):
        raise NotImplementedError

    def subtpm(self, fixed_nodes, state):
        """Return the TPM for a subset of nodes, conditioned on other nodes.

        Arguments:
            fixed_nodes (tuple[int]): The nodes to select.
            state (tuple[int]): The state of the fixed nodes.

        Returns:
            ExplicitTPM: The TPM of just the subsystem of the free nodes.

        Examples:
            >>> from pyphi import examples
            >>> # Get the TPM for nodes only 1 and 2, conditioned on node 0 = OFF
            >>> reconstitute_tpm(examples.grid3_network().tpm).subtpm((0,), (0,))
            ExplicitTPM(
            [[[[0.02931223 0.04742587]
               [0.07585818 0.88079708]]
            <BLANKLINE>
              [[0.81757448 0.11920292]
               [0.92414182 0.95257413]]]]
            )
        """
        N = self.shape[-1]
        free_nodes = sorted(set(range(N)) - set(fixed_nodes))
        condition = FrozenMap(zip(fixed_nodes, state))
        conditioned_tpm = self.condition_tpm(condition)

        if isinstance(self, ExplicitTPM):
            return conditioned_tpm[..., free_nodes]

        return type(self)(
            tuple(
                node for node in conditioned_tpm.nodes
                if node.index in free_nodes
            )
        )


    def infer_edge(self, a, b, contexts):
        """Infer the presence or absence of an edge from node A to node B.

        Let |S| be the set of all nodes in a network. Let |A' = S - {A}|. We
        call the state of |A'| the context |C| of |A|. There is an edge from |A|
        to |B| if there exists any context |C(A)| such that
        |Pr(B | C(A), A=0) != Pr(B | C(A), A=1)|.

        Args:
            a (int): The index of the putative source node.
            b (int): The index of the putative sink node.
            contexts (tuple[tuple[int]]): The tuple of states of ``a``
        Returns:
            bool: ``True`` if the edge |A -> B| exists, ``False`` otherwise.
        """

        def a_in_context(context):
            """Given a context C(A), return the states of the full system with A
            OFF and ON, respectively.
            """
            a_off = context[:a] + OFF + context[a:]
            a_on = context[:a] + ON + context[a:]
            return (a_off, a_on)

        def a_affects_b_in_context(tpm, context):
            """Return ``True`` if A has an effect on B, given a context."""
            a_off, a_on = a_in_context(context)
            return tpm[a_off][b] != tpm[a_on][b]

        tpm = self.to_multidimensional_state_by_node()
        return any(a_affects_b_in_context(tpm, context) for context in contexts)

    def infer_cm(self):
        """Infer the connectivity matrix associated with a state-by-node TPM in
        multidimensional form.
        """
        tpm = self.to_multidimensional_state_by_node()
        network_size = tpm.shape[-1]
        all_contexts = tuple(all_states(network_size - 1))
        cm = np.empty((network_size, network_size), dtype=int)
        for a, b in np.ndindex(cm.shape):
            cm[a][b] = self.infer_edge(a, b, all_contexts)
        return cm

    def tpm_indices(self, reconstituted=False):
        """Return the indices of nodes in the TPM."""
        shape = self._reconstituted_shape if reconstituted else self.shape
        return tuple(np.where(np.array(shape[:-1]) != 1)[0])

    def print(self):
        raise NotImplementedError

    def permute_nodes(self, permutation):
        raise NotImplementedError

    def backward_tpm(self, current_state, system_indices):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


class ExplicitTPM(data_structures.ArrayLike, TPM):

    """An explicit network TPM in multidimensional form.

    Args:
        tpm (np.array): The transition probability matrix of the |Network|.

            The TPM can be provided in any of three forms: **state-by-state**,
            **state-by-node**, or **multidimensional state-by-node** form.
            In the state-by-node forms, row indices must follow the
            little-endian convention (see :ref:`little-endian-convention`). In
            state-by-state form, column indices must also follow the
            little-endian convention.

            If the TPM is given in state-by-node form, it can be either
            2-dimensional, so that ``tpm[i]`` gives the probabilities of each
            node being ON if the previous state is encoded by |i| according to
            the little-endian convention, or in multidimensional form, so that
            ``tpm[(0, 0, 1)]`` gives the probabilities of each node being ON if
            the previous state is |N_0 = 0, N_1 = 0, N_2 = 1|.

            The shape of the 2-dimensional form of a state-by-node TPM must be
            ``(s, n)``, and the shape of the multidimensional form of the TPM
            must be ``[2] * n + [n]``, where ``s`` is the number of states and
            ``n`` is the number of nodes in the network.

    Keyword Args:
        validate (bool): Whether to check the shape and content of the input
            array for correctness.

    Example:
        In a 3-node network, ``tpm[(0, 0, 1)]`` gives the
        transition probabilities for each node at |t| given that state at |t-1|
        was |N_0 = 0, N_1 = 0, N_2 = 1|.

    Attributes:
        _VALUE_ATTR (str): The key of the attribute holding the TPM array value.
        __wraps__ (type): The class of the array referenced by ``_VALUE_ATTR``.
        __closures__ (frozenset): np.ndarray method names proxied by this class.
    """

    _VALUE_ATTR = "_tpm"

    # TODO(tpm) remove pending ArrayLike refactor
    __wraps__ = np.ndarray

    # TODO(tpm) remove pending ArrayLike refactor
    # Casting semantics: values belonging to our custom TPM class should
    # remain closed under the following methods:
    __closures__ = frozenset(
        {
            "argpartition",
            "astype",
            "byteswap",
            "choose",
            "clip",
            "compress",
            "conj",
            "conjugate",
            "copy",
            "cumprod",
            "cumsum",
            "diagonal",
            "dot",
            "fill",
            "flatten",
            "getfield",
            "item",
            "itemset",
            "max",
            "mean",
            "min",
            "newbyteorder",
            "partition",
            "prod",
            "ptp",
            "put",
            "ravel",
            "repeat",
            "reshape",
            "resize",
            "round",
            "setfield",
            "sort",
            "squeeze",
            "std",
            "sum",
            "swapaxes",
            "take",
            "transpose",
            "var",
            "view",
        }
    )

    def __getattr__(self, name):
        if name in self.__closures__:
            return _new_attribute(name, self.__closures__, self._tpm)
        else:
            return getattr(self.__getattribute__(self._VALUE_ATTR), name)

    def __len__(self):
        return len(self.__getattribute__(self._VALUE_ATTR))

    def __init__(self, tpm, validate=False):
        self._tpm = np.array(tpm)

        if validate:
            self.validate(
                check_independence=config.VALIDATE_CONDITIONAL_INDEPENDENCE,
                network_tpm=True
            )
            self._tpm = self.to_multidimensional_state_by_node()

        self._tpm = np_immutable(self._tpm)
        self._hash = np_hash(self._tpm)

    @property
    def tpm(self):
        """np.ndarray: The underlying `tpm` object."""
        return self._tpm

    @property
    def number_of_units(self):
        if self.is_state_by_state():
            # Assumes binary nodes
            return int(math.log2(self._tpm.shape[1]))
        return self._tpm.shape[-1]

    def validate(self, check_independence=True, network_tpm=False):
        """Validate this TPM."""
        return self._validate_probabilities(network_tpm) and self._validate_shape(
            check_independence
        )

    def _validate_probabilities(self, network_tpm=False):
        """Check that the probabilities in a TPM are valid."""
        # Validate TPM image is within [0, 1] (first axiom of probability).
        if (self._tpm < 0.0).any() or (self._tpm > 1.0).any():
            raise ValueError(self._ERROR_MSG_PROBABILITY_IMAGE)

        # Validate that probabilities sum to 1.
        if not self.is_unitary(network_tpm):
            raise ValueError(self._ERROR_MSG_PROBABILITY_SUM)

        return True

    def is_unitary(self, network_tpm=False):
        """Whether the TPM satisfies the second axiom of probability theory.

        A TPM is unitary if and only if for every current state of the system,
        the probability distribution over next states conditioned on the current
        state sums to 1 (up to |config.PRECISION|).

        Keyword Args:
            network_tpm (bool): Whether ``self`` is an old-style system TPM
                instead of a node TPM.

        Returns:
            bool:
        """
        tpm = self
        if network_tpm and not tpm.is_state_by_state():
            tpm = convert.state_by_node2state_by_state(self)

        # Marginalize last dimension, then check that all integrals are close to 1.
        measures_over_current_states = tpm.sum(axis=-1).ravel()
        return all(eq(p, 1.0) for p in measures_over_current_states)

    def _validate_shape(self, check_independence=True):
        """Validate this TPM's shape.

        The TPM can be in

            * 2-dimensional state-by-state form,
            * 2-dimensional state-by-node form, or
            * multidimensional state-by-node form.
        """
        see_tpm_docs = (
            "See the documentation on TPM conventions and the `pyphi.Network` "
            "object for more information on TPM forms."
        )
        tpm = self._tpm
        # Get the number of nodes from the state-by-node TPM.
        N = tpm.shape[-1]
        if tpm.ndim == 2:
            if not (
                (tpm.shape[0] == 2 ** N and tpm.shape[1] == N)
                or (tpm.shape[0] == tpm.shape[1])
            ):
                raise ValueError(
                    "Invalid shape for 2-D TPM: {}\nFor a state-by-node TPM, "
                    "there must be "
                    "2^N rows and N columns, where N is the "
                    "number of nodes. State-by-state TPM must be square. "
                    "{}".format(tpm.shape, see_tpm_docs)
                )
            if tpm.shape[0] == tpm.shape[1] and check_independence:
                self.conditionally_independent()
        elif tpm.ndim == (N + 1):
            if tpm.shape != tuple([2] * N + [N]):
                raise ValueError(
                    "Invalid shape for multidimensional state-by-node TPM: {}\n"
                    "The shape should be {} for {} nodes. {}".format(
                        tpm.shape, ([2] * N) + [N], N, see_tpm_docs
                    )
                )
        else:
            raise ValueError(
                "Invalid TPM: Must be either 2-dimensional or multidimensional. "
                "{}".format(see_tpm_docs)
            )
        return True

    def to_multidimensional_state_by_node(self):
        """Return the current TPM re-represented in multidimensional
        state-by-node form.

        See the PyPhi documentation on :ref:`tpm-conventions` for more
        information.

        Returns:
            np.ndarray: The TPM in multidimensional state-by-node format.
        """
        if self.is_state_by_state():
            tpm = convert.state_by_state2state_by_node(self._tpm)
        else:
            tpm = convert.to_multidimensional(self._tpm)

        return tpm

    def conditionally_independent(self):
        """Validate that the TPM is conditionally independent."""
        tpm = self._tpm
        tpm = np.array(tpm)
        if self.is_state_by_state():
            there_and_back_again = convert.state_by_node2state_by_state(
                convert.state_by_state2state_by_node(tpm)
            )
        else:
            there_and_back_again = convert.state_by_state2state_by_node(
                convert.state_by_node2state_by_state(tpm)
            )
        if not np.allclose((tpm - there_and_back_again), 0.0):
            raise exceptions.ConditionallyDependentError(
                "TPM is not conditionally independent.\n"
                "See the conditional independence example in the documentation "
                "for more info."
            )
        return True

    def condition_tpm(self, condition: Mapping[int, int]):
        """Return a TPM conditioned on the given fixed node indices, whose
        states are fixed according to the given state-tuple.

        The dimensions of the new TPM that correspond to the fixed nodes are
        collapsed onto their state, making those dimensions singletons suitable
        for broadcasting. The number of dimensions of the conditioned TPM will
        be the same as the unconditioned TPM.

        Args:
            condition (dict[int, int]): A mapping from node indices to the state
                to condition on for that node.

        Returns:
            TPM: A conditioned TPM with the same number of dimensions, with
            singleton dimensions for nodes in a fixed state.
        """
        # Assumes multidimensional form
        conditioning_indices = [[slice(None)]] * (self.ndim - 1)
        for i, state_i in condition.items():
            # Ignore dimensions that are already singletons
            if self.shape[i] != 1:
                # Preserve singleton dimensions in output array with `np.newaxis`
                conditioning_indices[i] = [state_i, np.newaxis]
        # Flatten the indices.
        conditioning_indices = tuple(chain.from_iterable(conditioning_indices))
        # Obtain the actual conditioned TPM by indexing with the conditioning
        # indices.
        return self[conditioning_indices]

    def marginalize_out(self, node_indices):
        """Marginalize out nodes from this TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.

        Returns:
            ExplicitTPM: A TPM with the same number of dimensions, with the nodes
            marginalized out.
        """
        tpm = self.sum(tuple(node_indices), keepdims=True) / (
            np.array(self.shape)[list(node_indices)].prod()
        )
        # Return new TPM object of the same type as self. Assume self had
        # already been validated and converted formatted. Further validation
        # would be problematic for singleton dimensions.
        return type(self)(tpm)

    def is_deterministic(self):
        """Return whether the TPM is deterministic."""
        return np.all(np.logical_or(self._tpm == 1, self._tpm == 0))

    def is_state_by_state(self):
        """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
        ``False``.
        """
        return self.ndim == 2 and self.shape[0] == self.shape[1]

    def remove_singleton_dimensions(self):
        """Remove singleton dimensions from the TPM.

        Singleton dimensions are created by conditioning on a set of elements.
        This removes those elements from the TPM, leaving a TPM that only
        describes the non-conditioned elements.

        Note that indices used in the original TPM must be reindexed for the
        smaller TPM.
        """
        # Don't squeeze out the final dimension (which contains the probability)
        # for networks with one element.
        if self.ndim <= 2:
            return self

        return self.squeeze()[..., self.tpm_indices()]

    def expand_tpm(self):
        """Broadcast a state-by-node TPM so that singleton dimensions are expanded
        over the full network.
        """
        unconstrained = np.ones([2] * (self._tpm.ndim - 1) + [self._tpm.shape[-1]])
        return type(self)(self._tpm * unconstrained)

    def print(self):
        tpm = convert.to_multidimensional(self._tpm)
        for state in all_states(tpm.shape[-1]):
            print(f"{state}: {tpm[state]}")

    # TODO(4.0) docstring
    def permute_nodes(self, permutation):
        if not len(permutation) == self.ndim - 1:
            raise ValueError(
                f"Permutation must have length {self.ndim - 1}, but has length "
                f"{len(permutation)}."
            )
        dimension_permutation = tuple(permutation) + (self.ndim - 1,)
        return type(self)(
            self._tpm.transpose(dimension_permutation)[..., list(permutation)],
        )

    def probability_of_current_state(self, current_state):
        """Return the probability of the current state as a distribution over previous states.

        Arguments:
            current_state (tuple[int]): The current state.
        """
        state_probabilities = np.empty(self.shape)
        if not len(current_state) == self.shape[-1]:
            raise ValueError(
                f"current_state must have length {self.shape[-1]} "
                f"for state-by-node TPM of shape {self.shape}"
            )
        for i in range(self.shape[-1]):
            # TODO extend to nonbinary nodes
            state_probabilities[..., i] = (
                self[..., i] if current_state[i] else (1 - self[..., i])
            )
        return state_probabilities.prod(axis=-1, keepdims=True)

    def backward_tpm(
            self,
            current_state: tuple[int],
            system_indices: Iterable[int],
            remove_background: bool = False,
    ):
        """Compute the backward TPM for a given network state."""
        all_indices = tuple(range(self.number_of_units))
        system_indices = tuple(sorted(system_indices))
        background_indices = tuple(sorted(set(all_indices) - set(system_indices)))
        if not set(system_indices).issubset(set(all_indices)):
            raise ValueError(
                "system_indices must be a subset of `range(self.number_of_units))`"
            )

        # p(u_t | s_{t–1}, w_{t–1})
        pr_current_state = self.probability_of_current_state(current_state)
        # Σ_{s_{t–1}}  p(u_t | s_{t–1}, w_{t–1})
        pr_current_state_given_only_background = pr_current_state.sum(
            axis=tuple(system_indices), keepdims=True
        )
        # Σ_{u'_{t–1}} p(u_t | u'_{t–1})
        normalization = np.sum(pr_current_state)
        #                                              Σ_{s_{t–1}} p(u_t | s_{t–1}, w_{t–1})
        # Σ_{w_{t–1}}   p(s_{i,t} | s_{t–1}, w_{t–1}) ———————————————————————————————————————
        #                                                 Σ_{u'_{t–1}} p(u_t | u'_{t–1})
        backward_tpm = (
            self * pr_current_state_given_only_background / normalization
        ).sum(axis=background_indices, keepdims=True)
        if remove_background:
            # Remove background units from last dimension of the state-by-node TPM
            backward_tpm = backward_tpm[..., list(system_indices)]
        return ExplicitTPM(backward_tpm)

    def array_equal(self, o: object):
        """Return whether this TPM equals the other object.

        Two TPMs are equal if they are instances of the ExplicitTPM class
        and their numpy arrays are equal.
        """
        return isinstance(o, type(self)) and np.array_equal(self._tpm, o._tpm)

    def __getitem__(self, i):
        item = self._tpm[i]
        if isinstance(item, type(self._tpm)):
            item = type(self)(item)
        return item

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "ExplicitTPM(\n{}\n)".format(self._tpm)

    def __hash__(self):
        return self._hash


class ImplicitTPM(TPM):

    """An implicit network TPM containing |Node| TPMs in multidimensional form.

    Args:
        dataset (xr.Dataset):

    Attributes:
    """

    def __init__(self, nodes):
        """Args:
            nodes (pyphi.node.Node)
        """
        self._nodes = tuple(nodes)

    @property
    def nodes(self):
        """Tuple[xr.DataArray]: The node TPMs in this ImplicitTPM"""
        return self._nodes

    @property
    def tpm(self):
        """Tuple[np.ndarray]: Verbose representation of all node TPMs."""
        return tuple(node.tpm for node in self._nodes)

    @property
    def number_of_units(self):
        return len(self.nodes)

    @property
    def ndim(self):
        """int: The number of dimensions of the TPM."""
        return len(self.shape)

    @property
    def shape(self):
        """Tuple[int]: The size or number of coordinates in each dimension."""
        shapes = self.shapes
        return self._node_shapes_to_shape(shapes)

    @property
    def _reconstituted_shape(self):
        shapes = self.shapes
        return self._node_shapes_to_shape(shapes, reconstituted=True)

    @property
    def shapes(self):
        """Tuple[Tuple[int]]: The shapes of each node TPM in this TPM."""
        return [node.tpm.shape for node in self._nodes]

    @staticmethod
    def _node_shapes_to_shape(
            shapes: Iterable[Iterable[int]],
            reconstituted: Optional[bool] = None
    ) -> Tuple[int]:
        """Infer the shape of the equivalent multidimensional |ExplicitTPM|.

        Args:
            shapes (Iterable[Iterable[int]]): The shapes of the individual node
                TPMs in the network, ordered by node index.

        Returns:
            Tuple[int]: The inferred shape of the equivalent TPM.
        """
        # This should recompute the network TPM shape from individual node
        # shapes, as opposed to measuring the size of the state space.

        if not all(len(shape) == len(shapes[0]) for shape in shapes):
            raise ValueError(
                "The provided shapes contain varying number of dimensions."
            )

        N = len(shapes)
        if reconstituted:
            states_per_node = tuple(max(dim) for dim in zip(*shapes))[:-1]
        else:
            states_per_node = tuple(shape[-1] for shape in shapes)

        # Check consistency of shapes across nodes.

        dimensions_from_shapes = tuple(
            set(shape[node_index] for shape in shapes)
            for node_index in range(N)
        )

        for node_index in range(N):
            # Valid state cardinalities along a dimension can be either:
            #  {1, s_i}, s_i != 1  iff node provides input to only some nodes,
            #  {s_i}, s_i != 1     iff node provides input to all nodes.
            valid_cardinalities = (
                {max(dimensions_from_shapes[node_index]), 1},
                {max(dimensions_from_shapes[node_index])}
            )
            if not any(
                    dimensions_from_shapes[node_index] == cardinality
                    for cardinality in valid_cardinalities
            ):
                raise ValueError(
                    "The provided shapes disagree on the number of states of "
                    "node {}.".format(node_index)
                )

        return states_per_node + (N,)

    def validate(self, check_independence=True):
        """Validate this TPM."""
        return self._validate_probabilities() and self._validate_shape()

    def _validate_probabilities(self):
        """Check that the probabilities in a TPM are valid."""
        # An implicit TPM contains valid probabilities if and only if
        # individual node TPMs contain valid probabilities, for every node.
        if all(
                node.tpm._validate_probabilities()
                for node in self._nodes
        ):
            return True

    def is_unitary(self):
        """Whether the TPM satisfies the second axiom of probability theory.

        A TPM is unitary if and only if for every current state of the system,
        the probability distribution over next states conditioned on the current
        state sums to 1 (up to |config.PRECISION|).
        """
        return all(node.tpm.is_unitary() for node in self._nodes)

    def _validate_shape(self):
        """Validate this TPM's shape.

        The inferred shape of the implicit network TPM must be in
        multidimensional state-by-node form, nonbinary and heterogeneous units
        supported.
        """
        N = len(self.nodes)
        if N + 1 != self.ndim:
            raise ValueError(
                "Invalid TPM shape: {} nodes were provided, but their shapes"
                "suggest a {}-node network.".format(N, self.ndim - 1)
            )

        return True

    def to_multidimensional_state_by_node(self):
        """Return the current TPM re-represented in multidimensional
        state-by-node form.

        See the PyPhi documentation on :ref:`tpm-conventions` for more
        information.

        Returns:
            np.ndarray: The TPM in multidimensional state-by-node format.
        """
        return reconstitute_tpm(self)

    # TODO(tpm) accept node labels and state labels in the map.
    def condition_tpm(self, condition: Mapping[int, int]):
        """Return a TPM conditioned on the given fixed node indices, whose
        states are fixed according to the given state-tuple.

        The dimensions of the new TPM that correspond to the fixed nodes are
        collapsed onto their state, making those dimensions singletons suitable
        for broadcasting. The number of dimensions of the conditioned TPM will
        be the same as the unconditioned TPM.

        Args:
            condition (dict[int, int]): A mapping from node indices to the state
                to condition on for that node.

        Returns:
            TPM: A conditioned TPM with the same number of dimensions, with
            singleton dimensions for nodes in a fixed state.
        """
        # Wrapping index elements in a list is the xarray equivalent
        # of inserting a numpy.newaxis, which preserves the singleton even
        # after selection of a single state.
        conditioning_indices = {
            i: (state_i if isinstance(state_i, list) else [state_i])
            for i, state_i in condition.items()
        }

        return self.__getitem__(conditioning_indices, preserve_singletons=True)

    def marginalize_out(self, node_indices):
        """Marginalize out nodes from this TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.

        Returns:
            ImplicitTPM: A TPM with the same number of dimensions, with the nodes
            marginalized out.
        """
        # Leverage ExplicitTPM.marginalize_out() to distribute operation to
        # individual nodes, then assemble into a new ImplicitTPM.
        return type(self)(
            tuple(
                Node(
                    node.tpm.marginalize_out(node_indices),
                    node.dataarray.attrs["cm"],
                    node.dataarray.attrs["network_state_space"],
                    node.index,
                    node_labels=node.dataarray.attrs["node_labels"],
                ).node
                for node in self.nodes
            )
        )

    def is_state_by_state(self):
        """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
        ``False``.
        """
        return False

    def remove_singleton_dimensions(self):
        """Remove singleton dimensions from the TPM.

        Singleton dimensions are created by conditioning on a set of elements.
        This removes those elements from the TPM, leaving a TPM that only
        describes the non-conditioned elements.

        Note that indices used in the original TPM must be reindexed for the
        smaller TPM.
        """
        # Don't squeeze out the final dimension (which contains the probability)
        # for networks with one element.
        if self.ndim <= 2:
            return self

        # Find the set of singleton dimensions for this TPM.
        shape = self._reconstituted_shape
        singletons = set(np.where(np.array(shape) == 1)[0])

        # Squeeze out singleton dimensions and return a new TPM with
        # the surviving nodes.
        return type(self)(
            tuple(node for node in self.squeeze().nodes)
        )

    def probability_of_current_state(
            self,
            current_state: tuple[int]
    ) -> tuple[ExplicitTPM]:
        """Return probability of current state as distribution over previous states.

        Output format is similar to an |ImplicitTPM|, however the last dimension
        only contains the probability for the current state.

        Arguments:
           current_state (tuple[int]): The current state.
        Returns:
           tuple[ExplicitTPM]: Node-marginal distributions of the current state.
        """
        if not len(current_state) == self.number_of_units:
            raise ValueError(
                f"current_state must have length {self.number_of_units} "
                f"for state-by-node TPM of shape {self.shape}"
            )
        nodes = []
        for node in self.nodes:
            i = node.index
            state = current_state[i]
            # DataArray indexing: keep last dimension by wrapping index in list.
            pr_current_state = node.dataarray[..., [state]].data
            normalization = np.sum(pr_current_state)
            nodes.append(pr_current_state / normalization)
        return tuple(nodes)

    def backward_tpm(
            self,
            current_state: tuple[int],
            system_indices: Iterable[int],
    ):
        """Compute the backward TPM for a given network state."""
        all_indices = tuple(range(self.number_of_units))
        system_indices = tuple(sorted(system_indices))
        background_indices = tuple(sorted(set(all_indices) - set(system_indices)))
        if not set(system_indices).issubset(set(all_indices)):
            raise ValueError(
                "system_indices must be a subset of `range(self.number_of_units))`"
            )
        #                                                       p(u_t | s_{t–1}, w_{t–1})
        pr_current_state_nodes = self.probability_of_current_state(current_state)
        # TODO Avoid computing the full joint probability. Find uninformative
        # dimensions after each product and propagate their dismissal.
        pr_current_state = functools.reduce(np.multiply, pr_current_state_nodes)
        #                                           Σ_{s_{t–1}} p(u_t | s_{t–1}, w_{t–1})
        pr_current_state_given_only_background = pr_current_state.sum(
            axis=tuple(system_indices), keepdims=True
        )
        #                                           Σ_{s_{t–1}} p(u_t | s_{t–1}, w_{t–1})
        #                                           —————————————————————————————————————
        #                                           Σ_{u'_{t–1}} p(u_t | u'_{t–1})
        pr_current_state_given_only_background_normalized = (
            pr_current_state_given_only_background / np.sum(pr_current_state)
        )
        #                                           Σ_{s_{t–1}} p(u_t | s_{t–1}, w_{t–1})
        # Σ_{w_{t–1}} p(s_{i,t} | s_{t–1}, w_{t–1}) —————————————————————————————————————
        #                                           Σ_{u'_{t–1}} p(u_t | u'_{t–1})
        backward_tpm = tuple(
            (node_tpm * pr_current_state_given_only_background_normalized).sum(
                axis=background_indices, keepdims=True
            )
            for node_tpm in self.tpm
        )
        reference_node = self.nodes[0].dataarray
        return ImplicitTPM(
            tuple(
                Node(
                    backward_node_tpm,
                    reference_node.attrs["cm"],
                    reference_node.attrs["network_state_space"],
                    i,
                    reference_node.attrs["node_labels"],
                ).node
                for i, backward_node_tpm in enumerate(backward_tpm)
            )
        )

    def equals(self, o: object):
        """Return whether this TPM equals the other object.

        Two TPMs are equal if they are instances of the same class
        and their tuple of node TPMs are equal.
        """
        return isinstance(o, type(self)) and self.nodes == o.nodes

    def array_equal(self, o: object):
        return self.equals(o)

    def squeeze(self, axis=None):
        """Wrapper around numpy.squeeze."""
        # If axis is None, all axis should be considered.
        if axis is None:
            axis = set(range(len(self)))
        else:
            axis = set(axis) if isinstance(axis, Iterable) else set([axis])

        # Subtract non-singleton dimensions from `axis`, including fake
        # singletons (dimensions that are singletons only for a proper subset of
        # the nodes), since those should not be squeezed, not even within
        # individual node TPMs.
        shape = self._reconstituted_shape
        nonsingletons = tuple(np.where(np.array(shape) > 1)[0])
        axis = tuple(axis - set(nonsingletons))

        # From now on, we will only care about the first n-1 dimensions (parents).
        if shape[-1] > 1:
            nonsingletons = nonsingletons[:-1]

        # Recompute connectivity matrix and subset of node labels.
        # TODO(tpm) deduplicate commonalities with macro.MacroSubsystem._squeeze.
        some_node = self.nodes[0]

        new_cm = subadjacency(some_node.dataarray.attrs["cm"], nonsingletons)

        new_node_indices = iter(range(len(nonsingletons)))
        new_node_labels = tuple(some_node._node_labels[n] for n in nonsingletons)

        state_space = some_node.dataarray.attrs["network_state_space"]
        new_state_space = {n: state_space[n] for n in new_node_labels}

        # Leverage ExplicitTPM.squeeze to distribute squeezing to every node.
        return type(self)(
            tuple(
                Node(
                    node.tpm.squeeze(axis=axis),
                    new_cm,
                    new_state_space,
                    next(new_node_indices),
                    new_node_labels,
                ).node
                for node in self.nodes if node.index in nonsingletons
            )
        )

    def __getitem__(self, index, **kwargs):
        if isinstance(index, (int, slice, type(...), tuple)):
            return type(self)(
                tuple(
                    node.dataarray[node.project_index(index, **kwargs)].node
                    for node in self.nodes
                )
            )
        if isinstance(index, dict):
            return type(self)(
                tuple(
                    node.dataarray.loc[node.project_index(index, **kwargs)].node
                    for node in self.nodes
                )
            )
        raise TypeError(f"Invalid index {index} of type {type(index)}.")

    def __len__(self):
        """int: The number of nodes in the TPM."""
        return len(self._nodes)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "ImplicitTPM({})".format(self.nodes)

    def __hash__(self):
        return hash(tuple(hash(node) for node in self.nodes))



def reconstitute_tpm(subsystem):
    """Reconstitute the ExplicitTPM of a subsystem using individual node TPMs."""
    # The last axis of the node TPMs correponds to ON or OFF probabilities
    # (used in the conditioning step when calculating the repertoires); we want
    # ON probabilities.

    # TODO nonbinary nodes
    node_tpms = [np.asarray(node.tpm)[..., 1] for node in subsystem.nodes]

    external_indices = ()
    if hasattr(subsystem, "external_indices"):
        external_indices = subsystem.external_indices

    # Remove the singleton dimensions corresponding to external nodes
    node_tpms = [tpm.squeeze(axis=external_indices) for tpm in node_tpms]
    # We add a new singleton axis at the end so that we can use
    # pyphi.tpm.expand_tpm, which expects a state-by-node TPM (where the last
    # axis corresponds to nodes.)
    node_tpms = [np.expand_dims(tpm, -1) for tpm in node_tpms]
    # Now we expand the node TPMs to the full state space, so we can combine
    # them all (this uses the maximum entropy distribution).
    shapes = tuple(tpm.shape[:-1] for tpm in node_tpms)
    network_shape = tuple(max(dim) for dim in zip(*shapes))
    node_tpms = [
        tpm * np.ones(network_shape + (1,)) for tpm in node_tpms
    ]
    # We concatenate the node TPMs along a new axis to get a multidimensional
    # state-by-node TPM (where the last axis corresponds to nodes).
    return ExplicitTPM(np.concatenate(node_tpms, axis=-1))


# TODO(tpm) remove pending ArrayLike refactor
def _new_attribute(
    name: str,
    closures: Set[str],
    tpm: np.ndarray,
    cls=ExplicitTPM
) -> object:
    """Helper function to return adequate proxy attributes for TPM arrays.

    Args:
        name (str): The name of the attribute to  proxy.
        closures (set[str]): Attribute names which should return a PyPhi TPM.
        tpm (np.ndarray): The array to introspect for attributes.
        cls (type): The TPM type that the proxied attribute should return.

    Returns:
        object: A proxy to the underlying array's attribute, whether unmodified
            or decorated with casting to `cls`.
    """
    attribute = getattr(tpm, name)

    if name not in closures:
        return attribute

    def overriding_attribute(*args, **kwargs):
        # If second operand is a custom TPM object, access its array.
        if args and isinstance(args[0], cls):
            args = list(args)
            args[0] = args[0]._tpm
            args = tuple(args)

        # Evaluate n-ary operator with self and rest of operands.
        result = attribute(*args, **kwargs)

        # Test type of result and cast (or not) accordingly.

        # Array.
        if isinstance(result, np.ndarray):
            return cls(result)

        # Multivalued "functions" returning a tuple (__divmod__()).
        if isinstance(result, tuple):
            return (cls(r) for r in result)

        # Scalars (e.g. sum(), max()), etc.
        return result

    try:
        # TODO search and replace return type.
        overriding_attribute.__doc__ = attribute.__doc__
    except AttributeError:
        pass

    return overriding_attribute

