#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tpm.py

"""
Provides the ExplicitTPM and related classes.
"""

from itertools import chain
from typing import Iterable, Mapping, Set, Tuple

import numpy as np

from . import config, convert, data_structures, exceptions
from .constants import OFF, ON
from .data_structures import FrozenMap
from .node import node as Node
from .utils import all_states, np_hash, np_immutable

class TPM:
    """TPM interface for derived classes."""

    _ERROR_MSG_PROBABILITY_IMAGE = (
        "Invalid TPM: probabilities must be in the interval [0, 1]."
    )

    _ERROR_MSG_PROBABILITY_SUM = "Invalid TPM: probabilities must sum to 1."

    def validate(self, cm, check_independence=True):
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

    def _subtpm(self, fixed_nodes, state):
        N = self.shape[-1]
        free_nodes = sorted(set(range(N)) - set(fixed_nodes))
        condition = FrozenMap(zip(fixed_nodes, state))
        conditioned = self.condition_tpm(condition)
        return conditioned, free_nodes

    def expand_tpm(self):
        raise NotImplementedError

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

    def tpm_indices(self):
        """Return the indices of nodes in the TPM."""
        return tuple(np.where(np.array(self.shape[:-1]) != 1)[0])

    def print(self):
        raise NotImplementedError

    def permute_nodes(self, permutation):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError


# TODO(tpm) remove pending ArrayLike refactor
class ProxyMetaclass(type):
    """A metaclass to create wrappers for the TPM array's special attributes.

    The CPython interpreter resolves double-underscore attributes (e.g., the
    method definitions of mathematical operators) by looking up in the class'
    static methods, not in the instance methods. This makes it impossible to
    intercept calls to them when an instance's ``__getattr__()`` is implicitly
    invoked, which in turn means there are only two options to wrap the special
    methods of the array inside our custom objects (in order to perform
    arithmetic operations with the TPM while also casting the result to our
    custom class type):

    1. Manually "overload" all the necessary methods.
    2. Use this metaclass to introspect the underlying array
       and automatically overload methods in our custom TPM class definition.
    """

    def __init__(cls, type_name, bases, dct):

        # Casting semantics: values belonging to our custom TPM class should
        # remain closed under the following methods:
        __closures__ = frozenset(
            {
                # 1-ary
                "__abs__",
                "__copy__",
                "__invert__",
                "__neg__",
                "__pos__",
                # 2-ary
                "__add__",
                "__iadd__",
                "__radd__",
                "__sub__",
                "__isub__",
                "__rsub__",
                "__mul__",
                "__imul__",
                "__rmul__",
                "__matmul__",
                "__imatmul__",
                "__rmatmul__",
                "__truediv__",
                "__itruediv__",
                "__rtruediv__",
                "__floordiv__",
                "__ifloordiv__",
                "__rfloordiv__",
                "__mod__",
                "__imod__",
                "__rmod__",
                "__and__",
                "__iand__",
                "__rand__",
                "__lshift__",
                "__ilshift__",
                "__irshift__",
                "__rlshift__",
                "__rrshift__",
                "__rshift__",
                "__ior__",
                "__or__",
                "__ror__",
                "__xor__",
                "__ixor__",
                "__rxor__",
                "__eq__",
                "__ne__",
                "__ge__",
                "__gt__",
                "__lt__",
                "__le__",
                "__deepcopy__",
                # 3-ary
                "__pow__",
                "__ipow__",
                "__rpow__",
                # 2-ary, 2-valued
                "__divmod__",
                "__rdivmod__",
            }
        )

        def make_proxy(name):
            """Returns a function that acts as a proxy for the given method name.

            Args:
                name (str): The name of the method to introspect in self._tpm.

            Returns:
                function: The wrapping function.
            """

            def proxy(self):
                return _new_attribute(name, __closures__, self._tpm)

            return proxy

        type.__init__(cls, type_name, bases, dct)

        if not cls.__wraps__:
            return

        ignore = cls.__ignore__

        # Go through all the attribute strings in the wrapped array type.
        for name in dir(cls.__wraps__):
            # Filter special attributes, rest will be handled by `__getattr__()`
            if any((not name.startswith("__"), name in ignore, name in dct)):
                continue

            # Create function for `name` and bind to future instances of `cls`.
            setattr(cls, name, property(make_proxy(name)))


class Wrapper(metaclass=ProxyMetaclass):
    """Proxy to the array inside PyPhi's custom ExplicitTPM class."""

    __wraps__ = None

    __ignore__ = frozenset(
        {
            "__class__",
            "__mro__",
            "__new__",
            "__init__",
            "__setattr__",
            "__getattr__",
            "__getattribute__",
        }
    )

    def __init__(self):
        if self.__wraps__ is None:
            raise TypeError("Base class Wrapper may not be instantiated.")

        if not isinstance(self._tpm, self.__wraps__):
            raise ValueError(f"Wrapped object must be of type {self.__wraps__}")


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

    def __init__(self, tpm, validate=False):
        self._tpm = np.array(tpm)

        if validate:
            self.validate(check_independence=config.VALIDATE_CONDITIONAL_INDEPENDENCE)
            self._tpm = self.to_multidimensional_state_by_node()

        self._tpm = np_immutable(self._tpm)
        self._hash = np_hash(self._tpm)

    @property
    def tpm(self):
        """np.ndarray: The underlying `tpm` object."""
        return self._tpm

    def validate(self, cm=None, check_independence=True):
        """Validate this TPM."""
        return self._validate_probabilities() and self._validate_shape(
            check_independence
        )

    def _validate_probabilities(self):
        """Check that the probabilities in a TPM are valid."""
        if (self._tpm < 0.0).any() or (self._tpm > 1.0).any():
            raise ValueError(self._ERROR_MSG_PROBABILITY_IMAGE)
        if self.is_state_by_state() and np.any(np.sum(self._tpm, axis=1) != 1.0):
            raise ValueError(self._ERROR_MSG_PROBABILITY_SUM)
        return True

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
        tpm = self[conditioning_indices]
        # Create new TPM object of the same type as self.
        # self.tpm has already been validated and converted to multidimensional
        # state-by-node form. Further validation would be problematic for
        # singleton dimensions.
        return tpm

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
        conditioned, free_nodes = self._subtpm(fixed_nodes, state)
        return conditioned[..., free_nodes]

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
    def ndim(self):
        """int: The number of dimensions of the TPM."""
        return len(self.shape)

    @property
    def shape(self):
        """Tuple[int]: The size or number of coordinates in each dimension."""
        shapes = self.shapes
        return self._node_shapes_to_shape(shapes)

    @property
    def shapes(self):
        """Tuple[Tuple[int]]: The shapes of each node TPM in this TPM."""
        return [node.tpm.shape for node in self._nodes]

    @staticmethod
    def _node_shapes_to_shape(shapes: Iterable[Iterable[int]]) -> Tuple[int]:
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

    def validate(self, cm=None, check_independence=True):
        """Validate this TPM."""
        return self._validate_probabilities() and self._validate_shape(cm)

    def _validate_probabilities(self):
        """Check that the probabilities in a TPM are valid."""
        # An implicit TPM contains valid probabilities if and only if
        # individual node TPMs contain valid probabilities, for every node.

        # Validate that probabilities sum to 1.
        if any(
                (node.tpm.sum(axis=-1) != 1.0).any()
                for node in self._nodes
        ):
            raise ValueError(self._ERROR_MSG_PROBABILITY_SUM)

        # Leverage method in ExplicitTPM to distribute validation of
        # TPM image within [0, 1].
        if all(
                node.tpm._validate_probabilities()
                for node in self._nodes
        ):
            return True

    def _validate_shape(self, cm):
        """Validate this TPM's shape.

        The shapes of the individual node TPMs in multidimensional form are
        validated against the connectivity matrix specification. Additionally,
        the inferred shape of the implicit network TPM must be in
        multidimensional state-by-node form, nonbinary and heterogeneous units
        supported.
        """
        # Validate individual node TPM shapes.
        shapes = self.shapes

        for i, shape in enumerate(shapes):
            for j, val in enumerate(cm[..., i]):
                if (val == 0 and shape[j] != 1) or (val != 0 and shape[j] == 1):
                    raise ValueError(
                        "Node TPM {} of shape {} does not match the connectivity "
                        " matrix.".format(i, shape)
                    )

        # Validate whole network's shape.
        N = len(self.nodes)
        if N + 1 != self.ndim:
            raise ValueError(
                "Invalid TPM shape: {} nodes were provided, but their shapes"
                "suggest a {}-node network.".format(N, self.ndim - 1)
            )

    def to_multidimensional_state_by_node(self):
        """Return the current TPM re-represented in multidimensional
        state-by-node form.

        See the PyPhi documentation on :ref:`tpm-conventions` for more
        information.

        Returns:
            np.ndarray: The TPM in multidimensional state-by-node format.
        """
        return reconstitute_tpm(self)

    def conditionally_independent(self):
        raise NotImplementedError

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
                ).pyphi
                for node in self.nodes
            )
        )

    def is_deterministic(self):
        raise NotImplementedError

    def is_state_by_state(self):
        """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
        ``False``.
        """
        return False

    def subtpm(self, fixed_nodes, state):
        conditioned, free_nodes = self._subtpm(fixed_nodes, state)
        return type(self)(
            tuple(node for node in conditioned.nodes if node.index in free_nodes)
        )

    def expand_tpm(self):
        raise NotImplementedError

    def print(self):
        raise NotImplementedError

    def permute_nodes(self, permutation):
        raise NotImplementedError

    def equals(self, o: object):
        return isinstance(o, type(self)) and self.nodes == o.nodes

    def __getitem__(self, index, **kwargs):
        if isinstance(index, (int, slice, type(...), tuple)):
            return ImplicitTPM(
                tuple(
                    node.dataarray[node.project_index(index, **kwargs)].pyphi
                    for node in self.nodes
                )
            )
        if isinstance(index, dict):
            return ImplicitTPM(
                tuple(
                    node.dataarray.loc[node.project_index(index, **kwargs)].pyphi
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
        raise NotImplementedError


def reconstitute_tpm(subsystem):
    """Reconstitute the TPM of a subsystem using the individual node TPMs."""
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
    tpm: ExplicitTPM.__wraps__,
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
        if isinstance(result, cls.__wraps__):
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
