"""Joint distribution base class — multidimensional joint probability storage.

Shared base for typed wrappers around multidimensional ndarrays representing
joint probability distributions over substrate state spaces. The concrete
subclass in production use is :class:`pyphi.JointTPM`, which wraps
``P(s_{t+1} | s_t)`` with TPM-specific affordances.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from collections.abc import Mapping
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import convert
from pyphi import data_structures
from pyphi import exceptions
from pyphi.conf import config
from pyphi.constants import OFF
from pyphi.constants import ON
from pyphi.data_structures import FrozenMap
from pyphi.display import Displayable
from pyphi.utils import all_states
from pyphi.utils import np_hash
from pyphi.utils import np_immutable

from . import _display

if TYPE_CHECKING:
    from pyphi.display import Description


# TODO(tpm) remove pending ArrayLike refactor
def _new_attribute(
    name: str,
    closures: set[str] | frozenset[str],
    tpm: NDArray[np.float64],
    cls: type,
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

    with contextlib.suppress(AttributeError):
        # TODO search and replace return type.
        overriding_attribute.__doc__ = attribute.__doc__

    return overriding_attribute


class JointDistribution(data_structures.ArrayLike):
    """Multidimensional joint probability distribution over a state space.

    Stores and operates on a joint probability distribution as a multidimensional
    ndarray. Subclasses add semantics (e.g., conditional TPM or cause posterior).
    """

    _VALUE_ATTR = "_tpm"

    # TODO(tpm) remove pending ArrayLike refactor
    __wraps__ = np.ndarray

    # TODO(tpm) remove pending ArrayLike refactor
    # Casting semantics: values belonging to our custom TPM class should
    # remain closed under the following methods:

    # TODO attributes data, real and imag return arrays that should also be
    # cast, even though they are not callable.
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

    def __getattr__(self, name: str) -> Any:
        if name in self.__closures__:
            return _new_attribute(name, set(self.__closures__), self._tpm, type(self))
        return getattr(self.__getattribute__(self._VALUE_ATTR), name)

    def __len__(self) -> int:
        return len(self.__getattribute__(self._VALUE_ATTR))

    def __init__(self, tpm: ArrayLike, validate: bool = False) -> None:
        self._tpm = np.array(tpm)
        super().__init__()

        if validate:
            self.validate()
            self._tpm = self.to_multidimensional_state_by_node()

        self._tpm = np_immutable(self._tpm)
        self._hash = np_hash(self._tpm)

    @property
    def tpm(self) -> NDArray[np.float64]:
        """Return the underlying `tpm` object."""
        return self._tpm

    def _validate_probabilities(self) -> bool:
        """Check that the probabilities in a TPM are valid."""
        if (self._tpm < 0.0).any() or (self._tpm > 1.0).any():
            raise ValueError(
                "Invalid TPM: probabilities must be in the interval [0, 1]."
            )
        if self.is_state_by_state() and not np.all(
            np.isclose(np.sum(self._tpm, axis=1), 1.0, atol=1e-15)
        ):
            raise ValueError("Invalid TPM: probabilities must sum to 1.")
        return True

    def validate(self, check_independence: bool = False) -> bool:  # noqa: ARG002
        """Validate this distribution.

        Checks that all probability values are in [0, 1].
        """
        return self._validate_probabilities()

    @property
    def number_of_units(self) -> int:
        if self.is_state_by_state():
            # Assumes binary nodes
            return int(math.log2(self._tpm.shape[1]))
        return self._tpm.shape[-1]

    def to_multidimensional_state_by_node(self) -> NDArray[np.float64]:
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

    def marginalize_out(self, node_indices: Iterable[int]) -> JointDistribution:
        """Marginalize out nodes from this TPM.

        Args:
            node_indices (list[int]): The indices of nodes to be marginalized out.

        Returns:
            JointDistribution: A distribution with the same number of dimensions,
            with the nodes marginalized out.
        """
        tpm = self._tpm.sum(tuple(node_indices), keepdims=True) / (
            np.array(self.shape)[list(node_indices)].prod()
        )
        # Return new object of the same type as self.
        # self._tpm has already been validated and converted to multidimensional
        # state-by-node form. Further validation would be problematic for
        # singleton dimensions.
        return type(self)(tpm)

    def is_deterministic(self) -> bool:
        """Return whether the TPM is deterministic."""
        return bool(np.all(np.logical_or(self._tpm == 1, self._tpm == 0)))

    def is_state_by_state(self) -> bool:
        """Return ``True`` if ``tpm`` is in state-by-state form, otherwise
        ``False``.
        """
        return self.ndim == 2 and self.shape[0] == self.shape[1]

    def tpm_indices(self) -> tuple[int, ...]:
        """Return the substrate-unit axis indices for this distribution.

        Subclasses must override this method; the base class carries no
        semantic axis labels.
        """
        raise NotImplementedError(f"{type(self).__name__} must override tpm_indices()")

    def print(self) -> None:
        tpm = convert.to_multidimensional(self._tpm)
        for _state in all_states(tpm.shape[-1]):
            pass

    # TODO(4.0) docstring
    def permute_nodes(self, permutation: tuple[int, ...]) -> JointDistribution:
        if not len(permutation) == self.ndim - 1:
            raise ValueError(
                f"Permutation must have length {self.ndim - 1}, but has length "
                f"{len(permutation)}."
            )
        dimension_permutation = (*tuple(permutation), self.ndim - 1)
        return type(self)(
            self._tpm.transpose(dimension_permutation)[..., list(permutation)],
        )

    def __getitem__(
        self, i: int | slice | tuple[Any, ...] | Any
    ) -> JointDistribution | Any:
        item: Any = self._tpm[i]
        if isinstance(item, type(self._tpm)):
            item = type(self)(item)
        return item

    def array_equal(self, o: object) -> bool:
        """Return whether this distribution equals another, numerically.

        Compares the underlying numpy arrays; accepts any array-compatible
        object (including ``core.tpm.joint.JointTPM`` wrappers).
        """
        return np.array_equal(self._tpm, np.asarray(o))

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tpm})"

    def __hash__(self) -> int:
        return self._hash


class JointTPM(Displayable, JointDistribution):
    """A substrate TPM storing the full joint transition distribution."""

    def __init__(self, tpm: ArrayLike, validate: bool = False) -> None:
        self._tpm = np.array(tpm)
        super(JointDistribution, self).__init__()

        if validate:
            self.validate(
                check_independence=config.infrastructure.validate_conditional_independence
            )
            self._tpm = self.to_multidimensional_state_by_node()

        self._tpm = np_immutable(self._tpm)
        self._hash = np_hash(self._tpm)

    def validate(self, check_independence: bool = True) -> bool:
        """Validate this TPM."""
        return self._validate_probabilities() and self._validate_shape(
            check_independence
        )

    def _validate_shape(self, check_independence: bool = True) -> bool:
        """Validate this TPM's shape.

        The TPM can be in

            * 2-dimensional state-by-state form,
            * 2-dimensional state-by-node form, or
            * multidimensional state-by-node form.
        """
        see_tpm_docs = (
            "See the documentation on TPM conventions and the `pyphi.Substrate` "
            "object for more information on TPM forms."
        )
        tpm = self._tpm
        # Get the number of nodes from the state-by-node TPM.
        N = tpm.shape[-1]
        if tpm.ndim == 2:
            if not (
                (tpm.shape[0] == 2**N and tpm.shape[1] == N)
                or (tpm.shape[0] == tpm.shape[1])
            ):
                raise ValueError(
                    f"Invalid shape for 2-D TPM: {tpm.shape}\nFor a state-by-node TPM, "
                    "there must be "
                    "2^N rows and N columns, where N is the "
                    "number of nodes. State-by-state TPM must be square. "
                    f"{see_tpm_docs}"
                )
            if tpm.shape[0] == tpm.shape[1] and check_independence:
                self.conditionally_independent()
        elif tpm.ndim == (N + 1):
            if tpm.shape != tuple([2] * N + [N]):
                raise ValueError(
                    f"Invalid shape for multidimensional state-by-node TPM: "
                    f"{tpm.shape}\n"
                    f"The shape should be {([2] * N) + [N]} for {N} nodes. "
                    f"{see_tpm_docs}"
                )
        else:
            raise ValueError(
                "Invalid TPM: Must be either 2-dimensional or multidimensional. "
                f"{see_tpm_docs}"
            )
        return True

    def conditionally_independent(self) -> bool:
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

    def condition_tpm(self, condition: Mapping[int, int]) -> JointTPM:
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
        conditioning_indices: Any = [[slice(None)]] * (self.ndim - 1)
        for i, state_i in condition.items():
            # Ignore dimensions that are already singletons
            if self.shape[i] != 1:
                # Preserve singleton dimensions in output array with `np.newaxis`
                conditioning_indices[i] = [state_i, np.newaxis]
        # Flatten the indices.
        conditioning_indices = tuple(chain.from_iterable(conditioning_indices))
        # Obtain the actual conditioned TPM by indexing with the conditioning
        # indices.
        tpm = self._tpm[conditioning_indices]
        # Create new TPM object of the same type as self.
        # self.tpm has already been validated and converted to multidimensional
        # state-by-node form. Further validation would be problematic for
        # singleton dimensions.
        return type(self)(tpm)

    def subtpm(self, fixed_nodes: tuple[int, ...], state: tuple[int, ...]) -> JointTPM:
        """Return the TPM for a subset of nodes, conditioned on other nodes.

        Arguments:
            fixed_nodes (tuple[int]): The nodes to select.
            state (tuple[int]): The state of the fixed nodes.

        Returns:
            JointTPM: The TPM of just the system of the free nodes.

        Examples:
            >>> from pyphi import JointTPM, examples
            >>> # Get the TPM for nodes only 1 and 2, conditioned on node 0 = OFF
            >>> sub = examples.grid3_substrate()
            >>> result = JointTPM(sub._legacy_binary_joint()).subtpm((0,), (0,))
            >>> result.shape
            (1, 2, 2, 2)
            >>> result.number_of_units
            2
        """
        free_nodes = sorted(set(range(self.number_of_units)) - set(fixed_nodes))
        condition: Mapping[int, int] = FrozenMap(zip(fixed_nodes, state, strict=False))
        conditioned = self.condition_tpm(condition)
        # TODO test indicing behavior on xr.DataArray
        result = conditioned[..., free_nodes]
        return result  # type: ignore[return-value]

    def tpm_indices(self) -> tuple[int, ...]:
        """Binary-only substrate-unit axis indices via size-2 grep.

        Returns leading axes whose size equals 2, excluding singleton (size-1)
        axes that arise from marginalizing out non-input nodes, so the
        indices stay aligned with a ``squeeze()``-d view of the array.
        Multi-valued substrates use :meth:`FactoredTPM.tpm_indices`
        instead.
        """
        return tuple(np.where(np.array(self.shape[:-1]) == 2)[0])

    def expand_tpm(self) -> JointTPM:
        """Broadcast a state-by-node TPM so that singleton dimensions are expanded
        over the full substrate.
        """
        unconstrained = np.ones([2] * (self._tpm.ndim - 1) + [self._tpm.shape[-1]])
        return type(self)(self._tpm * unconstrained)

    def infer_edge(self, a: int, b: int, contexts: tuple[tuple[int, ...], ...]) -> bool:
        """Infer the presence or absence of an edge from node A to node B.

        Let |S| be the set of all nodes in a substrate. Let |A' = S - {A}|. We
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

    def infer_cm(self) -> NDArray[np.int_]:
        """Infer the connectivity matrix associated with a state-by-node TPM in
        multidimensional form.
        """
        tpm = self.to_multidimensional_state_by_node()
        substrate_size = tpm.shape[-1]
        all_contexts = tuple(all_states(substrate_size - 1))
        cm = np.empty((substrate_size, substrate_size), dtype=int)
        for a, b in np.ndindex(cm.shape):
            cm[a][b] = self.infer_edge(a, b, all_contexts)
        return cm

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        multidim = self.to_multidimensional_state_by_node()
        n = int(multidim.shape[-1])
        axis_sizes = multidim.shape[:-1]
        total = int(math.prod(axis_sizes)) if axis_sizes else 1
        return _display.state_by_node_description(
            title="JointTPM",
            compact=f"JointTPM({n} units, {total} states)",
            unit_labels=[str(i) for i in range(n)],
            state_axis_sizes=axis_sizes,
            prob_on_for_state=lambda state: multidim[state],
        )

    def to_xarray(self) -> Any:
        """Return this TPM as a labeled :class:`xarray.DataArray`.

        Dims are ``("u0", ..., "u{N-1}", "next_unit")``: the leading axes index
        each unit's current state and the trailing ``next_unit`` axis selects
        the output unit, with values ``P(next unit on | current state)``.
        Requires the optional ``xarray`` dependency.
        """
        xr = _display.require_xarray()
        multidim = self.to_multidimensional_state_by_node()
        n = int(multidim.shape[-1])
        in_dims = tuple(f"u{j}" for j in range(n))
        coords: dict[str, list[int]] = {
            in_dims[j]: list(range(int(multidim.shape[j]))) for j in range(n)
        }
        coords["next_unit"] = list(range(n))
        return xr.DataArray(
            multidim,
            dims=(*in_dims, "next_unit"),
            coords=coords,
            name="P(next unit on)",
        )


def reconstitute_tpm(system: Any) -> NDArray[np.float64]:
    """Reconstitute the TPM of a system using the individual node TPMs."""
    # The last axis of the node TPMs correponds to ON or OFF probabilities
    # (used in the conditioning step when calculating the repertoires); we want
    # ON probabilities.
    node_tpms = [node.effect_marginal.tpm[..., 1] for node in system.nodes]
    # Remove the singleton dimensions corresponding to external nodes
    node_tpms = [tpm.squeeze(axis=system.external_indices) for tpm in node_tpms]
    # We add a new singleton axis at the end so that we can use
    # expand_tpm, which expects a state-by-node TPM (where the last
    # axis corresponds to nodes.)
    node_tpms = [np.expand_dims(tpm, -1) for tpm in node_tpms]
    # Now we expand the node TPMs to the full state space, so we can combine
    # them all (this uses the maximum entropy distribution).
    node_tpms = [
        tpm * np.ones([2] * (tpm.ndim - 1) + [tpm.shape[-1]]) for tpm in node_tpms
    ]
    # We concatenate the node TPMs along a new axis to get a multidimensional
    # state-by-node TPM (where the last axis corresponds to nodes).
    return np.concatenate(node_tpms, axis=-1)


def simulate(
    tpm: JointTPM | ArrayLike,
    initial_state: int,
    timesteps: int,
    rng: np.random.Generator,
) -> list[int]:
    """Simulate the dynamics of a system.

    Generates a sequence of states using the TPM and a random number generator.

    Arguments:
        tpm (np.ndarray): TPM to simulate.
        initial_state (int): The initial state of the simulation.
        timesteps (int): The number of timesteps to simulate.
        rng (np.random.Generator): The random number generator to use.

    Returns:
        list: a list of (decimally-indexed) states.
    """
    # Ensure tpm is a JointTPM
    if not isinstance(tpm, JointTPM):
        tpm = JointTPM(tpm)

    if not tpm.is_state_by_state():
        raise ValueError("TPM must be in state-by-state form.")
    # Get the conditional cumulative distributions
    # Use .tpm to get the underlying numpy array
    cumulative_tpm = np.cumsum(tpm.tpm, axis=1)
    # We include the initial state so there are ``timesteps`` total in the
    # output
    timesteps = int(timesteps - 1)
    # Get a random draw for each timestep
    draws = rng.random(timesteps)
    # Initialize the state trajectory
    path = [initial_state]
    for draw in draws:
        # The next state is the first one whose cumulative probability beats
        # the random draw
        path.append(
            next(
                state
                for state, cumulative_probability in enumerate(cumulative_tpm[path[-1]])
                if cumulative_probability > draw
            )
        )
    return path
