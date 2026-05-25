"""Joint distribution base class — multidimensional joint probability storage.

Shared base for typed wrappers around multidimensional ndarrays representing
joint probability distributions over substrate state spaces. Concrete
subclasses:

- :class:`pyphi.JointTPM` — joint conditional ``P(s_{t+1} | s_t)`` with
  TPM-specific affordances.
- :class:`pyphi.CausePosterior` — joint posterior over past states given an
  observed mechanism state.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import convert
from pyphi import data_structures
from pyphi.utils import all_states
from pyphi.utils import np_hash
from pyphi.utils import np_immutable


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
        """Return whether this distribution equals the other object.

        Two distributions are equal if they are instances of the same class
        and their numpy arrays are equal.
        """
        return isinstance(o, type(self)) and np.array_equal(self._tpm, o._tpm)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tpm})"

    def __hash__(self) -> int:
        return self._hash
