"""Per-node-factored conditional TPM.

Represents the joint conditional ``P(s_{t+1} | s_t)`` as a product of N
per-node conditional marginals ``P(s_{i,t+1} | s_t)``. The joint is the
product of the factors under conditional independence (IIT's standing
assumption that nodes update independently given the joint past).

Factor ``i`` has shape ``(a_1, ..., a_N, a_i)`` where ``a_j`` is the
alphabet size of node ``j``. Input dims for non-input nodes are size 1
and are semantically load-bearing — they encode the connectivity
structure and are never squeezed away.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import exceptions
from pyphi.conf import config

from ._factored_backends import _make_default_backend
from ._factored_backends import _NdarrayBackend

# Set from the storage-backend benchmark result.
_FACTORED_TPM_DEFAULT_BACKEND: Literal["ndarray", "xarray"] = "ndarray"


class FactoredTPM:
    """Per-node-factored conditional TPM."""

    __slots__ = ("_alphabet_sizes", "_backend")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        alphabet_sizes: Sequence[int] | None = None,
        backend: Literal["ndarray", "xarray"] | None = None,
    ) -> None:
        factor_arrays = tuple(np.asarray(f, dtype=np.float64) for f in factors)
        if alphabet_sizes is None:
            alphabet_sizes = tuple(int(f.shape[-1]) for f in factor_arrays)
        else:
            alphabet_sizes = tuple(int(a) for a in alphabet_sizes)
        self._alphabet_sizes = alphabet_sizes
        self._backend = _make_default_backend(factor_arrays, alphabet_sizes, backend)
        _validate(self)

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self._alphabet_sizes, self.n_nodes)

    @property
    def n_nodes(self) -> int:
        return self._backend.n_factors()

    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        return self._alphabet_sizes

    @property
    def factors(self) -> tuple[NDArray[np.float64], ...]:
        return self._backend.all_factors()

    def factor(self, i: int) -> NDArray[np.float64]:
        return self._backend.get_factor(i)

    @classmethod
    def from_joint(
        cls,
        joint: ArrayLike,
        /,
        alphabet_sizes: Sequence[int] | None = None,
    ) -> FactoredTPM:
        """Convert a joint conditional TPM into the factored form.

        Accepts either:

        - Legacy binary form: shape ``(2,) * n + (n,)``, where the last
          dim's entry ``i`` is ``P(node_i = 1 | s_t)``. Factor ``i`` is
          built by stacking ``[1 - p_on, p_on]`` along an explicit
          alphabet dim.

        - Explicit-alphabet form: shape ``(a_1, ..., a_N, N, a_i)``.
          Factor ``i`` is ``joint[..., i, :]``.

        ``alphabet_sizes`` defaults to ``(2,) * n`` for the legacy form;
        for the explicit form it must be supplied and must match the
        per-row last-dim shapes.
        """
        joint_arr = np.asarray(joint, dtype=np.float64)
        ndim = joint_arr.ndim
        if alphabet_sizes is None:
            if ndim < 2 or joint_arr.shape[-1] != ndim - 1:
                raise ValueError(
                    f"Cannot infer alphabet_sizes from joint shape "
                    f"{joint_arr.shape}; expected legacy form "
                    f"(2,)*n + (n,) or pass alphabet_sizes explicitly."
                )
            n = ndim - 1
            alphabet_sizes = (2,) * n
        else:
            alphabet_sizes = tuple(int(a) for a in alphabet_sizes)
            n = len(alphabet_sizes)

        if joint_arr.shape[:-1] != alphabet_sizes:
            # Explicit-alphabet form: (a_1, ..., a_N, N, a_max)
            if (
                ndim == n + 2
                and joint_arr.shape[:n] == alphabet_sizes
                and joint_arr.shape[n] == n
            ):
                factors = tuple(joint_arr[..., i, : alphabet_sizes[i]] for i in range(n))
                return cls(factors=factors, alphabet_sizes=alphabet_sizes)
            raise ValueError(
                f"Joint shape {joint_arr.shape} not consistent with "
                f"alphabet_sizes {alphabet_sizes}."
            )

        if joint_arr.shape[-1] != n:
            raise ValueError(
                f"Legacy joint shape requires last dim == n_nodes={n}; "
                f"got shape {joint_arr.shape}."
            )
        if alphabet_sizes != (2,) * n:
            raise ValueError(
                f"Legacy joint form is binary-only; "
                f"alphabet_sizes={alphabet_sizes} requires explicit-form joint."
            )

        factors_list: list[NDArray[np.float64]] = []
        for i in range(n):
            p_on = joint_arr[..., i]
            factor_i = np.stack([1.0 - p_on, p_on], axis=-1)
            factors_list.append(factor_i)
        return cls(factors=tuple(factors_list), alphabet_sizes=alphabet_sizes)

    def tpm_indices(self) -> tuple[int, ...]:
        """Substrate-unit indices: one entry per output unit (the leading
        factor axes); the trailing alphabet axis is per-unit.
        """
        return tuple(range(self.n_nodes))

    def condition(self, fixed: Mapping[int, int]) -> FactoredTPM:
        conditioned = [self._backend.select(i, fixed) for i in range(self.n_nodes)]
        return FactoredTPM(
            factors=conditioned,
            alphabet_sizes=self._alphabet_sizes,
            backend=None,
        )

    def condition_factor(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        return self._backend.select(i, fixed)

    def to_array(self) -> NDArray[np.float64]:
        return self.to_joint()

    def to_joint(self) -> NDArray[np.float64]:
        """Materialize the joint conditional ``P(s_{t+1} | s_t)`` from the factors.

        Output shape is ``alphabet_sizes + (n_nodes, max_alphabet)``: the
        per-row last dim holds factor ``i``'s distribution in slots
        ``[:alphabet_sizes[i]]``; trailing slots are zero when alphabets are
        heterogeneous. For uniform alphabets this collapses to
        ``(a, ..., a, n, a)`` with no padding. Slow path — only used at
        boundaries (serialization, legacy fixture comparison,
        ``Substrate.joint_tpm()``).
        """
        n = self.n_nodes
        max_alphabet = max(self._alphabet_sizes)
        shape = (*self._alphabet_sizes, n, max_alphabet)
        out = np.zeros(shape, dtype=np.float64)
        for i in range(n):
            factor = self.factor(i)
            a_i = self._alphabet_sizes[i]
            broadcast_shape = (*self._alphabet_sizes, a_i)
            out[..., i, :a_i] = np.broadcast_to(factor, broadcast_shape)
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FactoredTPM):
            return NotImplemented
        if self._alphabet_sizes != other._alphabet_sizes:
            return False
        if self.n_nodes != other.n_nodes:
            return False
        return all(
            np.array_equal(self.factor(i), other.factor(i)) for i in range(self.n_nodes)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._alphabet_sizes,
                tuple((self.factor(i) + 0.0).tobytes() for i in range(self.n_nodes)),
            )
        )

    def __repr__(self) -> str:
        return (
            f"FactoredTPM(n_nodes={self.n_nodes}, alphabet_sizes={self._alphabet_sizes})"
        )

    def __reduce__(self) -> tuple:  # type: ignore[override]
        backend_name = (
            "ndarray" if isinstance(self._backend, _NdarrayBackend) else "xarray"
        )
        return (
            _factored_tpm_from_pickle,
            (tuple(self.factors), self._alphabet_sizes, backend_name),
        )


def _factored_tpm_from_pickle(
    factors: tuple,  # type: ignore[type-arg]
    alphabet_sizes: tuple,  # type: ignore[type-arg]
    backend: str,
) -> FactoredTPM:
    return FactoredTPM(factors=factors, alphabet_sizes=alphabet_sizes, backend=backend)  # type: ignore[arg-type]


def _validate(factored: FactoredTPM) -> None:
    """Validate a freshly constructed FactoredTPM."""
    a = factored.alphabet_sizes
    if factored.n_nodes != len(a):
        raise exceptions.InvalidTPM(
            f"n_nodes={factored.n_nodes} does not match alphabet_sizes length {len(a)}"
        )
    if any(size < 2 for size in a):
        raise exceptions.InvalidTPM(f"alphabet_sizes must all be >= 2; got {a}")
    tol = max(10 ** (-config.numerics.precision), 1e-15)
    for i in range(factored.n_nodes):
        f = factored.factor(i)
        if f.shape[-1] != a[i]:
            raise exceptions.InvalidTPM(
                f"factor {i} last-dim size {f.shape[-1]} != alphabet_sizes[{i}]={a[i]}"
            )
        for j, dim_size in enumerate(f.shape[:-1]):
            if dim_size not in (1, a[j]):
                raise exceptions.InvalidTPM(
                    f"factor {i} input dim {j} has size {dim_size}; "
                    f"expected 1 (non-input) or {a[j]} (input)"
                )
        sums = f.sum(axis=-1)
        if not np.allclose(sums, 1.0, atol=tol):
            if sums.ndim == 0:
                raise exceptions.InvalidTPM(
                    f"factor {i} sums to 1 violated: got {sums.item()}, tolerance {tol}"
                )
            worst = np.unravel_index(np.abs(sums - 1.0).argmax(), sums.shape)
            raise exceptions.InvalidTPM(
                f"factor {i} sums to 1 violated at input state {worst}: "
                f"got {sums[worst]}, tolerance {tol}"
            )
