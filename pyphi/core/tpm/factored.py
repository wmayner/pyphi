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
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import exceptions
from pyphi.conf import config
from pyphi.display import Displayable

from . import _display
from ._factored_backends import _make_default_backend
from ._factored_backends import _NdarrayBackend

if TYPE_CHECKING:
    from pyphi.display import Description

# Set from the storage-backend benchmark result.
_FACTORED_TPM_DEFAULT_BACKEND: Literal["ndarray", "xarray"] = "ndarray"

# Type alias for the state_space argument.
StateSpace = Sequence[Any] | Sequence[Sequence[Any]] | None


def _normalize_state_space(
    raw: StateSpace,
    factors: Sequence[NDArray[np.float64]],
) -> tuple[tuple[Any, ...], ...]:
    """Normalize state_space input to a per-node tuple-of-tuples.

    ``raw`` may be:

    - ``None``: integer labels ``0..k-1`` are inferred from each factor's
      last-dim size.
    - A flat sequence: the same labels are applied uniformly to all nodes.
    - A sequence of sequences: each inner sequence gives per-node labels.
    """
    n_factors = len(factors)
    if raw is None:
        return tuple(tuple(range(int(f.shape[-1]))) for f in factors)

    raw_tuple = tuple(raw)
    if len(raw_tuple) == 0:
        raise ValueError("state_space cannot be empty")

    def _is_sequence_not_string(x: Any) -> bool:
        return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))

    if all(_is_sequence_not_string(elem) for elem in raw_tuple):
        # Per-node form: each element is a sub-sequence of labels.
        if len(raw_tuple) != n_factors:
            raise exceptions.InvalidTPM(
                f"state_space has {len(raw_tuple)} per-node entries; "
                f"factors imply {n_factors} nodes"
            )
        return tuple(tuple(elem) for elem in raw_tuple)  # type: ignore[arg-type]
    # Uniform form: same labels for all nodes.
    uniform = tuple(raw_tuple)
    return tuple(uniform for _ in range(n_factors))


class FactoredTPM(Displayable):
    """Per-node-factored conditional TPM."""

    __slots__ = ("_backend", "_node_labels", "_state_space")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        state_space: StateSpace = None,
        backend: Literal["ndarray", "xarray"] | None = None,
        node_labels: Sequence[str] | None = None,
    ) -> None:
        factor_arrays = tuple(np.asarray(f, dtype=np.float64) for f in factors)
        self._state_space = _normalize_state_space(state_space, factor_arrays)
        alphabet_sizes = tuple(len(s) for s in self._state_space)
        self._backend = _make_default_backend(factor_arrays, alphabet_sizes, backend)
        # Optional per-unit display labels (node names); do not affect equality
        # or hashing — purely for rendering. None falls back to integer indices.
        self._node_labels = tuple(node_labels) if node_labels is not None else None
        _validate(self)

    @property
    def state_space(self) -> tuple[tuple[Any, ...], ...]:
        """Per-node label tuples, e.g. ``((0, 1), (0, 1))`` for binary nodes."""
        return self._state_space

    @property
    def node_labels(self) -> tuple[str, ...] | None:
        """Per-unit display labels (node names), or ``None`` for integer indices."""
        return self._node_labels

    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        """Number of states per node, derived from ``state_space``."""
        return tuple(len(s) for s in self._state_space)

    @property
    def shape(self) -> tuple[int, ...]:
        return (*self.alphabet_sizes, self.n_nodes)

    @property
    def n_nodes(self) -> int:
        return self._backend.n_factors()

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
        state_space: StateSpace = None,
    ) -> FactoredTPM:
        """Convert a joint conditional TPM into the factored form.

        Accepts either:

        - Legacy binary form: shape ``(2,) * n + (n,)``, where the last
          dim's entry ``i`` is ``P(node_i = 1 | s_t)``. Factor ``i`` is
          built by stacking ``[1 - p_on, p_on]`` along an explicit
          alphabet dim.

        - Explicit-alphabet form: shape ``(a_1, ..., a_N, N, a_i)``.
          Factor ``i`` is ``joint[..., i, :]``.

        ``state_space`` follows the same rules as ``FactoredTPM.__init__``.
        When ``None``, integer labels are inferred from the joint's shape.
        """
        joint_arr = np.asarray(joint, dtype=np.float64)
        ndim = joint_arr.ndim

        # Determine alphabet_sizes from state_space or infer from shape.
        if state_space is None:
            # Try explicit-alphabet shape first: (a_1, ..., a_N, N, max_alpha).
            # Identified by ndim >= 3, second-to-last axis equals N, leading
            # axes give the per-unit alphabet sizes, and the last axis equals
            # max(alphabet_sizes).
            if ndim >= 3:
                n_cand = int(joint_arr.shape[-2])
                if ndim == n_cand + 2:
                    leading = joint_arr.shape[:n_cand]
                    if all(s >= 2 for s in leading) and joint_arr.shape[-1] == max(
                        leading
                    ):
                        n = n_cand
                        alphabet_sizes: tuple[int, ...] = leading
                        factors = tuple(
                            joint_arr[..., i, : alphabet_sizes[i]] for i in range(n)
                        )
                        return cls(factors=factors, state_space=None)
            if ndim < 2 or joint_arr.shape[-1] != ndim - 1:
                raise ValueError(
                    f"Cannot infer state_space from joint shape "
                    f"{joint_arr.shape}; expected legacy form "
                    f"(2,)*n + (n,) or pass state_space explicitly."
                )
            n = ndim - 1
            alphabet_sizes = (2,) * n
        else:
            # Build a temporary normalized state_space to extract alphabet_sizes.
            # We use a dummy factors list since we need n to build it.
            # For per-node form, len gives n directly; for flat form we need joint shape.
            raw_tuple = tuple(state_space)  # type: ignore[arg-type]

            def _is_sequence_not_string(x: Any) -> bool:
                return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))

            if all(_is_sequence_not_string(elem) for elem in raw_tuple):
                # Per-node form.
                n = len(raw_tuple)
                alphabet_sizes = tuple(len(elem) for elem in raw_tuple)  # type: ignore[arg-type]
            else:
                # Uniform form: need n from joint shape.
                if ndim < 2 or joint_arr.shape[-1] != ndim - 1:
                    raise ValueError(
                        f"Cannot infer n_nodes from joint shape "
                        f"{joint_arr.shape} with flat state_space; expected legacy form "
                        f"(2,)*n + (n,) or use per-node state_space."
                    )
                n = ndim - 1
                alphabet_sizes = (len(raw_tuple),) * n

        if joint_arr.shape[:-1] != alphabet_sizes:
            # Explicit-alphabet form: (a_1, ..., a_N, N, a_max)
            if (
                ndim == n + 2
                and joint_arr.shape[:n] == alphabet_sizes
                and joint_arr.shape[n] == n
            ):
                factors = tuple(joint_arr[..., i, : alphabet_sizes[i]] for i in range(n))
                return cls(factors=factors, state_space=state_space)
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
        return cls(factors=tuple(factors_list), state_space=state_space)

    def tpm_indices(self) -> tuple[int, ...]:
        """Substrate-unit indices: one entry per output unit (the leading
        factor axes); the trailing alphabet axis is per-unit.
        """
        return tuple(range(self.n_nodes))

    def condition(self, fixed: Mapping[int, int]) -> FactoredTPM:
        conditioned = [self._backend.select(i, fixed) for i in range(self.n_nodes)]
        return FactoredTPM(
            factors=conditioned,
            state_space=self._state_space,
            backend=None,
            node_labels=self._node_labels,
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
        a = self.alphabet_sizes
        max_alphabet = max(a)
        shape = (*a, n, max_alphabet)
        out = np.zeros(shape, dtype=np.float64)
        for i in range(n):
            factor = self.factor(i)
            a_i = a[i]
            broadcast_shape = (*a, a_i)
            out[..., i, :a_i] = np.broadcast_to(factor, broadcast_shape)
        return out

    @staticmethod
    def _varies_along_axis(factor: NDArray[np.float64], axis: int, tol: float) -> bool:
        """Whether ``factor`` is non-constant along the given input axis.

        A size-1 axis (a declared non-input) is constant by definition. For a
        full-size axis, the factor varies iff some slice differs from the first
        by more than ``tol`` in absolute value. The comparison is absolute (no
        relative tolerance), so a small-but-real dependence is not swallowed.
        """
        if factor.shape[axis] == 1:
            return False
        base = factor.take(0, axis=axis)
        return any(
            not bool(np.all(np.abs(factor.take(idx, axis=axis) - base) <= tol))
            for idx in range(1, factor.shape[axis])
        )

    def infer_edge(self, a: int, b: int) -> bool:
        """Whether the TPM implies a causal edge from node ``a`` to node ``b``.

        There is an edge iff node ``b``'s conditional distribution depends on
        node ``a``'s state — i.e. factor ``b`` is not constant along input axis
        ``a``. Equivalently, ``P(b | s) != P(b | s')`` for two input states
        ``s, s'`` differing only in node ``a``. Holds for any per-node alphabet
        size.
        """
        tol = max(10 ** (-config.numerics.precision), 1e-15)
        return self._varies_along_axis(self.factor(b), a, tol)

    def infer_cm(self) -> NDArray[np.int_]:
        """Infer the connectivity matrix implied by the TPM.

        Entry ``[a, b]`` is 1 iff the TPM implies an edge from node ``a`` to
        node ``b`` (see :meth:`infer_edge`); diagonal entries mark self-edges.
        Inferred per factor without materializing the joint, so the cost is
        ``O(N^2 * factor_size)``.
        """
        n = self.n_nodes
        tol = max(10 ** (-config.numerics.precision), 1e-15)
        cm = np.zeros((n, n), dtype=int)
        for b in range(n):
            factor = self.factor(b)
            for a in range(n):
                if self._varies_along_axis(factor, a, tol):
                    cm[a, b] = 1
        return cm

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FactoredTPM):
            return NotImplemented
        if self.alphabet_sizes != other.alphabet_sizes:
            return False
        if self.n_nodes != other.n_nodes:
            return False
        return all(
            np.array_equal(self.factor(i), other.factor(i)) for i in range(self.n_nodes)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.alphabet_sizes,
                tuple((self.factor(i) + 0.0).tobytes() for i in range(self.n_nodes)),
            )
        )

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        n = self.n_nodes
        a = self.alphabet_sizes
        unit_labels = list(self._node_labels or (str(i) for i in range(n)))
        compact = f"FactoredTPM(n_nodes={n}, alphabet_sizes={a})"
        if all(size == 2 for size in a):
            return _display.state_by_node_description(
                title="FactoredTPM",
                compact=compact,
                unit_labels=unit_labels,
                state_axis_sizes=a,
                prob_on_for_state=lambda state: [
                    self.factor(i)[state][1] for i in range(n)
                ],
            )
        return _display.distribution_grid_description(
            title="FactoredTPM",
            compact=compact,
            unit_labels=unit_labels,
            alphabet_sizes=a,
            dist_for_state=lambda state: [self.factor(i)[state] for i in range(n)],
        )

    def to_xarray(self) -> Any:
        """Return the factored conditional as a labeled :class:`xarray.Dataset`.

        Each unit ``i`` is a data variable ``"unit_{i}"`` holding ``P(unit i
        next | inputs)`` with dims ``("u0", ..., "u{N-1}", "u{i}_next")`` and
        integer coordinates from :attr:`state_space`. Requires the optional
        ``xarray`` dependency.
        """
        xr = _display.require_xarray()
        n = self.n_nodes
        state_space = self.state_space
        in_dims = tuple(f"u{j}" for j in range(n))
        in_coords = {in_dims[j]: list(state_space[j]) for j in range(n)}
        data_vars = {}
        for i in range(n):
            out_dim = f"u{i}_next"
            data_vars[f"unit_{i}"] = xr.DataArray(
                self.factor(i),
                dims=(*in_dims, out_dim),
                coords={**in_coords, out_dim: list(state_space[i])},
            )
        return xr.Dataset(data_vars)

    def __reduce__(self) -> tuple:  # type: ignore[override]
        backend_name = (
            "ndarray" if isinstance(self._backend, _NdarrayBackend) else "xarray"
        )
        return (
            _factored_tpm_from_pickle,
            (tuple(self.factors), self._state_space, backend_name, self._node_labels),
        )


def _factored_tpm_from_pickle(
    factors: tuple,  # type: ignore[type-arg]
    state_space: tuple,  # type: ignore[type-arg]
    backend: str,
    node_labels: tuple | None = None,  # type: ignore[type-arg]
) -> FactoredTPM:
    return FactoredTPM(
        factors=factors,
        state_space=state_space,
        backend=backend,  # type: ignore[arg-type]
        node_labels=node_labels,
    )


def _validate(factored: FactoredTPM) -> None:
    """Validate a freshly constructed FactoredTPM."""
    a = factored.alphabet_sizes
    ss = factored.state_space
    n = factored.n_nodes

    if len(ss) != n:
        raise exceptions.InvalidTPM(f"state_space has {len(ss)} entries; n_nodes={n}")
    for i, labels in enumerate(ss):
        if len(labels) != a[i]:
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has {len(labels)} labels but "
                f"factor[{i}] has alphabet size {a[i]}"
            )
        if len(set(labels)) != len(labels):
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has duplicate labels: {labels}"
            )

    if n != len(a):
        raise exceptions.InvalidTPM(
            f"n_nodes={n} does not match alphabet_sizes length {len(a)}"
        )
    if any(size < 2 for size in a):
        raise exceptions.InvalidTPM(f"alphabet_sizes must all be >= 2; got {a}")
    nl = factored.node_labels
    if nl is not None and len(nl) != n:
        raise exceptions.InvalidTPM(f"node_labels has {len(nl)} entries; n_nodes={n}")
    tol = max(10 ** (-config.numerics.precision), 1e-15)
    for i in range(n):
        f = factored.factor(i)
        if f.ndim != n + 1:
            raise exceptions.InvalidTPM(
                f"factor {i} has {f.ndim - 1} leading axes; expected {n} "
                f"(one per substrate unit). Factors must be full-dimension "
                f"(*alphabet_sizes, k_i)."
            )
        if f.shape[-1] != a[i]:
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has {a[i]} labels but "
                f"factor[{i}] last-dim size is {f.shape[-1]}"
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
