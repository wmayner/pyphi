"""Causal marginalization — named operations against IIT 4.0 Eq. 3 / Eq. 4."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from pyphi import exceptions

from .base import TPM
from .factored import FactoredTPM
from .joint import JointTPM

# Cap on any single intermediate array in the sum-product contraction
# (~1 GiB of float64). Densely coupled substrates whose cheapest elimination
# step exceeds this fail fast with an informative error instead of OOM.
_MAX_INTERMEDIATE_ELEMENTS = 2**27


class CauseMarginals:
    """Cause factors for a set of output units — IIT 4.0 Eq. 4.

    Maps each output unit ``i`` to its cause factor of shape
    ``(*alphabet_sizes, k_i)`` in the substrate-global axis convention
    (size-1 input axes mark non-dependence, exactly as in
    :class:`~pyphi.core.tpm.factored.FactoredTPM` factors, and
    ``.factor(i)`` mirrors that class's accessor). Holds only the
    requested output units.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors: Mapping[int, NDArray[np.float64]]) -> None:
        self._factors = dict(factors)

    @property
    def indices(self) -> tuple[int, ...]:
        """The output-unit indices, ascending."""
        return tuple(sorted(self._factors))

    def factor(self, i: int) -> NDArray[np.float64]:
        """The cause factor for output unit ``i``."""
        return self._factors[i]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CauseMarginals):
            return NotImplemented
        return self._factors.keys() == other._factors.keys() and all(
            np.array_equal(self._factors[i], other._factors[i]) for i in self._factors
        )

    def __hash__(self) -> int:
        return hash(
            tuple(
                (i, self._factors[i].shape, (self._factors[i] + 0.0).tobytes())
                for i in sorted(self._factors)
            )
        )


def _check_intermediate(size: int) -> None:
    if size > _MAX_INTERMEDIATE_ELEMENTS:
        raise exceptions.IntractableCauseInversionError(
            f"cause inversion would materialize an intermediate of {size} "
            f"elements (cap: {_MAX_INTERMEDIATE_ELEMENTS}); the substrate's "
            f"coupling is too dense for the reduced inversion"
        )


def _merged_elements(shapes: list[tuple[int, ...]]) -> int:
    """Element count of the broadcast product of arrays with these shapes."""
    size = 1
    for k in range(len(shapes[0])):
        size *= max(s[k] for s in shapes)
    return size


def _sum_product(
    slices: list[NDArray[np.float64]],
    keep_axes: frozenset[int],
) -> NDArray[np.float64]:
    """Marginal of ``∏ slices`` over ``keep_axes`` by greedy elimination.

    All arrays are full-ndim with size-1 axes marking non-dependence, so
    ufunc broadcasting aligns factors with no explicit axis bookkeeping
    (valid up to numpy's 64-dimension limit; ``np.einsum`` and
    ``np.broadcast_shapes`` have lower caps and cannot be used here). Each
    step eliminates the axis whose merged product of involved slices is
    smallest, ties breaking toward the lowest axis index — deterministic
    given shapes.
    """
    factors = list(slices)
    n = factors[0].ndim
    remaining = [k for k in range(n) if k not in keep_axes]
    while remaining:
        best_axis = -1
        best_size = -1
        for k in remaining:
            shapes = [f.shape for f in factors if f.shape[k] > 1]
            size = _merged_elements(shapes) if shapes else 0
            if best_size < 0 or size < best_size:
                best_axis, best_size = k, size
        _check_intermediate(best_size)
        remaining.remove(best_axis)
        involved = [f for f in factors if f.shape[best_axis] > 1]
        rest = [f for f in factors if f.shape[best_axis] == 1]
        if involved:
            prod = involved[0]
            for f in involved[1:]:
                prod = prod * f
            rest.append(prod.sum(axis=best_axis, keepdims=True))
        factors = rest
    _check_intermediate(_merged_elements([f.shape for f in factors]))
    out = factors[0]
    for f in factors[1:]:
        out = out * f
    return out


def _cause_marginal_factored(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CauseMarginals:
    """Cause factors for the system units — IIT 4.0 Eq. 4.

    For each system unit ``i`` and output value ``s_i``:

        factor_i(s_t)[s_i] = Σ_{w_t} P(s_i | s_t, w_t) · (pr_bg(s_t, w_t) / norm)

    where ``pr_bg`` is the joint likelihood of the observed state summed
    over the system past, ``norm`` sums it over all past states, and the
    outer sum runs over background past states. Evaluated as a sum-product
    contraction over the factored TPM's dependence structure: the joint
    likelihood is never materialized over all substrate units, and the
    background weight carries real extent only on background axes some
    system factor depends on. Factors are returned only for output units
    in ``node_indices``.
    """
    n = factored.n_nodes
    system = frozenset(node_indices)
    background_axes = tuple(k for k in range(n) if k not in system)

    # Per-unit likelihood of the observed state given the past, full-ndim
    # with size-1 non-parent axes: factor_j(s_t)[state_j].
    slices = [factored.factor(j)[..., state[j]] for j in range(n)]

    # Background axes some system factor actually depends on — the only
    # axes on which the outputs can see the weight.
    relevant = frozenset(
        k
        for i in node_indices
        for k, dim in enumerate(factored.factor(i).shape[:-1])
        if dim > 1 and k not in system
    )

    pr_bg = _sum_product(slices, keep_axes=relevant)
    norm = pr_bg.sum()
    if norm <= 0.0:
        raise exceptions.StateUnreachableBackwardsError(state)
    weight = pr_bg / norm

    out_factors: dict[int, NDArray[np.float64]] = {}
    for i in node_indices:
        forward_i = factored.factor(i)
        _check_intermediate(_merged_elements([forward_i.shape, (*weight.shape, 1)]))
        weighted = forward_i * weight[..., np.newaxis]
        if background_axes:
            weighted = weighted.sum(axis=background_axes, keepdims=True)
        out_factors[i] = weighted
    return CauseMarginals(out_factors)


def cause_marginal(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CauseMarginals:
    """Cause factors for the system units — IIT 4.0 Eq. 4.

    Returns a :class:`CauseMarginals` mapping each unit in ``node_indices``
    to its cause factor of shape ``(*alphabet_sizes, k_i)`` in the
    substrate-global axis convention: ``P(s_i,t | s_{M,t+1} = state_M)``
    per output unit, with background units marginalized under
    ``pr_bg / norm`` weighting. Joint/array inputs are converted to
    :class:`~pyphi.core.tpm.factored.FactoredTPM` first.
    """
    if isinstance(tpm, FactoredTPM):
        return _cause_marginal_factored(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        factored = FactoredTPM.from_joint(tpm._inner)
        return cause_marginal(factored, state, node_indices)
    arr = tpm.to_array()
    factored = FactoredTPM.from_joint(arr)
    return cause_marginal(factored, state, node_indices)


def effect_marginal(
    tpm: TPM,
    background: Mapping[int, int],
) -> TPM:
    """Forward TPM conditioned on external state — IIT 4.0 Eq. 4."""
    if isinstance(tpm, FactoredTPM):
        return _effect_marginal_factored(tpm, background)
    return tpm.condition(background)


def _effect_marginal_factored(
    factored: FactoredTPM,
    background: Mapping[int, int],
) -> FactoredTPM:
    """Condition a factored TPM on background nodes."""
    return factored.condition(background)
