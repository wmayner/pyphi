"""xarray storage backend for FactoredTPM.

Imported lazily by :func:`_factored_backends._make_default_backend`
when ``backend="xarray"`` is requested. The module-level ``import
xarray as xr`` fails with ``ModuleNotFoundError`` if xarray is not
installed; users see the standard install hint via
``pip install pyphi[xarray]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from numpy.typing import NDArray


class _XarrayBackend:
    """Tuple of xr.DataArray with named input dims.

    Same semantic API as :class:`_NdarrayBackend`. Naming convention:
    factor ``i`` has dims ``("in_0", "in_1", ..., "in_{N-1}", "out_i")``.
    """

    __slots__ = ("_alphabet_sizes", "_factors")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        alphabet_sizes: Sequence[int],
    ) -> None:
        alphabet_sizes = tuple(int(a) for a in alphabet_sizes)
        self._alphabet_sizes = alphabet_sizes
        n = len(alphabet_sizes)
        wrapped: list[xr.DataArray] = []
        for i, f in enumerate(factors):
            arr = np.asarray(f, dtype=np.float64)
            dims = (*tuple(f"in_{j}" for j in range(n)), f"out_{i}")
            wrapped.append(xr.DataArray(arr, dims=dims))
        self._factors = tuple(wrapped)

    def get_factor(self, i: int) -> NDArray[np.float64]:
        return np.asarray(self._factors[i].values)

    def n_factors(self) -> int:
        return len(self._factors)

    def alphabet_sizes(self) -> tuple[int, ...]:
        return self._alphabet_sizes

    def all_factors(self) -> tuple[NDArray[np.float64], ...]:
        return tuple(np.asarray(f.values) for f in self._factors)

    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        factor = self._factors[i]
        idx: dict[str, int] = {f"in_{j}": state_j for j, state_j in fixed.items()}
        sliced = factor.isel(idx)
        out = sliced.values
        for j in sorted(fixed):
            out = np.expand_dims(out, axis=j)
        return out
