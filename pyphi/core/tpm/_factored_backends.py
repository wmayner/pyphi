"""Internal storage backends for FactoredTPM.

Not part of the public API. The chosen backend is selected by
:data:`pyphi.core.tpm.factored._FACTORED_TPM_DEFAULT_BACKEND` (set by
the in-project benchmark; see ``benchmarks/factored_tpm_backend.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


@runtime_checkable
class _StorageBackend(Protocol):
    """Internal storage abstraction for FactoredTPM."""

    def get_factor(self, i: int) -> NDArray[np.float64]: ...
    def n_factors(self) -> int: ...
    def alphabet_sizes(self) -> tuple[int, ...]: ...
    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]: ...
    def all_factors(self) -> tuple[NDArray[np.float64], ...]: ...


class _NdarrayBackend:
    """Tuple of ndarrays. Positional indexing.

    Name-based lookup goes through FactoredTPM's node-label mapping
    (FactoredTPM is the public surface for that).
    """

    __slots__ = ("_alphabet_sizes", "_factors")

    def __init__(
        self,
        factors: Sequence[ArrayLike],
        alphabet_sizes: Sequence[int],
    ) -> None:
        self._factors = tuple(np.asarray(f, dtype=np.float64) for f in factors)
        self._alphabet_sizes = tuple(int(a) for a in alphabet_sizes)

    def get_factor(self, i: int) -> NDArray[np.float64]:
        return self._factors[i]

    def n_factors(self) -> int:
        return len(self._factors)

    def alphabet_sizes(self) -> tuple[int, ...]:
        return self._alphabet_sizes

    def all_factors(self) -> tuple[NDArray[np.float64], ...]:
        return self._factors

    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray[np.float64]:
        factor = self._factors[i]
        idx: list[Any] = [slice(None)] * factor.ndim
        for j, state_j in fixed.items():
            idx[j] = state_j
        out = factor[tuple(idx)]
        for j in sorted(fixed):
            out = np.expand_dims(out, axis=j)
        return out


def _make_default_backend(
    factors: Sequence[ArrayLike],
    alphabet_sizes: Sequence[int],
    requested: str | None,
) -> _StorageBackend:
    """Construct the requested storage backend. ``None`` uses the module default."""
    from pyphi.core.tpm.factored import _FACTORED_TPM_DEFAULT_BACKEND

    backend_name = requested or _FACTORED_TPM_DEFAULT_BACKEND
    if backend_name == "ndarray":
        return _NdarrayBackend(factors, alphabet_sizes)
    if backend_name == "xarray":
        from pyphi.core.tpm._factored_backends_xarray import (  # type: ignore[import]
            _XarrayBackend,
        )

        return _XarrayBackend(factors, alphabet_sizes)
    raise ValueError(
        f"Unknown backend {backend_name!r}; expected 'ndarray' or 'xarray'."
    )
