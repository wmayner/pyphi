"""Numpy-backed ExplicitTPM port behind the TPM Protocol.

Wraps :class:`pyphi.tpm.ExplicitTPM`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi import tpm as _legacy_tpm


class ExplicitTPM:
    """Numpy-backed transition probability matrix.

    Wraps the legacy :class:`pyphi.tpm.ExplicitTPM` and exposes the
    :class:`pyphi.core.tpm.TPM` Protocol surface. Numerical behavior is
    delegated to the legacy implementation; the wrapper exists to give the
    new layering a single, type-checked entry point.
    """

    __slots__ = ("_inner",)

    def __init__(self, data: ArrayLike) -> None:
        if isinstance(data, _legacy_tpm.ExplicitTPM):
            self._inner = data
        else:
            self._inner = _legacy_tpm.ExplicitTPM(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self._inner).shape)

    @property
    def n_nodes(self) -> int:
        return int(self.shape[-1]) if self.shape else 0

    def condition(self, fixed: Mapping[int, int]) -> ExplicitTPM:
        return ExplicitTPM(self._inner.condition_tpm(dict(fixed)))

    def squeeze(self) -> ExplicitTPM:
        return ExplicitTPM(self._inner.squeeze())

    def to_array(self) -> NDArray[np.float64]:
        return np.asarray(self._inner)

    def __getattr__(self, name: str) -> Any:
        # During the worktree, callers may still need legacy methods we
        # haven't lifted yet. This passthrough is removed at Phase 8.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"ExplicitTPM(shape={self.shape})"
