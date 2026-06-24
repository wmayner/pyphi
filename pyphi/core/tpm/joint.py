"""Numpy-backed JointTPM port behind the TPM Protocol.

Wraps :class:`pyphi.core.tpm.joint_distribution.JointTPM`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi.core.tpm.joint_distribution import JointTPM as _BackingJointTPM


class JointTPM:
    """Numpy-backed transition probability matrix.

    Wraps :class:`pyphi.core.tpm.joint_distribution.JointTPM` and exposes
    the :class:`pyphi.core.tpm.TPM` Protocol surface. Numerical behavior is
    delegated to the underlying implementation; the wrapper exists to give
    the new layering a single, type-checked entry point.
    """

    __slots__ = ("_inner",)

    def __init__(self, data: ArrayLike) -> None:
        if isinstance(data, _BackingJointTPM):
            self._inner = data
        else:
            self._inner = _BackingJointTPM(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self._inner).shape)

    @property
    def n_nodes(self) -> int:
        return int(self.shape[-1]) if self.shape else 0

    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        """All nodes are binary in the joint-storage form."""
        return (2,) * self.n_nodes

    def tpm_indices(self) -> tuple[int, ...]:
        """Substrate-unit indices: all leading axes are per-substrate-unit
        past-state axes; the trailing axis carries per-output-unit firing
        probability (SBN-form) or output state.
        """
        return tuple(range(len(self.shape) - 1))

    def condition(self, fixed: Mapping[int, int]) -> JointTPM:
        return JointTPM(self._inner.condition_tpm(dict(fixed)))

    def squeeze(self) -> JointTPM:
        return JointTPM(self._inner.squeeze())

    def to_array(self) -> NDArray[np.float64]:
        return np.asarray(self._inner)

    def __getitem__(self, key: Any) -> Any:
        return self._inner[key]

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        arr = np.asarray(self._inner)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def array_equal(self, other: object) -> bool:
        """Return whether this TPM equals another, numerically."""
        return np.array_equal(np.asarray(self), np.asarray(other))

    def __getattr__(self, name: str) -> Any:
        # Passthrough for legacy methods not yet lifted to the Protocol surface.
        # Guard: skip dunder/sunder names (pickle, copy, etc.) and bail
        # if _inner hasn't been set (during unpickling __new__).
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError as e:
            raise AttributeError(name) from e
        return getattr(inner, name)

    def __repr__(self) -> str:
        return f"JointTPM(shape={self.shape})"
