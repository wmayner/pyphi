"""TPM Protocol — structural contract every transition probability matrix satisfies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class TPM(Protocol):
    """Structural protocol satisfied by every PyPhi TPM.

    Implementations: :class:`pyphi.core.tpm.joint.JointTPM` (joint
    ndarray storage).
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def n_nodes(self) -> int: ...

    @property
    def alphabet_sizes(self) -> tuple[int, ...]: ...

    def condition(self, fixed: Mapping[int, int]) -> TPM: ...

    def to_array(self) -> NDArray[np.float64]: ...
