"""TPM Protocol — the structural contract every transition probability matrix satisfies.

Implementations:

- :class:`pyphi.core.tpm.explicit.ExplicitTPM` (numpy-backed; this project / P7).
- ``ImplicitTPM`` (factored per-node TPM; P12, drawing on PR #105).

The Protocol body deliberately omits ``alphabet_size`` introspection —
binary state spaces are implicit in P7. P12 lifts that assumption.

Causal marginalization (``cause_tpm``, ``effect_tpm``) lives as free
functions in :mod:`pyphi.core.tpm.marginalization` rather than on the
Protocol — those operations sit *above* a TPM, transforming one TPM
into another.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class TPM(Protocol):
    """Structural protocol satisfied by every PyPhi TPM."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def n_nodes(self) -> int: ...

    def condition(self, fixed: Mapping[int, int]) -> TPM: ...

    def squeeze(self) -> TPM: ...

    def to_array(self) -> NDArray[np.float64]: ...
