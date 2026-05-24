"""Joint posterior over past states given an observed future mechanism state.

Returned by ``cause_tpm`` after Bayesian inversion of a forward TPM. Stored
as a multidimensional probability distribution over the joint past-state
space.

Past nodes are generally NOT conditionally independent in this posterior —
observing the future couples the past inputs through any non-trivial
mechanism. So this is a sibling of :class:`JointTPM` (joint conditional)
under :class:`JointDistribution`, rather than a subtype.

Canonical shape: ``(*alphabet_sizes, n_observed_nodes)``, where each
``alphabet_sizes[i]`` is the alphabet size of past node ``i`` and
``n_observed_nodes`` is the number of observed mechanism nodes whose joint
state conditions the posterior. The trailing axis carries the per-output-node
probability slice used by downstream consumers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .joint_distribution import JointDistribution


class CausePosterior(JointDistribution):
    """Joint posterior ``P(s_t | s_{t+1,M} = mu)`` over past states.

    Inherits all storage, marginalization, and array machinery from
    :class:`JointDistribution`.
    """

    def factor(self, i: int) -> NDArray[np.float64]:
        """Per-output-unit factor of shape ``(*alphabet_sizes, 2)``.

        Constructs ``[1 - x, x]`` along a new trailing axis from the
        SBN-form ``P(node_i = 1 | s_t)`` slice at output index ``i``.

        Binary-only: the SBN storage encodes only firing probability and
        cannot represent k-ary alphabets.
        """
        arr = np.asarray(self._tpm)
        on = arr[..., i]
        off = 1.0 - on
        return np.stack([off, on], axis=-1)

    def __repr__(self) -> str:
        return f"CausePosterior({self._tpm!r})"
