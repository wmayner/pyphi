"""tpm_indices() semantics for FactoredTPM."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM


def test_factored_tpm_indices_returns_range_n_nodes() -> None:
    # Full-dimension factors for a 2-node substrate: (2, 2, 2) each.
    factors = [np.full((2, 2, 2), 0.5) for _ in range(2)]
    f = FactoredTPM(factors=factors)
    assert f.tpm_indices() == (0, 1)
