"""tpm_indices() semantics per concrete TPM type."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.cause_posterior import CausePosterior
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.joint import JointTPM


def test_joint_tpm_indices_returns_range_ndim_minus_one() -> None:
    arr = np.zeros((2, 2, 2, 2))  # 3-node binary SBN-form
    j = JointTPM(arr)
    assert j.tpm_indices() == (0, 1, 2)


def test_factored_tpm_indices_returns_range_n_nodes() -> None:
    factors = [np.full((2, 2), 0.5) for _ in range(2)]
    f = FactoredTPM(factors=factors)
    assert f.tpm_indices() == (0, 1)


def test_cause_posterior_indices_returns_range_ndim_minus_one() -> None:
    arr = np.zeros((2, 2, 3))  # 2-node-past x 3 mechanism observations
    c = CausePosterior(arr)
    assert c.tpm_indices() == (0, 1)
