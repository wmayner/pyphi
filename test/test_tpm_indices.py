"""tpm_indices() semantics per concrete TPM type."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.joint import JointTPM
from pyphi.tpm import JointTPM as LegacyJointTPM


def test_joint_tpm_indices_returns_range_ndim_minus_one() -> None:
    arr = np.zeros((2, 2, 2, 2))  # 3-node binary SBN-form
    j = JointTPM(arr)
    assert j.tpm_indices() == (0, 1, 2)


def test_factored_tpm_indices_returns_range_n_nodes() -> None:
    factors = [np.full((2, 2), 0.5) for _ in range(2)]
    f = FactoredTPM(factors=factors)
    assert f.tpm_indices() == (0, 1)


def test_legacy_joint_tpm_indices_excludes_singleton_axes() -> None:
    """Legacy ``pyphi.tpm.JointTPM`` excludes size-1 axes (those collapsed by
    marginalizing out non-input nodes) and returns only the size-2 leading
    axes. ``pyphi/macro.py`` indexes the squeezed array against these
    indices, so the singleton-exclusion semantics is load-bearing.
    """
    arr = np.zeros((2, 1, 2, 2))  # axis 1 marginalized to a singleton
    j = LegacyJointTPM(arr)
    assert j.tpm_indices() == (0, 2)
