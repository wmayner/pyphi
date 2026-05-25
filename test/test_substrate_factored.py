"""Substrate cutover to FactoredTPM canonical storage."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.substrate import Substrate


def test_substrate_stores_factored_tpm() -> None:
    joint = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    s = Substrate(tpm=joint)
    assert isinstance(s.factored_tpm, FactoredTPM)
    assert s.factored_tpm.n_nodes == 2


def test_substrate_joint_tpm_method() -> None:
    joint = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    s = Substrate(tpm=joint)
    materialized = s.joint_tpm()
    np.testing.assert_allclose(materialized[..., 0], joint[..., 0], atol=1e-12)


def test_substrate_marginals_keyword() -> None:
    f0 = np.full((2, 2, 2), 0.5)
    f1 = np.full((2, 2, 2), 0.5)
    s = Substrate(marginals=[f0, f1])
    assert s.factored_tpm.n_nodes == 2


def test_substrate_mutually_exclusive_tpm_marginals() -> None:
    joint = np.zeros((2, 2, 2))
    f0 = np.full((2, 2, 2), 0.5)
    with pytest.raises(ValueError, match=r"tpm.*marginals.*not both"):
        Substrate(tpm=joint, marginals=[f0])  # type: ignore[call-arg]


def test_substrate_from_factored_factory() -> None:
    f0 = np.full((2, 2, 2), 0.5)
    f1 = np.full((2, 2, 2), 0.5)
    factored = FactoredTPM(factors=[f0, f1], state_space=((0, 1), (0, 1)))
    s = Substrate.from_factored(factored)
    assert s.factored_tpm is factored or s.factored_tpm == factored


def test_substrate_rejects_factored_via_tpm_keyword() -> None:
    """FactoredTPM instances must use marginals= or from_factored, not tpm=."""
    f0 = np.full((2, 2, 2), 0.5)
    f1 = np.full((2, 2, 2), 0.5)
    factored = FactoredTPM(factors=[f0, f1], state_space=((0, 1), (0, 1)))
    with pytest.raises(ValueError, match="marginals"):
        Substrate(tpm=factored)  # type: ignore[arg-type]
