"""Tests for CausePosterior — joint posterior over past states."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.cause_posterior import CausePosterior
from pyphi.core.tpm.joint_distribution import JointDistribution
from pyphi.tpm import JointTPM


def test_cause_posterior_isinstance_jointdistribution() -> None:
    """CausePosterior is a JointDistribution."""
    cp = CausePosterior(np.full((2, 2), 0.25))
    assert isinstance(cp, JointDistribution)


def test_cause_posterior_not_isinstance_jointtpm() -> None:
    """CausePosterior is a sibling of JointTPM, not a subtype."""
    cp = CausePosterior(np.full((2, 2), 0.25))
    assert not isinstance(cp, JointTPM)


def test_jointtpm_not_isinstance_cause_posterior() -> None:
    """JointTPM is not a CausePosterior."""
    jtpm = JointTPM(np.full((2, 2, 2), 0.5))
    assert not isinstance(jtpm, CausePosterior)


def test_cause_posterior_marginalize_out_inherited() -> None:
    """marginalize_out is inherited from JointDistribution; returns CausePosterior."""
    cp = CausePosterior(np.full((2, 2, 2), 0.125))
    result = cp.marginalize_out([0])
    assert isinstance(result, CausePosterior)


def test_cause_posterior_repr() -> None:
    """__repr__ tags the type."""
    cp = CausePosterior(np.full((2, 2), 0.25))
    assert repr(cp).startswith("CausePosterior(")


def test_cause_posterior_factor_returns_per_unit_distribution() -> None:
    """factor(i) returns shape (*alphabet_sizes, 2) with [1-x, x] entries."""
    arr = np.array([[0.2, 0.7], [0.3, 0.4]], dtype=np.float64)
    posterior = CausePosterior(arr)  # SBN-form: trailing axis = n_nodes
    f0 = posterior.factor(0)
    np.testing.assert_allclose(f0[..., 1], arr[..., 0])
    np.testing.assert_allclose(f0[..., 0], 1.0 - arr[..., 0])
    f1 = posterior.factor(1)
    np.testing.assert_allclose(f1[..., 1], arr[..., 1])
    np.testing.assert_allclose(f1[..., 0], 1.0 - arr[..., 1])
