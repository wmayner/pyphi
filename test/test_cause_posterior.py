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
