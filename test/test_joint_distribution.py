"""Tests for the JointDistribution base class and its subclasses."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.joint_distribution import JointDistribution
from pyphi.tpm import JointTPM


def test_joint_distribution_exists() -> None:
    assert JointDistribution is not None


def test_jointtpm_isinstance_jointdistribution() -> None:
    arr = np.full((2, 2, 2), 0.5)
    jtpm = JointTPM(arr)
    assert isinstance(jtpm, JointDistribution)


def test_joint_distribution_marginalize_out_inherited() -> None:
    arr = np.full((2, 2, 2), 0.5)
    jtpm = JointTPM(arr)
    result = jtpm.marginalize_out([0])
    assert isinstance(result, JointTPM)
