"""Tests for pyphi.core.tpm — TPM Protocol and JointTPM port."""

from __future__ import annotations

import numpy as np


def test_tpm_protocol_importable() -> None:
    """The TPM Protocol must be importable from pyphi.core.tpm."""
    from pyphi.core.tpm import TPM  # noqa: F401


def test_tpm_protocol_is_runtime_checkable() -> None:
    """TPM Protocol is decorated with runtime_checkable."""
    from pyphi.core.tpm import TPM

    assert hasattr(TPM, "_is_runtime_protocol")
    assert TPM._is_runtime_protocol is True


def test_joint_tpm_is_a_tpm() -> None:
    """JointTPM satisfies the TPM Protocol via runtime_checkable."""
    from pyphi.core.tpm import TPM
    from pyphi.core.tpm.joint import JointTPM

    arr = np.array([[0.5, 0.5], [0.7, 0.3]])
    tpm = JointTPM(arr)
    assert isinstance(tpm, TPM)
