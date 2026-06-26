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


def test_joint_tpm_parity_with_legacy() -> None:
    """JointTPM produces bit-identical output to legacy JointTPM."""
    from pyphi.core.tpm import joint_distribution as legacy
    from pyphi.core.tpm.joint import JointTPM

    arr = np.array(
        [
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            [[0.7, 0.7, 0.7], [0.3, 0.3, 0.3]],
        ]
    )
    new = JointTPM(arr)
    old = legacy.JointTPM(arr)
    np.testing.assert_array_equal(new.to_array(), np.asarray(old))
    assert new.shape == old.shape
    new_squeezed = new.squeeze()
    old_squeezed = old.squeeze()
    np.testing.assert_array_equal(new_squeezed.to_array(), np.asarray(old_squeezed))


def test_effect_marginal_parity() -> None:
    """core.tpm.marginalization.effect_marginal matches legacy condition_tpm."""
    from pyphi import examples
    from pyphi import utils
    from pyphi.core.tpm.joint import JointTPM
    from pyphi.core.tpm.marginalization import effect_marginal

    substrate = examples.basic_substrate()
    state = (1, 0, 0)
    external_indices = (2,)
    external_state = utils.state_of(external_indices, state)
    background = dict(zip(external_indices, external_state, strict=False))

    from pyphi.core.tpm.joint_distribution import JointTPM as _LegacyJointTPM

    # Legacy JointTPM expects the (2,...,2,N) shape.
    joint = substrate._legacy_binary_joint()
    new_tpm = effect_marginal(JointTPM(joint), background)
    old_tpm = _LegacyJointTPM(joint).condition_tpm(background)
    np.testing.assert_array_equal(new_tpm.to_array(), np.asarray(old_tpm))
