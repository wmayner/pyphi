"""Tests for pyphi.core.tpm — TPM Protocol and ExplicitTPM port."""

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


def test_explicit_tpm_is_a_tpm() -> None:
    """ExplicitTPM satisfies the TPM Protocol via runtime_checkable."""
    from pyphi.core.tpm import TPM
    from pyphi.core.tpm.explicit import ExplicitTPM

    arr = np.array([[0.5, 0.5], [0.7, 0.3]])
    tpm = ExplicitTPM(arr)
    assert isinstance(tpm, TPM)


def test_explicit_tpm_parity_with_legacy() -> None:
    """ExplicitTPM produces bit-identical output to legacy ExplicitTPM."""
    import pyphi.tpm as legacy
    from pyphi.core.tpm.explicit import ExplicitTPM

    arr = np.array(
        [
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            [[0.7, 0.7, 0.7], [0.3, 0.3, 0.3]],
        ]
    )
    new = ExplicitTPM(arr)
    old = legacy.ExplicitTPM(arr)
    np.testing.assert_array_equal(new.to_array(), np.asarray(old))
    assert new.shape == old.shape
    new_squeezed = new.squeeze()
    old_squeezed = old.squeeze()
    np.testing.assert_array_equal(new_squeezed.to_array(), np.asarray(old_squeezed))
