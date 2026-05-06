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


def test_cause_tpm_parity() -> None:
    """core.tpm.marginalization.cause_tpm matches legacy backward_tpm."""
    from pyphi import examples
    from pyphi.core.tpm.explicit import ExplicitTPM
    from pyphi.core.tpm.marginalization import cause_tpm
    from pyphi.tpm import backward_tpm as legacy_backward_tpm

    network = examples.basic_network()
    state = (1, 0, 0)
    node_indices = (0, 1, 2)

    new_tpm = cause_tpm(ExplicitTPM(network.tpm), state, node_indices)
    old_tpm = legacy_backward_tpm(network.tpm, state, node_indices)
    np.testing.assert_array_equal(new_tpm.to_array(), np.asarray(old_tpm))


def test_effect_tpm_parity() -> None:
    """core.tpm.marginalization.effect_tpm matches legacy condition_tpm."""
    from pyphi import examples
    from pyphi import utils
    from pyphi.core.tpm.explicit import ExplicitTPM
    from pyphi.core.tpm.marginalization import effect_tpm

    network = examples.basic_network()
    state = (1, 0, 0)
    external_indices = (2,)
    external_state = utils.state_of(external_indices, state)
    background = dict(zip(external_indices, external_state, strict=False))

    new_tpm = effect_tpm(ExplicitTPM(network.tpm), background)
    old_tpm = network.tpm.condition_tpm(background)
    np.testing.assert_array_equal(new_tpm.to_array(), np.asarray(old_tpm))
