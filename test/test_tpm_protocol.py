"""Verify the TPM Protocol contract and ExplicitTPM conformance."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.base import TPM
from pyphi.core.tpm.explicit import ExplicitTPM


def test_tpm_protocol_has_alphabet_sizes() -> None:
    """The TPM Protocol exposes alphabet_sizes as a property."""
    tpm = ExplicitTPM(np.zeros((2, 2, 2, 3)))
    assert isinstance(tpm, TPM)
    assert tpm.alphabet_sizes == (2, 2, 2)


def test_tpm_protocol_lacks_squeeze() -> None:
    """The TPM Protocol no longer requires squeeze (lives on JointTPM only)."""
    assert not hasattr(TPM, "squeeze") or "squeeze" not in TPM.__protocol_attrs__  # type: ignore[attr-defined]
