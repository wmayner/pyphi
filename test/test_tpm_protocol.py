"""Verify the TPM Protocol contract and ExplicitTPM conformance."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.base import TPM
from pyphi.core.tpm.explicit import ExplicitTPM
from pyphi.core.unit import Unit


def test_tpm_protocol_has_alphabet_sizes() -> None:
    """The TPM Protocol exposes alphabet_sizes as a property."""
    tpm = ExplicitTPM(np.zeros((2, 2, 2, 3)))
    assert isinstance(tpm, TPM)
    assert tpm.alphabet_sizes == (2, 2, 2)


def test_tpm_protocol_lacks_squeeze() -> None:
    """The TPM Protocol no longer requires squeeze (lives on JointTPM only)."""
    assert not hasattr(TPM, "squeeze") or "squeeze" not in TPM.__protocol_attrs__  # type: ignore[attr-defined]


def test_unit_has_alphabet_size_default_2() -> None:
    """Unit defaults to alphabet_size=2."""
    u = Unit(index=0, label="A")
    assert u.alphabet_size == 2


def test_unit_alphabet_size_overridable() -> None:
    """Unit.alphabet_size accepts a non-default value."""
    u = Unit(index=0, label="A", alphabet_size=3)
    assert u.alphabet_size == 3
