"""Unit tests for FactoredTPM — per-node factored conditional TPM."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pyphi.core.tpm.base import TPM
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.exceptions import InvalidTPM


def _two_node_factored() -> FactoredTPM:
    f0 = np.array(
        [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    f1 = np.array(
        [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    return FactoredTPM(factors=[f0, f1], alphabet_sizes=(2, 2))


def test_factored_tpm_construction() -> None:
    f = _two_node_factored()
    assert f.n_nodes == 2
    assert f.alphabet_sizes == (2, 2)


def test_factored_tpm_satisfies_protocol() -> None:
    f = _two_node_factored()
    assert isinstance(f, TPM)


def test_factored_tpm_shape() -> None:
    f = _two_node_factored()
    assert f.shape == (2, 2, 2)


def test_factored_tpm_factor_access() -> None:
    f = _two_node_factored()
    assert f.factor(0).shape == (2, 2, 2)
    assert f.factor(1).shape == (2, 2, 2)
    np.testing.assert_allclose(f.factor(0), f.factors[0])


def test_factored_tpm_validation_rejects_nonsumming_factor() -> None:
    bad = np.array(
        [[[0.3, 0.3], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    good = bad.copy()
    good[0, 0, 0] = 0.5
    with pytest.raises(InvalidTPM, match="sums to 1"):
        FactoredTPM(factors=[bad, good], alphabet_sizes=(2, 2))


def test_factored_tpm_validation_rejects_alphabet_lt_2() -> None:
    f0 = np.array([[[1.0]]], dtype=np.float64)
    with pytest.raises(InvalidTPM, match="alphabet"):
        FactoredTPM(factors=[f0], alphabet_sizes=(1,))


def test_factored_tpm_equality() -> None:
    a = _two_node_factored()
    b = _two_node_factored()
    assert a == b
    assert (a != b) is False


def test_factored_tpm_repr() -> None:
    f = _two_node_factored()
    r = repr(f)
    assert "FactoredTPM" in r
    assert "n_nodes=2" in r


def test_factored_tpm_pickling() -> None:
    f = _two_node_factored()
    restored = pickle.loads(pickle.dumps(f))
    assert restored == f
