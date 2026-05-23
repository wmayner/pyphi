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


def test_factored_tpm_hash_consistent_with_eq_for_signed_zeros() -> None:
    """eq/hash contract: a == b implies hash(a) == hash(b), including for -0.0 vs 0.0."""
    f0a = np.array(
        [[[0.0, 1.0], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]], dtype=np.float64
    )
    f0b = f0a.copy()
    f0b[0, 0, 0] = -0.0
    a = FactoredTPM(factors=[f0a, f0a.copy()], alphabet_sizes=(2, 2))
    b = FactoredTPM(factors=[f0b, f0a.copy()], alphabet_sizes=(2, 2))
    assert a == b
    assert hash(a) == hash(b)


def test_factored_tpm_rejects_factor_count_mismatch() -> None:
    """n_nodes (from factors) must match len(alphabet_sizes)."""
    f = np.full((2, 2, 2), 0.5, dtype=np.float64)
    with pytest.raises(InvalidTPM, match="n_nodes"):
        FactoredTPM(factors=[f, f, f], alphabet_sizes=(2, 2))
    with pytest.raises(InvalidTPM, match="n_nodes"):
        FactoredTPM(factors=[f], alphabet_sizes=(2, 2))


# --- round-trip tests ---


def _random_joint_tpm(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random binary joint TPM, shape (2,)*n + (n,) — entries are P(node_i=1)."""
    return rng.uniform(size=(2,) * n + (n,))


def test_from_joint_round_trip_n2() -> None:
    rng = np.random.default_rng(42)
    joint = _random_joint_tpm(rng, 2)
    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2, 2))
    reconstructed = factored.to_joint()
    p_on = joint
    explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
    np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)


def test_from_joint_round_trip_n3() -> None:
    rng = np.random.default_rng(99)
    joint = _random_joint_tpm(rng, 3)
    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2, 2, 2))
    reconstructed = factored.to_joint()
    p_on = joint
    explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
    np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)


def test_from_joint_invalid_shape_raises() -> None:
    bad = np.zeros((2, 2))
    with pytest.raises(ValueError, match="shape"):
        FactoredTPM.from_joint(bad, alphabet_sizes=(2, 2))


def test_to_joint_shape() -> None:
    f = _two_node_factored()
    joint = f.to_joint()
    assert joint.shape[:-2] == (2, 2)
    assert joint.shape[-2] == 2  # n_nodes
    assert joint.shape[-1] == 2  # alphabet


def test_from_joint_to_joint_roundtrip_stability_binary() -> None:
    """Round-trip preserves the joint to floating-point precision."""
    rng = np.random.default_rng(2026)
    for n in (2, 3, 4):
        joint = _random_joint_tpm(rng, n)
        factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2,) * n)
        reconstructed = factored.to_joint()
        p_on = joint
        explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
        np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)


def test_from_joint_explicit_alphabet_uniform_k3() -> None:
    """Uniform k=3 round-trip via the explicit-alphabet form."""
    rng = np.random.default_rng(2027)
    # Build a uniform-k=3 joint of shape (3, 3, n, 3): for each input state
    # and each node i, the last dim is the per-node output distribution.
    n = 2
    raw = rng.uniform(size=(3, 3, n, 3))
    # Normalize each per-node distribution to sum to 1.
    joint = raw / raw.sum(axis=-1, keepdims=True)
    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(3, 3))
    reconstructed = factored.to_joint()
    np.testing.assert_allclose(reconstructed, joint, atol=1e-12)


def test_from_joint_explicit_alphabet_heterogeneous_round_trip() -> None:
    """Heterogeneous-alphabet round-trip: to_joint then from_joint preserves factors."""
    rng = np.random.default_rng(2028)
    n = 2
    a = (2, 3)
    # Build factor 0 with shape (2, 3, 2); factor 1 with shape (2, 3, 3).
    f0_raw = rng.uniform(size=(2, 3, 2))
    f0 = f0_raw / f0_raw.sum(axis=-1, keepdims=True)
    f1_raw = rng.uniform(size=(2, 3, 3))
    f1 = f1_raw / f1_raw.sum(axis=-1, keepdims=True)
    factored = FactoredTPM(factors=[f0, f1], alphabet_sizes=a)
    joint = factored.to_joint()
    # Round-trip through from_joint
    reconstructed = FactoredTPM.from_joint(joint, alphabet_sizes=a)
    for i in range(n):
        np.testing.assert_allclose(
            reconstructed.factor(i), factored.factor(i), atol=1e-12
        )


def test_from_joint_to_joint_bit_exact_for_legacy_binary() -> None:
    """Stack([1-p, p]) is exact in IEEE-754 for p in [0, 1]; lock that in."""
    joint = np.full((2, 2, 2), 0.5)
    factored = FactoredTPM.from_joint(joint, alphabet_sizes=(2, 2))
    reconstructed = factored.to_joint()
    expected = np.stack([1.0 - joint, joint], axis=-1)
    np.testing.assert_array_equal(reconstructed, expected)  # not allclose — exact
