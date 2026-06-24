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
    return FactoredTPM(factors=[f0, f1])


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
        FactoredTPM(factors=[bad, good])


def test_factored_tpm_validation_rejects_alphabet_lt_2() -> None:
    f0 = np.array([[[1.0]]], dtype=np.float64)
    with pytest.raises(InvalidTPM, match="alphabet"):
        FactoredTPM(factors=[f0])


def test_factored_tpm_equality() -> None:
    a = _two_node_factored()
    b = _two_node_factored()
    assert a == b
    assert (a != b) is False


def test_factored_tpm_repr() -> None:
    f = _two_node_factored()
    r = repr(f)
    assert "FactoredTPM" in r
    assert "2 units" in r  # state-by-node display card
    # The LOW-verbosity compact form still carries the precise attributes.
    import pyphi

    with pyphi.config.override(repr_verbosity=0):
        assert "n_nodes=2" in repr(f)


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
    a = FactoredTPM(factors=[f0a, f0a.copy()])
    b = FactoredTPM(factors=[f0b, f0a.copy()])
    assert a == b
    assert hash(a) == hash(b)


def test_factored_tpm_rejects_factor_count_mismatch() -> None:
    """n_nodes (from factors) must match len(state_space) when state_space is per-node."""
    f = np.full((2, 2, 2), 0.5, dtype=np.float64)
    with pytest.raises(InvalidTPM, match="state_space"):
        FactoredTPM(factors=[f, f, f], state_space=((0, 1), (0, 1)))
    with pytest.raises(InvalidTPM, match="state_space"):
        FactoredTPM(factors=[f], state_space=((0, 1), (0, 1)))


# --- round-trip tests ---


def _random_joint_tpm(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random binary joint TPM, shape (2,)*n + (n,) — entries are P(node_i=1)."""
    return rng.uniform(size=(2,) * n + (n,))


def test_from_joint_round_trip_n2() -> None:
    rng = np.random.default_rng(42)
    joint = _random_joint_tpm(rng, 2)
    factored = FactoredTPM.from_joint(joint)
    reconstructed = factored.to_joint()
    p_on = joint
    explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
    np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)


def test_from_joint_round_trip_n3() -> None:
    rng = np.random.default_rng(99)
    joint = _random_joint_tpm(rng, 3)
    factored = FactoredTPM.from_joint(joint)
    reconstructed = factored.to_joint()
    p_on = joint
    explicit_joint = np.stack([1.0 - p_on, p_on], axis=-1)
    np.testing.assert_allclose(reconstructed, explicit_joint, atol=1e-12)


def test_from_joint_invalid_shape_raises() -> None:
    bad = np.zeros((2, 2))
    with pytest.raises(ValueError, match="shape"):
        FactoredTPM.from_joint(bad)


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
        factored = FactoredTPM.from_joint(joint)
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
    factored = FactoredTPM.from_joint(joint, state_space=((0, 1, 2), (0, 1, 2)))
    reconstructed = factored.to_joint()
    np.testing.assert_allclose(reconstructed, joint, atol=1e-12)


def test_from_joint_explicit_alphabet_heterogeneous_round_trip() -> None:
    """Heterogeneous-alphabet round-trip: to_joint then from_joint preserves factors."""
    rng = np.random.default_rng(2028)
    n = 2
    # Build factor 0 with shape (2, 3, 2); factor 1 with shape (2, 3, 3).
    f0_raw = rng.uniform(size=(2, 3, 2))
    f0 = f0_raw / f0_raw.sum(axis=-1, keepdims=True)
    f1_raw = rng.uniform(size=(2, 3, 3))
    f1 = f1_raw / f1_raw.sum(axis=-1, keepdims=True)
    factored = FactoredTPM(factors=[f0, f1], state_space=((0, 1), (0, 1, 2)))
    joint = factored.to_joint()
    # Round-trip through from_joint
    reconstructed = FactoredTPM.from_joint(joint, state_space=((0, 1), (0, 1, 2)))
    for i in range(n):
        np.testing.assert_allclose(
            reconstructed.factor(i), factored.factor(i), atol=1e-12
        )


def test_from_joint_to_joint_bit_exact_for_legacy_binary() -> None:
    """Stack([1-p, p]) is exact in IEEE-754 for p in [0, 1]; lock that in."""
    joint = np.full((2, 2, 2), 0.5)
    factored = FactoredTPM.from_joint(joint)
    reconstructed = factored.to_joint()
    expected = np.stack([1.0 - joint, joint], axis=-1)
    np.testing.assert_array_equal(reconstructed, expected)  # not allclose — exact


# --- xarray backend (optional) ---

xarray = pytest.importorskip("xarray")


def _two_node_factored_xarray() -> FactoredTPM:
    f0 = np.array(
        [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
        dtype=np.float64,
    )
    f1 = f0.copy()
    return FactoredTPM(factors=[f0, f1], backend="xarray")


def test_factored_tpm_xarray_backend_selectable() -> None:
    f = _two_node_factored_xarray()
    assert f.n_nodes == 2
    assert f.alphabet_sizes == (2, 2)


def test_factored_tpm_xarray_factor_equals_ndarray_factor() -> None:
    """Cross-backend equality on identical factor data."""
    nd = FactoredTPM(
        factors=[np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)],
        backend="ndarray",
    )
    xr = FactoredTPM(
        factors=[np.full((2, 2, 2), 0.5), np.full((2, 2, 2), 0.5)],
        backend="xarray",
    )
    assert nd == xr


def test_factored_per_node_matches_joint_marginalize() -> None:
    """Per-node factor reads from FactoredTPM match the legacy
    joint-then-marginalize computation."""
    from pyphi.core.tpm.joint import JointTPM  # noqa: F401  (referenced for context)

    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    factored = FactoredTPM.from_joint(joint_arr)
    p_on_per_node = [joint_arr[..., i] for i in range(3)]
    for i in range(3):
        factor_i = factored.factor(i)
        # factor_i shape (2,2,2,2); factor_i[..., 1] is P(node_i = 1)
        np.testing.assert_allclose(factor_i[..., 1], p_on_per_node[i], atol=1e-12)


# --- state_space tests ---


def test_factored_tpm_default_state_space_is_integer_labels() -> None:
    """When state_space is omitted, integer labels 0..k-1 are inferred per node."""
    f = _two_node_factored()  # binary, no state_space
    assert f.state_space == ((0, 1), (0, 1))


def test_factored_tpm_uniform_state_space_string_labels() -> None:
    """A flat tuple of strings is parsed as uniform across all nodes."""
    f0 = np.full((3, 3, 3), 1.0 / 3.0)
    f = FactoredTPM(factors=[f0, f0.copy()], state_space=("LOW", "MID", "HIGH"))
    assert f.state_space == (("LOW", "MID", "HIGH"), ("LOW", "MID", "HIGH"))
    assert f.alphabet_sizes == (3, 3)


def test_factored_tpm_per_node_state_space() -> None:
    """A tuple of tuples is parsed as per-node labels."""
    f_binary = np.full((2, 3, 2), 0.5)
    f_ternary = np.full((2, 3, 3), 1.0 / 3.0)
    f = FactoredTPM(
        factors=[f_binary, f_ternary],
        state_space=(("OFF", "ON"), ("LOW", "MID", "HIGH")),
    )
    assert f.state_space == (("OFF", "ON"), ("LOW", "MID", "HIGH"))
    assert f.alphabet_sizes == (2, 3)


def test_factored_tpm_state_space_length_mismatch_raises() -> None:
    """state_space length must match factor count."""
    f0 = np.full((2, 2, 2), 0.5)
    with pytest.raises(InvalidTPM, match="state_space"):
        FactoredTPM(
            factors=[f0, f0.copy()],
            state_space=(
                ("OFF", "ON"),
                ("LOW", "HIGH"),
                ("EXTRA",),
            ),  # 3 entries, 2 factors
        )


def test_factored_tpm_state_space_label_alphabet_mismatch_raises() -> None:
    """state_space[i] length must match factor[i]'s last-dim size."""
    f_binary = np.full((2, 2, 2), 0.5)
    with pytest.raises(InvalidTPM, match="state_space"):
        FactoredTPM(
            factors=[f_binary, f_binary.copy()],
            state_space=(
                ("L", "M", "H"),
                ("L", "M", "H"),
            ),  # 3 labels, but factor has alphabet 2
        )


def test_factored_tpm_alphabet_sizes_not_constructor_kwarg() -> None:
    """alphabet_sizes is no longer a constructor parameter."""
    f0 = np.full((2, 2, 2), 0.5)
    with pytest.raises(TypeError, match="alphabet_sizes"):
        FactoredTPM(factors=[f0, f0.copy()], alphabet_sizes=(2, 2))  # type: ignore[call-arg]


def test_factored_tpm_rejects_reduced_dimension_factor() -> None:
    # 2-node binary substrate: full-dim factor is (2, 2, 2). A reduced factor
    # (2, 2) spans only one leading axis and is silently accepted today, then
    # crashes downstream. It must be rejected at construction.
    full = np.full((2, 2, 2), 0.5)
    reduced = np.full((2, 2), 0.5)
    with pytest.raises(InvalidTPM, match="leading axes"):
        FactoredTPM(factors=[full, reduced], state_space=((0, 1), (0, 1)))
