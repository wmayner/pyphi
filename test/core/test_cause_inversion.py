"""Reduced cause inversion (greedy sum-product) vs the dense implementation."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import exceptions
from pyphi.core.tpm import marginalization
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import CauseMarginals
from pyphi.core.tpm.marginalization import _cause_marginal_factored

from .inversion_oracle import dense_cause_marginal_reference as _dense_factors


def _normalized(raw: np.ndarray) -> np.ndarray:
    return raw / raw.sum(axis=-1, keepdims=True)


def _asymmetric_binary_factored(seed: int = 5) -> FactoredTPM:
    """5 binary units with deliberately unequal parent sets.

    parents: 0 <- {0, 3}; 1 <- {0, 1}; 2 <- {1}; 3 <- {2, 3, 4}; 4 <- {4}.
    Asymmetric on purpose: symmetric fixtures hide axis-order errors.
    """
    rng = np.random.default_rng(seed)
    parent_sets = [{0, 3}, {0, 1}, {1}, {2, 3, 4}, {4}]
    factors = []
    for parents in parent_sets:
        shape = tuple(2 if j in parents else 1 for j in range(5))
        factors.append(_normalized(rng.random((*shape, 2)) + 1e-3))
    return FactoredTPM(factors=factors)


def _kary_factored(seed: int = 7) -> FactoredTPM:
    """4 units with alphabets (2, 3, 2, 4) and unequal parent sets."""
    rng = np.random.default_rng(seed)
    alphabets = (2, 3, 2, 4)
    parent_sets = [{0, 1}, {1, 3}, {0, 2, 3}, {2}]
    factors = []
    for i, parents in enumerate(parent_sets):
        shape = tuple(alphabets[j] if j in parents else 1 for j in range(4))
        factors.append(_normalized(rng.random((*shape, alphabets[i])) + 1e-3))
    return FactoredTPM(
        factors=factors,
        state_space=tuple(tuple(range(a)) for a in alphabets),
    )


@pytest.mark.parametrize("state", [(0, 1, 0, 1, 0), (1, 0, 1, 1, 1)])
@pytest.mark.parametrize("system", [(1, 2), (0, 3, 4), (2,)])
def test_reduced_matches_dense_asymmetric_binary(state, system):
    factored = _asymmetric_binary_factored()
    reduced = _cause_marginal_factored(factored, state, system)
    dense = _dense_factors(factored, state, system)
    assert reduced.indices == system
    for i in system:
        assert reduced.factor(i).shape == dense[i].shape
        np.testing.assert_allclose(reduced.factor(i), dense[i], rtol=0, atol=1e-13)


@pytest.mark.parametrize("state", [(0, 2, 1, 3), (1, 0, 0, 2)])
@pytest.mark.parametrize("system", [(0, 2), (1,), (1, 2, 3)])
def test_reduced_matches_dense_kary(state, system):
    factored = _kary_factored()
    reduced = _cause_marginal_factored(factored, state, system)
    dense = _dense_factors(factored, state, system)
    for i in system:
        assert reduced.factor(i).shape == dense[i].shape
        np.testing.assert_allclose(reduced.factor(i), dense[i], rtol=0, atol=1e-13)


def test_different_states_give_different_marginals():
    """Genuine-difference guard: the comparison tests above must not be
    vacuously passing on state-independent outputs."""
    factored = _asymmetric_binary_factored()
    system = (1, 2)
    a = _cause_marginal_factored(factored, (0, 1, 0, 1, 0), system)
    b = _cause_marginal_factored(factored, (1, 0, 1, 1, 1), system)
    assert any(not np.allclose(a.factor(i), b.factor(i)) for i in system)


def test_full_substrate_system_is_bit_identical_to_dense():
    """With no background units the weight is exactly 1.0, so the reduced
    path must reproduce the dense path bit-for-bit."""
    factored = _asymmetric_binary_factored()
    state = (0, 1, 1, 0, 1)
    system = (0, 1, 2, 3, 4)
    reduced = _cause_marginal_factored(factored, state, system)
    dense = _dense_factors(factored, state, system)
    for i in system:
        assert np.array_equal(reduced.factor(i), dense[i])


def test_unreachable_state_raises():
    factors = [np.zeros((2, 2, 2)) for _ in range(2)]
    for f in factors:
        f[..., 0] = 1.0  # every unit always outputs 0
    factored = FactoredTPM(factors=factors)
    with pytest.raises(exceptions.StateUnreachableBackwardsError):
        _cause_marginal_factored(factored, state=(1, 1), node_indices=(0, 1))


def test_intractable_contraction_raises(monkeypatch):
    """All-to-all coupling with a tiny cap: every elimination step exceeds it."""
    rng = np.random.default_rng(11)
    n = 6
    factors = [_normalized(rng.random(((2,) * n) + (2,)) + 1e-3) for _ in range(n)]
    factored = FactoredTPM(factors=factors)
    monkeypatch.setattr(marginalization, "_MAX_INTERMEDIATE_ELEMENTS", 8)
    with pytest.raises(exceptions.IntractableCauseInversionError, match=r"\d+"):
        _cause_marginal_factored(factored, state=(0,) * n, node_indices=(0, 1))


def test_cause_marginals_value_semantics():
    rng = np.random.default_rng(3)
    f = rng.random((2, 1, 2))
    a = CauseMarginals({0: f})
    b = CauseMarginals({0: f.copy()})
    c = CauseMarginals({0: rng.random((2, 1, 2))})
    d = CauseMarginals({1: f})
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d
    assert a.indices == (0,)
    assert a.factor(0) is f
