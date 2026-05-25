"""Hypothesis property tests for FactoredTPM with k-ary alphabets.

Exercises the alphabet-generic internals against k in {2, 3, 4, 5}.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.core.tpm.factored import FactoredTPM

ALPHABET_SIZES = st.integers(min_value=2, max_value=5)


def _draw_alphabets(n: int) -> st.SearchStrategy[tuple[int, ...]]:
    return st.tuples(*([ALPHABET_SIZES] * n))


@st.composite
def _factored_strategy(draw: st.DrawFn, max_nodes: int = 4) -> FactoredTPM:
    n = draw(st.integers(min_value=2, max_value=max_nodes))
    alphabet_sizes = draw(_draw_alphabets(n))
    factors = []
    for i in range(n):
        shape = (*alphabet_sizes, alphabet_sizes[i])
        rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=10_000)))
        raw = rng.uniform(size=shape)
        normalized = raw / raw.sum(axis=-1, keepdims=True)
        factors.append(normalized)
    return FactoredTPM(
        factors=factors, state_space=tuple(tuple(range(a)) for a in alphabet_sizes)
    )


FAST_LANE = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

SLOW_LANE = settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _assert_round_trip(factored: FactoredTPM) -> None:
    """Core round-trip assertion shared by fast and slow tests."""
    joint = factored.to_joint()
    n = factored.n_nodes
    a = factored.alphabet_sizes
    ss = factored.state_space
    if len(set(a)) == 1:
        reconstructed = FactoredTPM.from_joint(joint, state_space=ss)
    else:
        explicit = np.zeros((*a, n, max(a)))
        for i in range(n):
            explicit[..., i, : a[i]] = factored.factor(i)
        reconstructed = FactoredTPM.from_joint(explicit, state_space=ss)
    for i in range(n):
        np.testing.assert_allclose(
            reconstructed.factor(i), factored.factor(i), atol=1e-10
        )


@FAST_LANE
@given(_factored_strategy())
def test_kary_to_joint_from_joint_round_trip(factored: FactoredTPM) -> None:
    """to_joint then reconstruct factors via from_joint should yield equal factors."""
    _assert_round_trip(factored)


@FAST_LANE
@given(_factored_strategy(max_nodes=3))
def test_kary_condition_commutes_with_reconstruction(factored: FactoredTPM) -> None:
    """condition(fixed).to_joint()[0] agrees with the equivalent slice on the joint."""
    n = factored.n_nodes
    if n < 2:
        return
    fixed = {0: 0}
    cond_factored = factored.condition(fixed)
    joint = factored.to_joint()
    sliced = joint[0]
    reconstructed = cond_factored.to_joint()
    # conditioning on dim 0 == 0 broadcasts that single row to all states along
    # dim 0; the first row of the reconstruction must match joint[0] exactly.
    np.testing.assert_allclose(reconstructed[0], sliced, atol=1e-10)


@FAST_LANE
@given(_factored_strategy(max_nodes=3))
def test_kary_factors_each_sum_to_one(factored: FactoredTPM) -> None:
    """Validation invariant — sums must be exactly 1 along the last dim."""
    for i in range(factored.n_nodes):
        sums = factored.factor(i).sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-12)


@pytest.mark.slow
@SLOW_LANE
@given(_factored_strategy())
def test_kary_round_trip_slow(factored: FactoredTPM) -> None:
    """Slow-lane variant of round-trip with max_examples=500."""
    _assert_round_trip(factored)


def test_k3_explicit_round_trip() -> None:
    """Spot-check k=3 with a hand-crafted FactoredTPM."""
    f0 = np.full((3, 3, 3), 1.0 / 3.0)
    f1 = np.full((3, 3, 3), 1.0 / 3.0)
    factored = FactoredTPM(factors=[f0, f1], state_space=((0, 1, 2), (0, 1, 2)))
    joint = factored.to_joint()
    reconstructed = FactoredTPM.from_joint(joint, state_space=((0, 1, 2), (0, 1, 2)))
    for i in range(2):
        np.testing.assert_allclose(
            reconstructed.factor(i), factored.factor(i), atol=1e-12
        )


def test_k4_explicit_construction_validates() -> None:
    """k=4 construction passes validation."""
    f0 = np.full((4, 4), 0.25)
    f1 = np.full((4, 4), 0.25)
    f0_full = np.broadcast_to(f0[:, np.newaxis, :], (4, 4, 4)).copy()
    f1_full = np.broadcast_to(f1[np.newaxis, :, :], (4, 4, 4)).copy()
    factored = FactoredTPM(
        factors=[f0_full, f1_full], state_space=((0, 1, 2, 3), (0, 1, 2, 3))
    )
    assert factored.n_nodes == 2
    assert factored.alphabet_sizes == (4, 4)
