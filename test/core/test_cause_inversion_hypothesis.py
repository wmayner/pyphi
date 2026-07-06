"""Property-based cross-validation: reduced inversion vs the dense oracle."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import _cause_marginal_factored

from .inversion_oracle import dense_cause_marginal_reference


@st.composite
def inversion_cases(draw):
    """Random factored TPM + state + nonempty system subset.

    Parent sets, alphabets, and the system subset all vary independently;
    probabilities come from a seeded generator (bounded away from zero so
    every state is reachable and the oracle's norm > 0).
    """
    n = draw(st.integers(min_value=2, max_value=6))
    alphabets = tuple(draw(st.integers(min_value=2, max_value=3)) for _ in range(n))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    factors = []
    for i in range(n):
        parents = draw(st.frozensets(st.integers(min_value=0, max_value=n - 1)))
        shape = tuple(alphabets[j] if j in parents else 1 for j in range(n))
        raw = rng.random((*shape, alphabets[i])) + 1e-3
        factors.append(raw / raw.sum(axis=-1, keepdims=True))
    factored = FactoredTPM(
        factors=factors,
        state_space=tuple(tuple(range(a)) for a in alphabets),
    )
    state = tuple(
        draw(st.integers(min_value=0, max_value=alphabets[j] - 1)) for j in range(n)
    )
    size = draw(st.integers(min_value=1, max_value=n))
    system = tuple(sorted(draw(st.permutations(tuple(range(n))))[:size]))
    return factored, state, system


@given(inversion_cases())
@settings(max_examples=300, deadline=None)
def test_reduced_matches_dense_oracle(case):
    factored, state, system = case
    reduced = _cause_marginal_factored(factored, state, system)
    dense = dense_cause_marginal_reference(factored, state, system)
    assert reduced.indices == system
    for i in system:
        assert reduced.factor(i).shape == dense[i].shape
        np.testing.assert_allclose(reduced.factor(i), dense[i], rtol=0, atol=1e-12)
