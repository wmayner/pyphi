"""K-ary cause inversion math: correctness + binary equivalence."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import _cause_tpm_factored_binary
from pyphi.core.tpm.marginalization import _cause_tpm_factored_kary


def _random_kary_factor(n_nodes: int, alphabet: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (alphabet,) * n_nodes + (alphabet,)
    arr = rng.uniform(size=shape)
    return arr / arr.sum(axis=-1, keepdims=True)


def _random_binary_factor(n_nodes: int, seed: int) -> np.ndarray:
    return _random_kary_factor(n_nodes, 2, seed)


def test_cause_kary_returns_factored_tpm() -> None:
    """The native k-ary path returns a FactoredTPM."""
    factors = [_random_kary_factor(2, 3, seed=10 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = _cause_tpm_factored_kary(factored, state=(0, 0), node_indices=(0, 1))
    assert isinstance(result, FactoredTPM)


def test_cause_kary_per_factor_sums_to_one() -> None:
    """Each per-output-unit factor of the returned FactoredTPM is a
    probability distribution over its trailing alphabet axis."""
    factors = [_random_kary_factor(2, 3, seed=20 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = _cause_tpm_factored_kary(factored, state=(1, 2), node_indices=(0, 1))
    for i in range(result.n_nodes):
        f = result.factor(i)
        assert f.shape[-1] == 3
        np.testing.assert_allclose(f.sum(axis=-1), 1.0, atol=1e-10)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25, deadline=None)
def test_cause_kary_binary_equivalent_to_binary_path(seed: int) -> None:
    """On binary inputs the k-ary path and the binary path produce
    equivalent factors (within atol=1e-10) per output unit."""
    factors = [_random_binary_factor(3, seed=seed + i) for i in range(3)]
    factored = FactoredTPM(factors=factors)
    state = (0, 1, 0)
    node_indices = (0, 1, 2)
    kary = _cause_tpm_factored_kary(factored, state, node_indices)
    binary = _cause_tpm_factored_binary(factored, state, node_indices)
    for i in range(factored.n_nodes):
        np.testing.assert_allclose(
            kary.factor(i),
            binary.factor(i),
            atol=1e-10,
            err_msg=f"factor {i} disagrees",
        )


def test_cause_kary_subset_system_uses_background_weighting() -> None:
    """When system_indices is a proper subset of the substrate, the
    posterior factor for system unit i depends on the background state
    via pr_bg / norm. Verify against a hand-built 2-node binary case."""
    # 2-node binary: node 0 is the mechanism (system), node 1 is background.
    f0 = np.array([[[0.8, 0.2], [0.5, 0.5]], [[0.1, 0.9], [0.4, 0.6]]], dtype=np.float64)
    f1 = np.array([[[0.7, 0.3], [0.2, 0.8]], [[0.6, 0.4], [0.3, 0.7]]], dtype=np.float64)
    factored = FactoredTPM(factors=[f0, f1])
    state = (1, 0)
    binary = _cause_tpm_factored_binary(factored, state, node_indices=(0,))
    kary = _cause_tpm_factored_kary(factored, state, node_indices=(0,))
    np.testing.assert_allclose(kary.factor(0), binary.factor(0), atol=1e-10)


def test_cause_unreachable_state_raises() -> None:
    from pyphi.exceptions import StateUnreachableBackwardsError

    factors = [np.zeros((2, 2, 2)) for _ in range(2)]
    for f in factors:
        f[..., 0] = 1.0  # always outputs 0
    factored = FactoredTPM(factors=factors)
    with pytest.raises(StateUnreachableBackwardsError):
        _cause_tpm_factored_kary(factored, state=(1, 1), node_indices=(0, 1))
