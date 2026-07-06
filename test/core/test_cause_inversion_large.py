"""Large-substrate cause inversion vs independent references.

These substrates are far beyond the dense implementation's reach (the
joint likelihood would have 2^40 entries), so each test checks the
reduced path against an independently written computation instead.
"""

from __future__ import annotations

import time

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import _cause_marginal_factored

from .inversion_oracle import dense_cause_marginal_reference

N_LARGE = 40


def _normalized(raw: np.ndarray) -> np.ndarray:
    return raw / raw.sum(axis=-1, keepdims=True)


def _chain_factor(n: int, parents: tuple[int, ...], rng) -> np.ndarray:
    """Binary factor over ``n`` input axes with real extent on ``parents``."""
    shape = tuple(2 if j in parents else 1 for j in range(n))
    return _normalized(rng.random((*shape, 2)) + 0.05)


def test_disconnected_block_matches_small_dense_oracle():
    """8-unit block containing the system + separate 32-unit block.

    The disconnected block's likelihood contributes a constant that
    cancels in pr_bg / norm, so the 40-unit result must match the dense
    oracle run on the 8-unit block alone.
    """
    rng = np.random.default_rng(42)
    system = (2, 3, 4)

    # Block A: units 0..7, a chain (unit j <- {j-1, j}; unit 0 <- {0}).
    # Build each factor twice from the same values: once over 40 axes,
    # once over 8 axes, so the two substrates share identical numbers.
    small_factors = []
    large_factors = []
    for j in range(8):
        parents = (j,) if j == 0 else (j - 1, j)
        f_small = _chain_factor(8, parents, rng)
        small_factors.append(f_small)
        pad = (1,) * (N_LARGE - 8)
        large_factors.append(f_small.reshape((*f_small.shape[:-1], *pad, 2)))

    # Block B: units 8..39, a chain among themselves, no cross edges.
    for j in range(8, N_LARGE):
        parents = (j,) if j == 8 else (j - 1, j)
        large_factors.append(_chain_factor(N_LARGE, parents, rng))

    state_small = tuple(int(b) for b in rng.integers(0, 2, size=8))
    state_large = state_small + tuple(
        int(b) for b in rng.integers(0, 2, size=N_LARGE - 8)
    )

    large = FactoredTPM(factors=large_factors)
    small = FactoredTPM(factors=small_factors)

    reduced = _cause_marginal_factored(large, state_large, system)
    oracle = dense_cause_marginal_reference(small, state_small, system)

    for i in system:
        got = np.squeeze(reduced.factor(i))
        want = np.squeeze(oracle[i])
        assert got.shape == want.shape
        # Not bit-exact: the disconnected block's constant cancels in a
        # float division.
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)


def _forward_backward_weight(g, relevant_axis):
    """Normalized marginal weight over one axis of a chain of pairwise
    likelihood slices — an independent sequential evaluation of the
    sum-product for chain topology. ``g[j]`` is unit j's likelihood slice
    as a small dense array over (s_{j-1}, s_j) (unit 0: over (s_0,))."""
    n = len(g)
    # Forward: m[j](s_j) = sum_{s_{j-1}} m[j-1](s_{j-1}) g[j](s_{j-1}, s_j)
    m = [g[0]]
    for j in range(1, n):
        m.append(m[j - 1] @ g[j])
    # Backward: b[j](s_j) = sum_{s_{j+1}} g[j+1](s_j, s_{j+1}) b[j+1](s_{j+1})
    b = [np.ones(2)] * n
    for j in range(n - 2, -1, -1):
        b[j] = g[j + 1] @ b[j + 1]
    r = relevant_axis
    pr = m[r] * b[r]  # unnormalized marginal over s_r
    return pr / pr.sum()


def test_connected_chain_matches_forward_backward():
    """40-unit connected chain, system in the middle, vs a transfer-matrix
    computation of the background weight and outputs."""
    rng = np.random.default_rng(7)
    system = (18, 19, 20, 21)

    factors = []
    dense_slices = []  # small (2,)- or (2, 2)-shaped likelihood slices
    state = tuple(int(bit) for bit in rng.integers(0, 2, size=N_LARGE))
    for j in range(N_LARGE):
        parents = (j,) if j == 0 else (j - 1, j)
        f = _chain_factor(N_LARGE, parents, rng)
        factors.append(f)
        dense_slices.append(np.squeeze(f[..., state[j]]))

    factored = FactoredTPM(factors=factors)
    t0 = time.perf_counter()
    reduced = _cause_marginal_factored(factored, state, system)
    elapsed = time.perf_counter() - t0
    # Feasibility gate: dense evaluation would need a 2^40 array. Generous
    # bound so CI noise cannot flake it, while still catching any
    # accidental fallback to dense evaluation.
    assert elapsed < 10.0

    # Relevant background axis: only unit 17 (parent of system unit 18).
    weight = _forward_backward_weight(dense_slices, 17)

    # Reference outputs. Unit 18 depends on (s_17, s_18): contract the
    # weight over s_17. Units 19..21 depend only on system axes: their
    # output is the forward factor itself (the weight sums to 1).
    f18 = np.squeeze(factors[18])  # (2, 2, 2): s_17, s_18, out
    want_18 = np.einsum("abk,a->bk", f18, weight)
    got_18 = np.squeeze(reduced.factor(18))
    np.testing.assert_allclose(got_18, want_18, rtol=0, atol=1e-12)

    for i in (19, 20, 21):
        got = np.squeeze(reduced.factor(i))
        want = np.squeeze(factors[i])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_two_backgrounds_give_different_weights():
    """Genuine-difference guard for the chain test: flipping the state of
    the relevant background unit changes the system's cause factors."""
    rng = np.random.default_rng(7)
    system = (18, 19, 20, 21)
    factors = [
        _chain_factor(N_LARGE, (j,) if j == 0 else (j - 1, j), rng)
        for j in range(N_LARGE)
    ]
    factored = FactoredTPM(factors=factors)
    state = [0] * N_LARGE
    a = _cause_marginal_factored(factored, tuple(state), system)
    state[17] = 1  # the relevant background parent
    b = _cause_marginal_factored(factored, tuple(state), system)
    assert not np.allclose(a.factor(18), b.factor(18))


def test_system_level_proper_cause_marginal_on_large_substrate():
    """End-to-end: Substrate/System over 40 units computes
    proper_cause_marginal without materializing the joint."""
    from pyphi import config
    from pyphi.substrate import Substrate
    from pyphi.system import System

    rng = np.random.default_rng(3)
    factors = [
        _chain_factor(N_LARGE, (j,) if j == 0 else (j - 1, j), rng)
        for j in range(N_LARGE)
    ]
    sub = Substrate(marginals=factors)
    state = tuple(int(bit) for bit in rng.integers(0, 2, size=N_LARGE))
    with config.override(validate_system_states=False):
        sys_ = System(sub, state=state, node_indices=(18, 19, 20, 21))
        proper = sys_.proper_cause_marginal
    assert proper.n_nodes == 4
    for slot in range(4):
        np.testing.assert_allclose(proper.factor(slot).sum(axis=-1), 1.0, atol=1e-10)
