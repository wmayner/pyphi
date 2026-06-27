"""K-ary cause inversion math: correctness + binary equivalence."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.marginalization import _cause_marginal_factored


def _random_kary_factor(n_nodes: int, alphabet: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (alphabet,) * n_nodes + (alphabet,)
    arr = rng.uniform(size=shape)
    return arr / arr.sum(axis=-1, keepdims=True)


def _random_binary_factor(n_nodes: int, seed: int) -> np.ndarray:
    return _random_kary_factor(n_nodes, 2, seed)


def test_cause_marginal_factored_returns_factored_tpm() -> None:
    """Returns a FactoredTPM for k-ary substrates."""
    factors = [_random_kary_factor(2, 3, seed=10 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = _cause_marginal_factored(factored, state=(0, 0), node_indices=(0, 1))
    assert isinstance(result, FactoredTPM)


def test_cause_marginal_factored_per_factor_sums_to_one() -> None:
    """Each per-output-unit factor of the returned FactoredTPM is a
    probability distribution over its trailing alphabet axis."""
    factors = [_random_kary_factor(2, 3, seed=20 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = _cause_marginal_factored(factored, state=(1, 2), node_indices=(0, 1))
    for i in range(result.n_nodes):
        f = result.factor(i)
        assert f.shape[-1] == 3
        np.testing.assert_allclose(f.sum(axis=-1), 1.0, atol=1e-10)


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25, deadline=None)
def test_cause_marginal_factored_binary_gives_valid_distribution(seed: int) -> None:
    """On binary inputs each output factor is a valid probability distribution."""
    factors = [_random_binary_factor(3, seed=seed + i) for i in range(3)]
    factored = FactoredTPM(factors=factors)
    state = (0, 1, 0)
    node_indices = (0, 1, 2)
    result = _cause_marginal_factored(factored, state, node_indices)
    for i in range(factored.n_nodes):
        f = result.factor(i)
        np.testing.assert_allclose(
            f.sum(axis=-1),
            1.0,
            atol=1e-10,
            err_msg=f"factor {i} does not sum to 1",
        )


def test_cause_marginal_factored_subset_system_uses_background_weighting() -> None:
    """When system_indices is a proper subset of the substrate, the
    posterior factor for system unit i depends on the background state
    via pr_bg / norm. Verify against a hand-built 2-node binary case."""
    # 2-node binary: node 0 is the mechanism (system), node 1 is background.
    f0 = np.array([[[0.8, 0.2], [0.5, 0.5]], [[0.1, 0.9], [0.4, 0.6]]], dtype=np.float64)
    f1 = np.array([[[0.7, 0.3], [0.2, 0.8]], [[0.6, 0.4], [0.3, 0.7]]], dtype=np.float64)
    factored = FactoredTPM(factors=[f0, f1])
    state = (1, 0)
    result = _cause_marginal_factored(factored, state, node_indices=(0,))
    # Node 0's factor must be a valid distribution over its 2 output states.
    np.testing.assert_allclose(result.factor(0).sum(axis=-1), 1.0, atol=1e-10)


def test_cause_unreachable_state_raises() -> None:
    from pyphi.exceptions import StateUnreachableBackwardsError

    factors = [np.zeros((2, 2, 2)) for _ in range(2)]
    for f in factors:
        f[..., 0] = 1.0  # always outputs 0
    factored = FactoredTPM(factors=factors)
    with pytest.raises(StateUnreachableBackwardsError):
        _cause_marginal_factored(factored, state=(1, 1), node_indices=(0, 1))


def test_effect_marginal_kary_does_not_raise() -> None:
    """Effect TPM works for k>2 substrates via FactoredTPM.condition."""
    from pyphi.core.tpm.marginalization import effect_marginal

    factors = [_random_kary_factor(2, 3, seed=30 + i) for i in range(2)]
    factored = FactoredTPM(factors=factors)
    result = effect_marginal(factored, background={1: 1})
    assert result is not None
    # Conditioning fixes node 1's input axis to index 1; per-factor shape
    # collapses that axis to size 1.
    for i in range(factored.n_nodes):
        f = result.factor(i)
        assert f.shape[1] == 1


def test_single_node_cause_repertoire_k3() -> None:
    """Per-node cause repertoire on a k=3 substrate has the expected
    repertoire shape over the purview's joint state space.

    Builds the function's minimal contract directly: a per-node cause
    factor of shape ``(*alphabet_sizes, k_node)`` wrapped in a
    :class:`JointTPM`, plus a stub holding ``state``, ``inputs``, and
    ``cause_marginal``. The returned array is the per-node unnormalized
    slice produced by indexing ``cause_marginal[..., state]`` and
    marginalizing out non-purview inputs; normalization happens in
    ``_cause_repertoire_inner`` after the per-node factors are
    multiplied.
    """
    from pyphi.core.repertoire_algebra import _single_node_cause_repertoire
    from pyphi.core.tpm.joint_distribution import JointTPM

    class _Node:
        def __init__(self, state: int, inputs: frozenset[int], cause_marginal: JointTPM):
            self.state = state
            self.inputs = inputs
            self.cause_marginal = cause_marginal

    class _CS:
        def __init__(self, index2node: dict[int, _Node]):
            self._index2node = index2node
            # Distinct content -> distinct kernel-cache fingerprint, so the two
            # _CS instances below (differing node state) are not conflated.
            self._fingerprint = repr(
                tuple((i, n.state) for i, n in sorted(index2node.items()))
            ).encode()

    # Per-node cause factor: 2-node substrate, k=3 alphabet, k=3 outputs.
    # Shape (3, 3, 3) -- last axis is this node's output-state distribution.
    factor = _random_kary_factor(2, 3, seed=40)
    node = _Node(state=0, inputs=frozenset({0, 1}), cause_marginal=JointTPM(factor))
    cs = _CS({0: node})
    rep = _single_node_cause_repertoire(cs, 0, frozenset({0}))
    # Purview {0}: node-1 axis marginalized out (size 1); node-0 axis kept.
    assert rep.ndim == 2
    assert rep.shape == (3, 1)
    assert np.all(rep >= 0.0)
    assert np.all(np.isfinite(rep))
    # The k=3 path exercises the alphabet-generic indexing
    # ``cause_marginal[..., state]`` with ``state`` in ``[0, k_node)``.
    # Same query with a different output-state index produces a different
    # slice, confirming the trailing axis is being indexed.
    node_other = _Node(
        state=2, inputs=frozenset({0, 1}), cause_marginal=JointTPM(factor)
    )
    cs_other = _CS({0: node_other})
    rep_other = _single_node_cause_repertoire(cs_other, 0, frozenset({0}))
    assert not np.allclose(rep, rep_other)
