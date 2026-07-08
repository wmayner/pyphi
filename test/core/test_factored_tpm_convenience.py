"""Dense-TPM convenience methods on FactoredTPM.

``is_deterministic``, ``permute_nodes``, and ``subtpm`` — restored (on the
canonical factored type) from the retired dense ``JointTPM`` container.
"""

import numpy as np
import pytest

from pyphi import examples
from pyphi.core.tpm.factored import FactoredTPM


def _random(n_nodes: int, alphabet: int, seed: int) -> FactoredTPM:
    rng = np.random.default_rng(seed)
    factors = []
    for _ in range(n_nodes):
        f = rng.uniform(size=(alphabet,) * n_nodes + (alphabet,))
        factors.append(f / f.sum(axis=-1, keepdims=True))
    return FactoredTPM(factors=factors)


def test_is_deterministic_true():
    assert examples.xor_substrate().factored_tpm.is_deterministic()


def test_is_deterministic_false():
    assert not _random(2, 2, seed=0).is_deterministic()


def test_permute_nodes_returns_factored_tpm():
    permuted = examples.grid3_substrate().factored_tpm.permute_nodes((2, 0, 1))
    assert isinstance(permuted, FactoredTPM)
    assert permuted.n_nodes == 3


def test_permute_nodes_roundtrips_to_identity():
    f = examples.grid3_substrate().factored_tpm
    perm = (2, 0, 1)
    inverse = tuple(int(i) for i in np.argsort(perm))
    back = f.permute_nodes(perm).permute_nodes(inverse)
    assert np.array_equal(back.to_joint(), f.to_joint())


def test_permute_nodes_reorders_and_transposes_factors():
    f = _random(3, 2, seed=1)
    perm = (1, 2, 0)
    permuted = f.permute_nodes(perm)
    for new_pos, old_node in enumerate(perm):
        factor = f.factor(old_node)
        expected = np.transpose(factor, (*perm, factor.ndim - 1))
        assert np.array_equal(permuted.factor(new_pos), expected)


def test_permute_nodes_carries_state_space_and_labels():
    f = FactoredTPM(
        factors=[np.full((2, 3, 2), 0.5), np.full((2, 3, 3), 1 / 3)],
        state_space=(("off", "on"), ("lo", "mid", "hi")),
        node_labels=("A", "B"),
    )
    permuted = f.permute_nodes((1, 0))
    assert permuted.state_space == (("lo", "mid", "hi"), ("off", "on"))
    assert permuted.node_labels == ("B", "A")


def test_permute_nodes_wrong_length_raises():
    with pytest.raises(ValueError):
        _random(2, 2, seed=2).permute_nodes((0,))


def test_subtpm_drops_fixed_nodes():
    f = examples.grid3_substrate().factored_tpm
    sub = f.subtpm((0,), (0,))
    assert isinstance(sub, FactoredTPM)
    assert sub.n_nodes == 2  # free nodes are 1 and 2
    conditioned = f.condition({0: 0})
    assert np.array_equal(sub.factor(0), np.squeeze(conditioned.factor(1), axis=0))
    assert np.array_equal(sub.factor(1), np.squeeze(conditioned.factor(2), axis=0))


def test_subtpm_all_fixed_leaves_no_free_nodes():
    f = _random(2, 2, seed=3)
    with pytest.raises((ValueError, Exception)):
        f.subtpm((0, 1), (0, 0))  # no free nodes → not a valid FactoredTPM
