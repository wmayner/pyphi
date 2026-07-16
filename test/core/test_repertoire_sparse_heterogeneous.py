"""Regression: cause/effect repertoires on sparse + heterogeneous-alphabet
networks, and background conditioning on connectivity-sparse factors
(size-1 non-input axes) at nonzero background states."""

import numpy as np
import pytest

from pyphi import Direction
from pyphi import Substrate
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.distribution import repertoire_shape
from pyphi.system import System


def _sparse_het_substrate():
    # node0 (k=3), node1 (k=3) -> node2 (k=4). Sparse cm: 0->2, 1->2.
    alph = (3, 3, 4)
    f0 = np.full((*alph, 3), 1 / 3)
    f1 = np.full((*alph, 3), 1 / 3)
    core = np.zeros((3, 3, 4))
    for a in range(3):
        for b in range(3):
            core[a, b, (a + b) % 4] = 1.0
    f2 = np.broadcast_to(core[:, :, None, :], (*alph, 4)).copy()
    cm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    return Substrate(
        marginals=[f0, f1, f2],
        state_space=((0, 1, 2), (0, 1, 2), (0, 1, 2, 3)),
        cm=cm,
    )


def test_sparse_het_cause_repertoire_shape():
    sub = _sparse_het_substrate()
    s = System(substrate=sub, state=(0, 0, 0), node_indices=(0, 1, 2))
    r = s.repertoire(Direction.CAUSE, (2,), (0,))  # mechanism k=4, purview k=3
    expected = repertoire_shape(
        s.node_indices, (0,), alphabet_sizes=sub.factored_tpm.alphabet_sizes
    )
    assert r.shape == tuple(expected)
    assert np.isclose(r.sum(), 1.0)


def test_sparse_het_effect_repertoire_shape():
    sub = _sparse_het_substrate()
    s = System(substrate=sub, state=(0, 0, 0), node_indices=(0, 1, 2))
    r = s.repertoire(Direction.EFFECT, (0,), (2,))  # mechanism k=3, purview k=4
    expected = repertoire_shape(
        s.node_indices, (2,), alphabet_sizes=sub.factored_tpm.alphabet_sizes
    )
    assert r.shape == tuple(expected)
    assert np.isclose(r.sum(), 1.0)


def _sparse_binary_factors():
    # 0 <-> 1 copy each other; 2 has a self-loop. Non-input axes have size 1.
    f0 = np.zeros((1, 2, 1, 2))
    f0[0, 0, 0] = [1, 0]
    f0[0, 1, 0] = [0, 1]
    f1 = np.zeros((2, 1, 1, 2))
    f1[0, 0, 0] = [1, 0]
    f1[1, 0, 0] = [0, 1]
    f2 = np.zeros((1, 1, 2, 2))
    f2[0, 0, 0] = [1, 0]
    f2[0, 0, 1] = [0, 1]
    return [f0, f1, f2]


def test_background_conditioning_on_size1_axis_matches_dense():
    factors = _sparse_binary_factors()
    sparse_sub = Substrate.from_factored(FactoredTPM(factors=factors))
    dense = [np.broadcast_to(f, (2, 2, 2, 2)).copy() for f in factors]
    dense_sub = Substrate.from_factored(FactoredTPM(factors=dense))
    # Node 2 is background, fixed ON — a size-1 non-input axis of the free
    # units' factors. The sparse and dense forms are the same TPM, so the
    # analysis must agree.
    state = (0, 0, 1)
    phi_sparse = System(sparse_sub, state, (0, 1)).sia().phi
    phi_dense = System(dense_sub, state, (0, 1)).sia().phi
    assert phi_sparse == pytest.approx(phi_dense, abs=1e-12)
