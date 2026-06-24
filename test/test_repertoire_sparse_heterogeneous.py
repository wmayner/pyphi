"""Regression: cause/effect repertoires on sparse + heterogeneous-alphabet
networks. These currently raise a shape error (the node's own dimension is
not collapsed for k>2 sparse nodes)."""

import numpy as np

from pyphi import Direction
from pyphi import Substrate
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
