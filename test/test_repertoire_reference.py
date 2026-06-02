"""pyphi cause/effect repertoires vs an independent reference."""

import numpy as np

from test.reference.repertoire import ref_cause
from test.reference.repertoire import ref_effect


def _swap_factors():
    # 2 binary nodes, SWAP dynamics: node0' = prev1, node1' = prev0.
    # cm: 1->0 and 0->1. Factor i shape (prev0, prev1, out_i).
    f0 = np.zeros((2, 2, 2))
    f1 = np.zeros((2, 2, 2))
    for p0 in range(2):
        for p1 in range(2):
            f0[p0, p1, p1] = 1.0  # node0' copies prev1
            f1[p0, p1, p0] = 1.0  # node1' copies prev0
    cm = np.array([[0, 1], [1, 0]])
    return [f0, f1], (2, 2), cm


def test_reference_matches_hand_computed_swap():
    factors, alph, cm = _swap_factors()
    # Effect of {node0 = 1} on purview {node1}: node1' = prev0 = 1 -> [0, 1].
    eff = ref_effect(factors, alph, cm, (0,), {0: 1, 1: 0}, (1,), 2)
    assert np.allclose(eff.reshape(-1), [0.0, 1.0])
    # Cause of {node1 = 1} over purview {node0}: node1 = prev0 -> prev0 = 1 -> [0, 1].
    cau = ref_cause(factors, alph, cm, (1,), {0: 0, 1: 1}, (0,), 2)
    assert np.allclose(cau.reshape(-1), [0.0, 1.0])
