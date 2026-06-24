"""Fast guard: actual causation on a small sparse + heterogeneous-alphabet
transition. Exercises the AC path through the k-ary repertoire fix."""

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import actual


def _sparse_het_ac_substrate():
    # node0 (k=3), node1 (k=3) -> node2 (k=4). node2 = (v0 + v1) mod 4.
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


def test_kary_ac_actual_cause_runs():
    sub = _sparse_het_ac_substrate()
    # voters (1,1) -> node2 = (1+1)%4 = 2
    t = actual.Transition(
        sub, (1, 1, 0), (1, 1, 2), cause_indices=(0, 1), effect_indices=(2,)
    )
    cause = t.find_actual_cause((2,))
    # The actual cause of node2 = (v0 + v1) % 4 = 2 is the joint voter state
    # (1, 1). Under a uniform prior over the 9 voter combinations, 3 are
    # consistent with node2 = 2, so alpha = log2(9 / 3) = log2(3) bits.
    assert cause.alpha == pytest.approx(np.log2(3), abs=1e-6)
    assert set(cause.purview) == {0, 1}
