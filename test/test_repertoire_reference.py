"""pyphi cause/effect repertoires vs an independent reference."""

import itertools
import zlib

import numpy as np
import pytest

from pyphi import Direction
from pyphi import Substrate
from pyphi.distribution import repertoire_shape
from pyphi.system import System
from pyphi.utils import state_of
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


def _make_substrate(seed, alph, connectivity):
    rng = np.random.default_rng(seed)
    n = len(alph)
    if connectivity == "dense":
        cm = np.ones((n, n), dtype=int)
    elif connectivity == "chain":
        cm = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            cm[i, i + 1] = 1
    elif connectivity == "cycle":
        cm = np.zeros((n, n), dtype=int)
        for i in range(n):
            cm[i, (i + 1) % n] = 1
    else:
        raise ValueError(connectivity)
    factors = []
    for i in range(n):
        f = rng.uniform(size=(*alph, alph[i]))
        f /= f.sum(axis=-1, keepdims=True)
        factors.append(f)
    state_space = tuple(tuple(range(k)) for k in alph)
    return factors, cm, state_space


def _cut_cms(base_cm):
    edges = [
        (a, b)
        for a in range(base_cm.shape[0])
        for b in range(base_cm.shape[1])
        if base_cm[a, b] == 1
    ]
    cms = [base_cm.copy()]
    for e in edges:
        c = base_cm.copy()
        c[e] = 0
        cms.append(c)
    if len(edges) >= 2:
        c = base_cm.copy()
        c[edges[0]] = 0
        c[edges[1]] = 0
        cms.append(c)
    return cms


_SWEEP_CASES = [
    (2, (2, 2)),
    (2, (3, 3)),
    (2, (2, 3)),
    (3, (2, 2, 2)),
    (3, (3, 3, 4)),
    (3, (2, 3, 4)),
]
_CONNECTIVITY = ["dense", "chain", "cycle"]


@pytest.mark.parametrize("n,alph", _SWEEP_CASES)
@pytest.mark.parametrize("connectivity", _CONNECTIVITY)
def test_repertoires_match_reference_sweep(n, alph, connectivity):
    # Deterministic seed (zlib.crc32 is stable across runs, unlike hash()).
    seed = zlib.crc32(repr((n, alph, connectivity)).encode())
    factors, base_cm, state_space = _make_substrate(seed, alph, connectivity)
    state = tuple(0 for _ in range(n))
    mstate = dict(enumerate(state_of(range(n), state)))
    subsets = [s for k in range(1, n + 1) for s in itertools.combinations(range(n), k)]
    for cut_cm in _cut_cms(base_cm):
        sub = Substrate(
            marginals=[f.copy() for f in factors],
            state_space=state_space,
            cm=cut_cm,
        )
        sys = System(substrate=sub, state=state, node_indices=tuple(range(n)))
        for direction, reffn in (
            (Direction.CAUSE, ref_cause),
            (Direction.EFFECT, ref_effect),
        ):
            for mech in subsets:
                for purv in subsets:
                    got = np.asarray(sys.repertoire(direction, mech, purv))
                    expected = reffn(factors, alph, cut_cm, mech, mstate, purv, n)
                    assert got.shape == expected.shape
                    assert np.allclose(got, expected, atol=1e-12)
                    assert np.isclose(got.sum(), 1.0, atol=1e-12)
                    assert np.all(got >= -1e-12)
                    assert tuple(got.shape) == tuple(
                        repertoire_shape(range(n), purv, alphabet_sizes=alph)
                    )
