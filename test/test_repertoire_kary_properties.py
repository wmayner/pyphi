"""Property tests: cause/effect repertoires match the independent reference,
are canonical-shaped, and normalized for arbitrary alphabet sizes and
connectivity."""

import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import Direction
from pyphi import Substrate
from pyphi.distribution import repertoire_shape
from pyphi.system import System
from pyphi.utils import state_of
from test.reference.repertoire import ref_cause
from test.reference.repertoire import ref_effect


def _random_substrate(seed, alphabets, dense):
    rng = np.random.default_rng(seed)
    n = len(alphabets)
    alph = tuple(alphabets)
    if dense:
        cm = np.ones((n, n), dtype=int)
    else:
        # chain: node i -> node i+1
        cm = np.zeros((n, n), dtype=int)
        for i in range(n - 1):
            cm[i, i + 1] = 1
    marginals = []
    for i in range(n):
        f = rng.uniform(size=(*alph, alph[i]))
        f /= f.sum(axis=-1, keepdims=True)
        marginals.append(f)
    state_space = tuple(tuple(range(k)) for k in alph)
    return marginals, cm, state_space


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(0, 2**31 - 1),
    alphabets=st.lists(st.integers(2, 4), min_size=2, max_size=3),
    dense=st.booleans(),
    direction=st.sampled_from([Direction.CAUSE, Direction.EFFECT]),
)
def test_repertoire_matches_reference(seed, alphabets, dense, direction):
    marginals, cm, state_space = _random_substrate(seed, alphabets, dense)
    n = len(alphabets)
    alph = tuple(alphabets)
    state = tuple(0 for _ in range(n))
    sub = Substrate(marginals=marginals, state_space=state_space, cm=cm)
    s = System(substrate=sub, state=state, node_indices=tuple(range(n)))
    mechanism, purview = (0,), (n - 1,)
    got = np.asarray(s.repertoire(direction, mechanism, purview))
    reffn = ref_cause if direction == Direction.CAUSE else ref_effect
    mstate = dict(enumerate(state_of(range(n), state)))
    expected = reffn(marginals, alph, cm, mechanism, mstate, purview, n)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-12)
    assert tuple(got.shape) == tuple(
        repertoire_shape(s.node_indices, purview, alphabet_sizes=alph)
    )
    assert np.isclose(got.sum(), 1.0)
