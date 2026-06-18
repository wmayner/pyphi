import itertools

import numpy as np
import pytest

from pyphi import utils
from pyphi.matching import environment as env


def _sums_to_one(dist):
    return utils.eq(sum(dist.values()), 1.0)


def test_segment_hand_computed():
    # n=4, length=2, p=0.6: 3 positions, each contiguous-2 run gets 0.6/3=0.2;
    # all-off gets 1-0.6=0.4.
    dist = env.segment(4, 2, 0.6)
    assert _sums_to_one(dist)
    assert dist[(0, 0, 0, 0)] == pytest.approx(0.4)
    assert dist[(1, 1, 0, 0)] == pytest.approx(0.2)
    assert dist[(0, 1, 1, 0)] == pytest.approx(0.2)
    assert dist[(0, 0, 1, 1)] == pytest.approx(0.2)
    # No non-contiguous or wrong-length states present.
    assert (1, 0, 1, 0) not in dist
    assert (1, 1, 1, 0) not in dist


def test_segment_full_length_is_all_on_or_off():
    dist = env.segment(3, 3, 0.7)
    assert dist[(1, 1, 1)] == pytest.approx(0.7)
    assert dist[(0, 0, 0)] == pytest.approx(0.3)
    assert _sums_to_one(dist)


def test_point_equals_segment_length_one():
    assert env.point(5, 0.4) == env.segment(5, 1, 0.4)


def test_noise_is_product_bernoulli():
    dist = env.noise(3, 0.25)
    for state in itertools.product((0, 1), repeat=3):
        expected = np.prod([0.25 if s else 0.75 for s in state])
        assert dist[state] == pytest.approx(expected)
    assert _sums_to_one(dist)


def test_noise_half_is_uniform():
    dist = env.noise(4, 0.5)
    assert len(dist) == 16
    for v in dist.values():
        assert v == pytest.approx(1 / 16)


def test_generator_argument_validation():
    with pytest.raises(ValueError):
        env.segment(3, 4, 0.5)  # length > n
    with pytest.raises(ValueError):
        env.segment(3, 0, 0.5)  # length < 1
    with pytest.raises(ValueError):
        env.segment(3, 1, 1.5)  # p out of range
    with pytest.raises(ValueError):
        env.noise(3, -0.1)  # p out of range


def test_superpose_or_combines_independently():
    # Deterministic point at index 0 OR deterministic point at index 1.
    a = {(1, 0): 1.0}
    b = {(0, 1): 1.0}
    assert env.superpose(a, b) == {(1, 1): 1.0}


def test_superpose_with_all_off_is_identity():
    a = env.segment(4, 2, 0.6)
    off = {(0, 0, 0, 0): 1.0}
    combined = env.superpose(a, off)
    assert set(combined) == set(a)
    for state in a:
        assert combined[state] == pytest.approx(a[state])


def test_superpose_hand_computed_probability():
    # noise(2, 0.5) OR a deterministic point at index 0.
    # Result state (1, x): index 0 always on; index 1 on iff noise set it.
    combined = env.superpose(env.noise(2, 0.5), {(1, 0): 1.0})
    assert combined[(1, 0)] == pytest.approx(0.5)  # noise gave (0,0) or (1,0)
    assert combined[(1, 1)] == pytest.approx(0.5)  # noise gave (0,1) or (1,1)
    assert _sums_to_one(combined)


def test_superpose_requires_matching_n():
    with pytest.raises(ValueError):
        env.superpose({(0, 0): 1.0}, {(0,): 1.0})


def test_mixture_weights():
    a = {(1, 0): 1.0}
    b = {(0, 1): 1.0}
    m = env.mixture([a, b], weights=[3, 1])
    assert m[(1, 0)] == pytest.approx(0.75)
    assert m[(0, 1)] == pytest.approx(0.25)
    assert _sums_to_one(m)


def test_mixture_uniform_default():
    a = {(1, 0): 1.0}
    b = {(0, 1): 1.0}
    m = env.mixture([a, b])
    assert m[(1, 0)] == pytest.approx(0.5)
    assert m[(0, 1)] == pytest.approx(0.5)


def test_sample_is_seed_deterministic_and_isolated():
    dist = env.noise(3, 0.3)
    a = env.sample(dist, 50, seed=7)
    b = env.sample(dist, 50, seed=7)
    assert a == b  # reproducible
    c = env.sample(dist, 50, seed=8)
    assert a != c  # seed changes output
    # No global-RNG dependence.
    np.random.seed(0)
    d = env.sample(dist, 50, seed=7)
    np.random.seed(123)
    e = env.sample(dist, 50, seed=7)
    assert d == e


def test_sample_only_draws_supported_states():
    dist = env.segment(4, 2, 0.6)
    drawn = set(env.sample(dist, 200, seed=1))
    assert drawn <= set(dist)


def test_sample_empirical_frequencies_converge():
    dist = env.noise(2, 0.5)
    draws = env.sample(dist, 20000, seed=42)
    counts = dict.fromkeys(dist, 0)
    for state in draws:
        counts[state] += 1
    for state, prob in dist.items():
        assert counts[state] / len(draws) == pytest.approx(prob, abs=0.02)
