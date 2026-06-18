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
