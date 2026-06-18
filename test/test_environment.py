import itertools

import numpy as np
import pytest

import pyphi
from pyphi import utils
from pyphi.matching import MatchingAnalysis
from pyphi.matching import Perception
from pyphi.matching import PerceptualSystem
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


def _e1(n):
    return env.superpose(
        env.segment(n, 3, 0.6), env.segment(n, 2, 0.9), env.noise(n, 0.05)
    )


def _e2(n):
    return env.superpose(env.segment(n, 3, 0.6), env.point(n, 0.9), env.noise(n, 0.05))


def _e1b(n):
    return env.superpose(env.segment(n, 2, 0.9), env.noise(n, 0.05))


def test_paper_environments_normalized_with_full_support():
    n = 5
    for environment in (_e1(n), _e2(n), _e1b(n), env.noise(n, 0.5)):
        assert _sums_to_one(environment)
        assert all(len(state) == n for state in environment)


def test_e3_pure_noise_is_uniform():
    n = 5
    e3 = env.noise(n, 0.5)
    assert all(v == pytest.approx(1 / 2**n) for v in e3.values())


def test_e1_all_off_probability_hand_computed():
    # All-off occurs iff no segment fires AND the noise background is all-off.
    n = 5
    e1 = _e1(n)
    expected = (1 - 0.6) * (1 - 0.9) * (0.95**n)
    assert e1[(0, 0, 0, 0, 0)] == pytest.approx(expected)


def test_top_level_exports():
    assert pyphi.matching.segment is env.segment
    assert pyphi.matching.superpose is env.superpose
    assert pyphi.matching.sample is env.sample


def test_matching_analysis_runs_on_generated_world_distribution():
    substrate = pyphi.examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    perceptions = {}
    for stimulus in [(0,), (1,)]:
        y = ttpm.argmax_state(stimulus)
        full = [0, 0, 0]
        full[sensory[0]] = stimulus[0]
        for j, idx in enumerate(system):
            full[idx] = y[j]
        ces = substrate.ces(state=tuple(full), indices=system)
        perceptions[stimulus] = Perception(
            ces=ces, triggered_tpm=ttpm, stimulus=stimulus
        )
    world = env.noise(1, 0.3)  # {(0,): 0.7, (1,): 0.3}
    analysis = MatchingAnalysis(perceptions=perceptions, world_distribution=world)
    result = analysis.matching(seed=0, n_trials=5, k=3)
    assert result is not None
