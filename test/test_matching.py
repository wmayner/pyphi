"""Tests for MatchingAnalysis / MatchingResult (matching M)."""

import numpy as np
import pytest

from pyphi import examples
from pyphi.matching import MatchingAnalysis
from pyphi.matching import PerceptualSystem
from pyphi.matching.perception import Perception


def _full_state(sensory_indices, system_indices, x, y):
    n = len(sensory_indices) + len(system_indices)
    full = [0] * n
    for i, xi in zip(sensory_indices, x, strict=True):
        full[i] = xi
    for i, yi in zip(system_indices, y, strict=True):
        full[i] = yi
    return tuple(full)


@pytest.fixture(scope="module")
def perceptions():
    substrate = examples.grid3_substrate()
    sensory, system = (0,), (1, 2)
    ps = PerceptualSystem(substrate, system_indices=system, sensory_indices=sensory)
    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    result = {}
    for stimulus in [(0,), (1,)]:
        y = ttpm.argmax_state(stimulus)
        ces = substrate.ces(
            state=_full_state(sensory, system, stimulus, y), indices=system
        )
        result[stimulus] = Perception(ces=ces, triggered_tpm=ttpm, stimulus=stimulus)
    return result


@pytest.fixture(scope="module")
def analysis(perceptions):
    return MatchingAnalysis(
        perceptions=perceptions,
        world_distribution={(0,): 0.75, (1,): 0.25},
    )


# --- Validation --------------------------------------------------------------


def test_world_keys_must_have_perceptions(perceptions):
    with pytest.raises(ValueError, match="without a perceptual structure"):
        MatchingAnalysis(perceptions=perceptions, world_distribution={(7,): 1.0})


def test_world_probabilities_must_be_nonnegative(perceptions):
    with pytest.raises(ValueError, match="nonnegative"):
        MatchingAnalysis(
            perceptions=perceptions,
            world_distribution={(0,): 1.5, (1,): -0.5},
        )


def test_world_probabilities_must_sum_to_one(perceptions):
    with pytest.raises(ValueError, match="sum to 1"):
        MatchingAnalysis(
            perceptions=perceptions,
            world_distribution={(0,): 0.4, (1,): 0.4},
        )


def test_perceptions_must_be_nonempty():
    with pytest.raises(ValueError, match="at least one"):
        MatchingAnalysis(perceptions={}, world_distribution={})


def test_matching_parameter_validation(analysis):
    with pytest.raises(ValueError, match="n_trials"):
        analysis.matching(seed=0, n_trials=0, k=1)
    with pytest.raises(ValueError, match="k"):
        analysis.matching(seed=0, n_trials=1, k=0)


def test_noise_distribution_is_uniform(analysis):
    assert analysis.noise_distribution == {(0,): 0.5, (1,): 0.5}


# --- Sampling ----------------------------------------------------------------


def test_fixed_seed_is_reproducible(analysis):
    a = analysis.matching(seed=42, n_trials=5, k=3)
    b = analysis.matching(seed=42, n_trials=5, k=3)
    assert a == b


def test_uniform_world_matching_is_zero(perceptions):
    uniform = MatchingAnalysis(
        perceptions=perceptions,
        world_distribution={(0,): 0.5, (1,): 0.5},
    )
    for seed in [0, 1, 2026]:
        result = uniform.matching(seed=seed, n_trials=4, k=3)
        # World and noise distributions coincide, so the paired sampling
        # yields identical sequences and the gap is exactly zero.
        assert result.value == 0.0
        assert result.world_differentiation == result.noise_differentiation


def test_raw_arrays_reconstruct_value(analysis):
    result = analysis.matching(seed=7, n_trials=6, k=2)
    assert len(result.world_differentiation) == 6
    assert len(result.noise_differentiation) == 6
    reconstructed = float(
        np.mean(result.world_differentiation) - np.mean(result.noise_differentiation)
    )
    assert result.value == pytest.approx(reconstructed)
    assert result.seed == 7
    assert result.n_trials == 6
    assert result.k == 2
    assert result.subsequence is None


def test_subsequence_max_on_k1_equals_default(analysis):
    default = analysis.matching(seed=3, n_trials=4, k=1)
    fancy = analysis.matching(seed=3, n_trials=4, k=1, subsequence_max=True)
    assert fancy.value == pytest.approx(default.value)
    assert fancy.subsequence == (1, 1)


def test_subsequence_max_dominates_full_sequence(analysis):
    default = analysis.matching(seed=3, n_trials=4, k=3)
    fancy = analysis.matching(seed=3, n_trials=4, k=3, subsequence_max=True)
    assert fancy.value >= default.value - 1e-12
    assert fancy.subsequence is not None
    a, b = fancy.subsequence
    assert 1 <= a <= b <= 3
    # Raw arrays reconstruct the value in subsequence mode too.
    reconstructed = float(
        np.mean(fancy.world_differentiation) - np.mean(fancy.noise_differentiation)
    )
    assert fancy.value == pytest.approx(reconstructed)
