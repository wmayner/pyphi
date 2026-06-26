"""Tests for PerceptualSystem (the environment->system layer)."""

import pytest

from pyphi import examples
from pyphi import utils
from pyphi.matching import PerceptualSystem


@pytest.fixture(scope="module")
def perceptual_system():
    return PerceptualSystem(
        examples.basic_substrate(), system_indices=(1, 2), sensory_indices=(0,)
    )


def test_environment_indices(perceptual_system):
    assert perceptual_system.environment_indices == (0,)


def test_rejects_overlapping_partition():
    with pytest.raises(ValueError, match="disjoint"):
        PerceptualSystem(
            examples.basic_substrate(), system_indices=(0, 1), sensory_indices=(1,)
        )


def test_rejects_sensory_in_system():
    with pytest.raises(ValueError):
        PerceptualSystem(
            examples.basic_substrate(), system_indices=(0, 1, 2), sensory_indices=(0,)
        )


def test_triggered_tpm_delegates(perceptual_system):
    ttpm = perceptual_system.triggered_tpm(tau=2, tau_clamp=1)
    assert ttpm.array.shape == (2, 2, 2)


def test_triggered_states_mapping(perceptual_system):
    states = perceptual_system.triggered_states(tau=2, tau_clamp=1)
    assert set(states) == set(utils.all_states(1))
    for response in states.values():
        assert len(response) == 2


def test_triggered_state_single(perceptual_system):
    response = perceptual_system.triggered_state((1,), tau=2, tau_clamp=1)
    assert response == perceptual_system.triggered_states(tau=2, tau_clamp=1)[(1,)]


def test_invalid_tau_raises(perceptual_system):
    with pytest.raises(ValueError):
        perceptual_system.triggered_tpm(tau=1, tau_clamp=2)
