"""Tests for enforcement of the Realization principle (Albantakis et al. 2019)."""

import numpy as np
import pytest

from pyphi import exceptions
from pyphi import validate
from pyphi.substrate import Substrate


@pytest.fixture
def swap_substrate():
    # Each unit copies the other's previous state: unit 0 next = unit 1
    # previous, unit 1 next = unit 0 previous. Deterministic, so from
    # (0, 0) the only successor is (0, 0).
    tpm = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    cm = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    return Substrate(tpm, cm)


def test_transition_unreachable_error_attributes():
    err = exceptions.TransitionUnreachableError((0, 0), (1, 0))
    assert isinstance(err, exceptions.StateUnreachableError)
    assert err.before_state == (0, 0)
    assert err.after_state == (1, 0)
    assert err.state == (1, 0)


def test_transition_states_rejects_impossible_pair(swap_substrate):
    # Unit 0 cannot turn on from (0, 0).
    with pytest.raises(exceptions.TransitionUnreachableError):
        validate.transition_states(swap_substrate, (0, 0), (1, 0))


def test_transition_states_accepts_possible_pair(swap_substrate):
    validate.transition_states(swap_substrate, (0, 0), (0, 0))
    validate.transition_states(swap_substrate, (1, 0), (0, 1))


def test_transition_states_rejects_malformed_states(swap_substrate):
    with pytest.raises(ValueError):
        validate.transition_states(swap_substrate, (0, 0, 0), (0, 0))
    with pytest.raises(ValueError):
        validate.transition_states(swap_substrate, (0, 0), (0, 2))
