# dynamics.py

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .data_structures import FrozenMap
from .tpm import ExplicitTPM
from . import utils


def mean_dynamics(
    tpm: ArrayLike,
    repetitions: int = 100,
    **kwargs,
):
    """Return a sample of the dynamics averaged over all initial states."""
    tpm = ExplicitTPM(tpm)
    clamp = kwargs.get("clamp", FrozenMap())
    initial_states = [
        insert_clamp(clamp, state)
        for state in utils.all_states(number_of_units(tpm) - len(clamp))
    ]
    data = np.array(
        [
            [
                simulate(tpm, initial_state=initial_state, **kwargs)
                for initial_state in initial_states
            ]
            for _ in range(repetitions)
        ]
    )
    return data.mean(axis=(0, 1))


def simulate(
    tpm: ArrayLike,
    initial_state: tuple[int] = None,
    timesteps: int = 100,
    clamp: FrozenMap = None,
    rng: np.random.Generator = None,
):
    """Return a simulated timeseries of system states."""
    tpm = ExplicitTPM(tpm)
    if rng is None:
        rng = np.random.default_rng(seed=None)
    if initial_state is None:
        initial_state = tuple(rng.integers(low=0, high=2, size=number_of_units(tpm)))
    elif len(initial_state) != number_of_units(tpm):
        raise ValueError("initial_state must have length equal to the number of units")

    states = [apply_clamp(clamp, initial_state)]
    for _ in range(timesteps):
        # Assumes state-by-node multidimensional TPM
        elementwise_probabilities = tpm[states[-1]]
        next_state = simulate_one_timestep(elementwise_probabilities, rng)
        next_state = apply_clamp(clamp, next_state)
        states.append(next_state)
    return states


def simulate_one_timestep(
    elementwise_probabilities: Iterable[float], rng: np.random.Generator
):
    thresholds = rng.random(len(elementwise_probabilities))
    return tuple(
        1 if probability > threshold else 0
        for probability, threshold in zip(elementwise_probabilities, thresholds)
    )


# TODO(4.0): move to tpm module
def number_of_units(tpm: ArrayLike):
    return tpm.shape[-1]


def apply_clamp(clamp, state):
    if not clamp:
        return state
    state = list(state)
    for index, unit_state in clamp.items():
        state[index] = unit_state
    return tuple(state)


def insert_clamp(clamp, state):
    if not clamp:
        return state
    state = list(state)
    for index, unit_state in sorted(clamp.items()):
        state.insert(index, unit_state)
    return tuple(state)
