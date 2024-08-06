# dynamics.py
"""Functions for simulating system state trajectories."""

from typing import Iterable, Optional, Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from . import utils
from .tpm import ExplicitTPM


def mean_dynamics(
    tpm: ArrayLike,
    repetitions: int = 100,
    **kwargs,
):
    """Return a sample of the dynamics averaged over all initial states."""
    tpm = ExplicitTPM(tpm)
    clamp = kwargs.get("clamp", dict())
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
    initial_state: Optional[tuple[int]] = None,
    timesteps: Optional[int] = 100,
    clamp: Optional[Iterable[Mapping] | Mapping] = None,
    rng: np.random.Generator = None,
):
    """Return a simulated timeseries of system states."""
    if isinstance(tpm, pd.DataFrame):
        N = len(tpm.index[0])
        simulate_one_timestep = simulate_one_timestep_from_pandas_state_by_state
    else:
        # Assumes state-by-node multidimensional TPM
        tpm = ExplicitTPM(tpm)
        N = tpm.number_of_units
        simulate_one_timestep = simulate_one_timestep_from_explicit_tpm_state_by_node

    if rng is None:
        rng = np.random.default_rng(seed=None)

    if clamp is None:
        clamp = dict()

    if initial_state is None:
        initial_state = tuple(rng.integers(low=0, high=2, size=tpm.number_of_units))
    elif len(initial_state) != N:
        raise ValueError("initial_state must have length equal to the number of units")

    if isinstance(clamp, Mapping):
        clamps = [clamp] * timesteps
    else:
        clamps = clamp

    states = [apply_clamp(clamps[0], initial_state)]
    for current_clamp in clamps[1:]:
        current_state = states[-1]
        next_state = simulate_one_timestep(rng, tpm, current_state)
        next_state = apply_clamp(current_clamp, next_state)
        states.append(next_state)
    return states


def simulate_one_timestep_from_pandas_state_by_state(rng, tpm, state):
    """Simulate one timestep given a DataFrame containing probabilities indexed
    by state along both dimensions."""
    state_probabilities = tpm.loc[state]
    return state_probabilities.sample(weights=state_probabilities).index[0]


def simulate_one_timestep_from_explicit_tpm_state_by_node(rng, tpm, state):
    """Simulate one timestep given an ExplicitTPM in multidimensional
    state-by-node form."""
    # Assumes state-by-node multidimensional TPM
    elementwise_probabilities = tpm[state]
    thresholds = rng.random(len(elementwise_probabilities))
    return tuple((elementwise_probabilities > thresholds).astype(int))


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
