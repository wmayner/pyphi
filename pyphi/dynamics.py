# dynamics.py
"""Functions for simulating system state trajectories."""

from collections.abc import Iterable
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from . import utils
from .core.tpm.joint_distribution import JointTPM
from .exceptions import NonConvergenceError


def mean_dynamics(
    tpm: ArrayLike,
    repetitions: int = 100,
    **kwargs,
):
    """Return a sample of the dynamics averaged over all initial states."""
    tpm = JointTPM(tpm)
    clamp = kwargs.get("clamp", {})
    initial_states = [
        insert_clamp(clamp, state)
        for state in utils.all_states(number_of_units(tpm) - len(clamp))
    ]
    data = np.array(
        [
            [
                simulate(tpm, initial_state=initial_state, **kwargs)  # pyright: ignore[reportArgumentType]
                for initial_state in initial_states
            ]
            for _ in range(repetitions)
        ]
    )
    return data.mean(axis=(0, 1))


def simulate(
    tpm: ArrayLike,
    initial_state: tuple[int, ...] | None = None,
    timesteps: int | None = 100,
    clamp: Iterable[Mapping] | Mapping | None = None,
    rng: np.random.Generator | None = None,
):
    """Return a simulated timeseries of system states."""
    if isinstance(tpm, pd.DataFrame):
        N = len(tpm.index[0])  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        simulate_one_timestep = simulate_one_timestep_from_pandas_state_by_state
    else:
        # Assumes state-by-node multidimensional TPM
        tpm = JointTPM(tpm)
        N = tpm.number_of_units
        simulate_one_timestep = simulate_one_timestep_from_explicit_tpm_state_by_node

    if rng is None:
        rng = np.random.default_rng(seed=None)

    if clamp is None:
        clamp = {}

    if initial_state is None:
        initial_state = tuple(rng.integers(low=0, high=2, size=tpm.number_of_units))  # pyright: ignore[reportAttributeAccessIssue]
    elif len(initial_state) != N:
        raise ValueError("initial_state must have length equal to the number of units")

    if isinstance(clamp, Mapping):
        clamps = [clamp] * timesteps  # pyright: ignore[reportOperatorIssue]
    else:
        clamps = clamp

    states = [apply_clamp(clamps[0], initial_state)]  # pyright: ignore[reportIndexIssue]
    for current_clamp in clamps[1:]:  # pyright: ignore[reportIndexIssue]
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
    """Simulate one timestep given an JointTPM in multidimensional
    state-by-node form."""
    # Assumes state-by-node multidimensional TPM
    elementwise_probabilities = tpm[state]
    thresholds = rng.random(len(elementwise_probabilities))
    return tuple((elementwise_probabilities > thresholds).astype(int))


def most_probable_next_state(tpm, state):
    """Return the deterministic most-probable next state (binary).

    Counterpart of the sampled `simulate_one_timestep_*`: each unit takes its
    most-probable next value (ON iff P(ON) > 0.5).
    """
    tpm = JointTPM(tpm)
    elementwise_probabilities = np.asarray(tpm[state])
    return tuple((elementwise_probabilities > 0.5).astype(int))


def settle(tpm, initial_state, *, clamp=None, max_steps=None):
    """Iterate the most-probable-transition map to a fixed point.

    Deterministic complement to `simulate`: each step takes the most-probable
    next state instead of sampling. Returns the trajectory (a list of states)
    ending at the fixed point; the fixed point is the last element and the
    settling time is ``len(result) - 1``. Raises
    :class:`~pyphi.exceptions.NonConvergenceError` on a limit cycle.

    Args:
        tpm: A state-by-node multidimensional TPM (binary).
        initial_state (tuple[int, ...]): The starting state.

    Keyword Args:
        clamp (Mapping[int, int] | None): Units held fixed every step.
        max_steps (int | None): Optional cap; raises if exceeded.
    """
    if clamp is None:
        clamp = {}
    state = apply_clamp(clamp, tuple(initial_state))
    trajectory = [state]
    seen = {state}
    while True:
        nxt = apply_clamp(clamp, most_probable_next_state(tpm, state))
        if nxt == state:
            return trajectory
        if nxt in seen:
            raise NonConvergenceError(
                f"no fixed point; entered a limit cycle at {nxt} "
                f"(trajectory: {[*trajectory, nxt]})"
            )
        trajectory.append(nxt)
        seen.add(nxt)
        state = nxt
        if max_steps is not None and len(trajectory) > max_steps:
            raise NonConvergenceError(f"did not settle within max_steps={max_steps}")


# TODO: move to tpm module
def number_of_units(tpm: ArrayLike):
    return tpm.shape[-1]  # pyright: ignore[reportAttributeAccessIssue]


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
