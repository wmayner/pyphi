# dynamics.py
"""Functions for simulating system state trajectories."""

from collections.abc import Iterable
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from . import utils
from .exceptions import NonConvergenceError


def mean_dynamics(
    tpm: ArrayLike,
    repetitions: int = 100,
    **kwargs,
):
    """Return a sample of the dynamics averaged over all initial states."""
    tpm = np.asarray(tpm, dtype=float)
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
    if rng is None:
        rng = np.random.default_rng(seed=None)

    if isinstance(tpm, pd.DataFrame):
        N = len(tpm.index[0])  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        step = _state_by_state_stepper(tpm, rng)
    else:
        # Assumes state-by-node multidimensional TPM.
        step = _state_by_node_stepper(np.asarray(tpm, dtype=float), rng)
        N = number_of_units(np.asarray(tpm))

    if clamp is None:
        clamp = {}

    if initial_state is None:
        initial_state = tuple(rng.integers(low=0, high=2, size=N))
    elif len(initial_state) != N:
        raise ValueError("initial_state must have length equal to the number of units")

    if isinstance(clamp, Mapping):
        clamps = [clamp] * timesteps  # pyright: ignore[reportOperatorIssue]
    else:
        clamps = clamp

    states = [apply_clamp(clamps[0], initial_state)]  # pyright: ignore[reportIndexIssue]
    for current_clamp in clamps[1:]:  # pyright: ignore[reportIndexIssue]
        states.append(apply_clamp(current_clamp, step(states[-1])))
    return states


def _state_by_state_stepper(tpm, rng):
    """Build a one-step sampler for a state-by-state DataFrame.

    Precomputes the per-row cumulative distribution once, so each step samples
    the next state in ``O(log K)`` via inverse-CDF (``numpy.searchsorted``)
    rather than re-normalizing and drawing per call. Samples the full joint
    next state, so correlations between units at ``t+1`` are preserved.
    """
    cumulative = tpm.to_numpy().cumsum(axis=1)
    labels = list(tpm.columns)
    row_of = {label: i for i, label in enumerate(tpm.index)}
    last = len(labels) - 1

    def step(state):
        row = cumulative[row_of[state]]
        j = int(np.searchsorted(row, rng.random(), side="right"))
        return labels[min(j, last)]

    return step


def _state_by_node_stepper(tpm, rng):
    """Build a one-step sampler for a multidimensional state-by-node TPM.

    Samples each unit independently from its own ``P(on | state)`` — exact for
    the conditionally-independent TPMs IIT uses, and ``O(N)`` vectorized numpy
    per step.
    """

    def step(state):
        probabilities = tpm[state]
        return tuple((probabilities > rng.random(len(probabilities))).astype(int))

    return step


def most_probable_next_state(tpm, state):
    """Return the deterministic most-probable next state (binary).

    Deterministic counterpart of the sampled step: each unit takes its
    most-probable next value (ON iff P(ON) > 0.5).
    """
    tpm = np.asarray(tpm, dtype=float)
    elementwise_probabilities = tpm[state]
    return tuple((elementwise_probabilities > 0.5).astype(int))


def settle(tpm, initial_state, *, clamp=None, max_steps=None):
    """Iterate the most-probable-transition map to a fixed point.

    Deterministic complement to :func:`simulate`: each step takes the
    most-probable next state (each unit ON iff its ON-probability exceeds 0.5)
    instead of sampling.

    Parameters
    ----------
    tpm : np.ndarray
        A state-by-node multidimensional TPM (binary).
    initial_state : tuple[int, ...]
        The starting state.
    clamp : Mapping[int, int] or None, optional
        Units held fixed to a given value every step.
    max_steps : int or None, optional
        Optional cap on the number of steps; raises if exceeded.

    Returns
    -------
    list[tuple[int, ...]]
        The trajectory of states ending at the fixed point. The fixed point is
        the last element and the settling time is ``len(result) - 1``.

    Raises
    ------
    ~pyphi.exceptions.NonConvergenceError
        If the map enters a limit cycle, or does not settle within
        ``max_steps``.
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
        # The just-appended state may itself be the fixed point (confirmed on
        # the next iteration), so the best-case settling time here is
        # len(trajectory) - 1; raise only when even that exceeds the cap.
        if max_steps is not None and len(trajectory) - 1 > max_steps:
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
