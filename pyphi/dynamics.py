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
    seed: int | None = None,
    **kwargs,
):
    """Return a sample of the dynamics averaged over all initial states.

    A single generator seeded with ``seed`` is shared across all
    repetitions, so a given seed reproduces the whole sample. An explicit
    ``rng`` keyword argument takes precedence over ``seed``.
    """
    tpm = np.asarray(tpm, dtype=float)
    kwargs.setdefault("rng", np.random.default_rng(seed))
    clamp = kwargs.get("clamp", {})
    alphabets = _alphabet_sizes(tpm)
    free = tuple(a for i, a in enumerate(alphabets) if i not in clamp)
    initial_states = [insert_clamp(clamp, state) for state in utils.all_states(free)]
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
    seed: int | None = None,
):
    """Return a simulated timeseries of system states.

    Parameters
    ----------
    tpm : pandas.DataFrame or numpy.ndarray
        A state-by-state DataFrame (rows indexed by state tuples), a
        state-by-node multidimensional binary TPM, or an explicit-alphabet
        TPM of shape ``(*alphabet_sizes, n_units, max_alphabet)`` (as
        produced by :meth:`~pyphi.substrate.Substrate.joint_tpm`).
    initial_state : tuple[int, ...], optional
        The starting state. If None, each unit is drawn uniformly from its
        own alphabet.
    timesteps : int, optional
        Number of steps to simulate.
    clamp : Mapping or Iterable[Mapping], optional
        Units held fixed to a given value, either one mapping applied every
        step or one mapping per step.
    rng : numpy.random.Generator, optional
        Generator to draw from. Takes precedence over ``seed``.
    seed : int, optional
        Seed for the generator created when ``rng`` is None. If both are
        None, the trajectory is not reproducible.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if isinstance(tpm, pd.DataFrame):
        N = len(tpm.index[0])  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
        step = _state_by_state_stepper(tpm, rng)
        if initial_state is None:
            # Sample from the TPM's own state labels so every unit is drawn
            # over its full alphabet, not just {0, 1}.
            labels = tpm.index  # pyright: ignore[reportAttributeAccessIssue]
            initial_state = tuple(labels[int(rng.integers(len(labels)))])
    else:
        arr = np.asarray(tpm, dtype=float)
        layout = _explicit_alphabet_layout(arr)
        if layout is None:
            # Binary state-by-node multidimensional TPM.
            step = _state_by_node_stepper(arr, rng)
            N = number_of_units(arr)
            if initial_state is None:
                initial_state = tuple(rng.integers(low=0, high=2, size=N))
        else:
            N, alphabets = layout
            step = _explicit_alphabet_stepper(arr, alphabets, rng)
            if initial_state is None:
                # Draw each unit from its own alphabet.
                initial_state = tuple(int(rng.integers(a)) for a in alphabets)

    if clamp is None:
        clamp = {}

    if len(initial_state) != N:
        raise ValueError("initial_state must have length equal to the number of units")

    if isinstance(clamp, Mapping):
        if timesteps is None:
            raise ValueError(
                "timesteps=None requires an explicit iterable of per-step "
                "clamps; with a single clamp mapping, pass an integer "
                "number of timesteps"
            )
        clamps = [clamp] * timesteps
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


def _explicit_alphabet_layout(tpm):
    """Detect the explicit-alphabet TPM layout.

    Returns ``(n_units, alphabet_sizes)`` if ``tpm`` has the explicit-alphabet
    shape ``(*alphabet_sizes, n_units, max_alphabet)`` (as produced by
    :meth:`~pyphi.substrate.Substrate.joint_tpm`), and ``None`` otherwise.
    """
    shape = tpm.shape
    if tpm.ndim >= 3:
        n = int(shape[-2])
        if (
            tpm.ndim == n + 2
            and all(s >= 2 for s in shape[:n])
            and shape[-1] == max(shape[:n])
        ):
            return n, tuple(int(s) for s in shape[:n])
    return None


def _alphabet_sizes(tpm):
    """Per-unit alphabet sizes for either accepted ndarray TPM form."""
    layout = _explicit_alphabet_layout(tpm)
    if layout is not None:
        return layout[1]
    return (2,) * number_of_units(tpm)


def _explicit_alphabet_stepper(tpm, alphabet_sizes, rng):
    """Build a one-step sampler for an explicit-alphabet TPM.

    Samples each unit independently from its own next-state distribution
    (slots ``[:alphabet_sizes[i]]`` of the trailing axis, ignoring padding).
    """

    def step(state):
        dists = tpm[tuple(state)]  # shape (n_units, max_alphabet)
        nxt = []
        for i, a in enumerate(alphabet_sizes):
            cumulative = np.cumsum(dists[i, :a])
            j = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], "right"))
            nxt.append(min(j, a - 1))
        return tuple(nxt)

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
    """Return the deterministic most-probable next state.

    Deterministic counterpart of the sampled step: each unit takes its
    most-probable next value (for binary state-by-node TPMs, ON iff
    P(ON) > 0.5; for explicit-alphabet TPMs, the argmax over the unit's own
    alphabet).
    """
    tpm = np.asarray(tpm, dtype=float)
    layout = _explicit_alphabet_layout(tpm)
    if layout is not None:
        _, alphabets = layout
        dists = tpm[tuple(state)]  # shape (n_units, max_alphabet)
        return tuple(int(np.argmax(dists[i, :a])) for i, a in enumerate(alphabets))
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
        A state-by-node multidimensional TPM (binary), or an
        explicit-alphabet TPM of shape
        ``(*alphabet_sizes, n_units, max_alphabet)``.
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
    if len(initial_state) != number_of_units(tpm):
        raise ValueError("initial_state must have length equal to the number of units")
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
    """Number of units of a state-by-node or explicit-alphabet ndarray TPM."""
    tpm = np.asarray(tpm)
    layout = _explicit_alphabet_layout(tpm)
    if layout is not None:
        return layout[0]
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
