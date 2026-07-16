"""The triggered TPM: the system's fixed-lag response to each stimulus."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyphi import convert
from pyphi import utils
from pyphi.labels import NodeLabels


@dataclass(frozen=True)
class TriggeredTPM:
    """Pr(Sₜ = s | ∂S_{t−τ} = x), one distribution over system states per
    stimulus.

    Attributes
    ----------
    array : numpy.ndarray
        A multidimensional array with one binary axis per unit, ordered
        ``(sensory axes..., system axes...)``; ``array[x + s]`` is
        Pr(S = s | ∂S = x). Marginalizing a unit subset is a uniform axis sum.
    sensory_indices : tuple of int
        Substrate indices of the sensory-interface units, in axis order.
    system_indices : tuple of int
        Substrate indices of the system units, in axis order.
    node_labels : NodeLabels
        Labels for the substrate units, for the labeled ``to_pandas`` view.
    """

    array: np.ndarray
    sensory_indices: tuple[int, ...]
    system_indices: tuple[int, ...]
    node_labels: NodeLabels

    def row(self, stimulus: tuple[int, ...]) -> np.ndarray:
        """The system-state distribution for one stimulus."""
        return self.array[tuple(stimulus)]

    def argmax_state(self, stimulus: tuple[int, ...]) -> tuple[int, ...]:
        """The most-probable system state for a stimulus (the triggered state).

        Ties resolve to the first maximum in little-endian state order.
        """
        flat = int(np.argmax(self.row(stimulus).ravel(order="F")))
        return convert.le_index2state(flat, len(self.system_indices))

    def _marginalize_system(self, distribution, mechanism, state) -> float:
        """Return Pr(mechanism = state) from a distribution over the system axes.

        Sums out the system units not in ``mechanism``. Requires ``mechanism``
        to be a subset of ``system_indices`` (without duplicates) and ``state``
        to match its length; the (mechanism, state) pairs may be given in any
        order.
        """
        mechanism = tuple(mechanism)
        if len(set(mechanism)) != len(mechanism):
            raise ValueError(f"duplicate units in mechanism {mechanism}")
        if not set(mechanism) <= set(self.system_indices):
            raise ValueError(
                f"mechanism {mechanism} is not a subset of system_indices "
                f"{self.system_indices}"
            )
        if len(state) != len(mechanism):
            raise ValueError(f"state {state} length != mechanism {mechanism} length")
        # Canonicalize: sort the (mechanism, state) pairs together so the
        # axis bookkeeping below can assume increasing mechanism order.
        pairs = sorted(zip(mechanism, state, strict=True))
        mechanism = tuple(m for m, _ in pairs)
        state = tuple(s for _, s in pairs)
        keep = [self.system_indices.index(m) for m in mechanism]
        sum_axes = tuple(a for a in range(len(self.system_indices)) if a not in keep)
        reduced = distribution.sum(axis=sum_axes) if sum_axes else distribution
        # `mechanism` is sorted above and `system_indices` is validated sorted
        # at construction, so `keep` is increasing and the remaining axes are
        # already in mechanism order.
        return float(reduced[tuple(state)])

    def conditional_probability(self, mechanism, state, stimulus) -> float:
        """Pr(mechanism = state | ∂S = stimulus)."""
        return self._marginalize_system(self.row(stimulus), mechanism, state)

    def marginal_probability(self, mechanism, state) -> float:
        """Pr(mechanism = state), the uniform-prior marginal over stimuli."""
        marginal = self.array.mean(axis=tuple(range(len(self.sensory_indices))))
        return self._marginalize_system(marginal, mechanism, state)

    def to_pandas(self) -> pd.DataFrame:
        """Labeled view: rows = stimulus states, columns = system states,
        values = Pr(s | x)."""
        from pyphi.models.pandas import state_multiindex

        index = state_multiindex(self.node_labels, self.sensory_indices)
        columns = state_multiindex(self.node_labels, self.system_indices)
        data = [[self.array[tuple(x) + tuple(s)] for s in columns] for x in index]
        return pd.DataFrame(data, index=index, columns=columns)


def _full_state(sensory_indices, system_indices, x, s_sys, n):
    full = [0] * n
    for i, xi in zip(sensory_indices, x, strict=True):
        full[i] = xi
    for i, si in zip(system_indices, s_sys, strict=True):
        full[i] = si
    return tuple(full)


def _validate_binary_substrate(substrate) -> None:
    """Raise if the substrate has any non-binary unit.

    The triggered-TPM construction operates on the binary state-by-node
    representation; only binary substrates are currently supported.
    """
    sizes = substrate.factored_tpm.alphabet_sizes
    if any(size != 2 for size in sizes):
        raise ValueError(
            f"only binary substrates are currently supported; got alphabet sizes {sizes}"
        )


def _validate_sorted_indices(name: str, indices) -> None:
    """Raise unless ``indices`` is strictly increasing (sorted, no duplicates).

    Triggered-TPM axes and stimulus/state tuples are positional relative
    to these index tuples, so only the sorted form is unambiguous.
    """
    if not all(a < b for a, b in itertools.pairwise(indices)):
        raise ValueError(
            f"{name} must be strictly increasing (sorted, without "
            f"duplicates); got {tuple(indices)}"
        )


def _system_step_tpm(sbn_full, sensory_indices, system_indices, n, *, clamp_to):
    """A one-step state-by-node TPM over the system, with the sensory interface
    either clamped to a state (``clamp_to=x``) or marginalized
    (``clamp_to=None``)."""
    system = list(system_indices)
    shape_s = (2,) * len(system_indices)
    step = np.zeros((*shape_s, len(system_indices)))
    for s_sys in utils.all_states(len(system_indices)):
        if clamp_to is not None:
            full = _full_state(sensory_indices, system_indices, clamp_to, s_sys, n)
            step[s_sys] = sbn_full[full][system]
        else:
            acc = np.zeros(len(system_indices))
            for x in utils.all_states(len(sensory_indices)):
                full = _full_state(sensory_indices, system_indices, x, s_sys, n)
                acc += sbn_full[full][system]
            step[s_sys] = acc / (2 ** len(sensory_indices))
    return step


def _lagged_sbs(step_sbn, t):
    sbs = convert.state_by_node2state_by_state(step_sbn)
    if t == 0:
        return np.eye(sbs.shape[0])
    return np.linalg.matrix_power(sbs, t)


def build_triggered_tpm(
    substrate, sensory_indices, system_indices, *, tau, tau_clamp
) -> TriggeredTPM:
    """Construct the triggered TPM by clamp-then-noise evolution.

    Clamp the sensory interface to the stimulus for ``tau_clamp`` steps, then
    marginalize it for the remaining ``tau - tau_clamp`` steps; compose and
    average over the initial system state. Only binary substrates are
    currently supported.
    """
    _validate_binary_substrate(substrate)
    _validate_sorted_indices("sensory_indices", sensory_indices)
    _validate_sorted_indices("system_indices", system_indices)
    n = len(substrate.node_indices)
    sbn_full = np.asarray(substrate.tpm.to_array())[..., 1]  # binary ON-prob slice

    noised = _lagged_sbs(
        _system_step_tpm(sbn_full, sensory_indices, system_indices, n, clamp_to=None),
        tau - tau_clamp,
    )
    rows = []
    for x in utils.all_states(len(sensory_indices)):
        clamped = _lagged_sbs(
            _system_step_tpm(sbn_full, sensory_indices, system_indices, n, clamp_to=x),
            tau_clamp,
        )
        composed = clamped @ noised
        rows.append(composed.mean(axis=0))  # marginalize initial system state

    flat = np.array(rows)  # (n_stimuli, n_system_states), little-endian flat order
    n_sensory, n_system = len(sensory_indices), len(system_indices)
    array = flat.reshape((2,) * (n_sensory + n_system))
    # The flat orders are little-endian (first unit varies fastest) but the
    # C-order reshape unpacks last-axis-fastest, leaving each axis group in
    # reversed unit order; transpose each group back to unit order.
    sensory_axes = tuple(reversed(range(n_sensory)))
    system_axes = tuple(n_sensory + a for a in reversed(range(n_system)))
    return TriggeredTPM(
        array=array.transpose(sensory_axes + system_axes),
        sensory_indices=tuple(sensory_indices),
        system_indices=tuple(system_indices),
        node_labels=substrate.node_labels,
    )
