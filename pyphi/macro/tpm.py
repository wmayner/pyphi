"""The four-step macro TPM construction (Marshall et al. 2024, Eqs. 26-40).

Step 1 discounts micro connections extrinsic to the macro unit being
updated (Eqs. 26-30). Step 2 chains the modified probabilities into
micro-update sequences (Eq. 31). Step 3 causally marginalizes the
background, conditioning on the current micro state for effects (Eq. 33)
and Bayesian-weighting the pre-window background state for causes
(Eq. 34). Step 4 compresses sequences into macro states via the mapping
preimages and the sequence-proportion weights ``r(u^S, s)``
(Eqs. 35-40). Steps 2 and 4 are fused: sequence probabilities are
accumulated per chaining step into per-update state classes of the
unit's micro constituents, never materializing the full sequence tensor.
"""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.macro.units import MacroUnit


def _system_micro_indices(units) -> tuple[int, ...]:
    """``U^S``: the sorted union of the units' micro constituents."""
    out: set[int] = set()
    for unit in units:
        out |= set(unit.micro_constituents)
    return tuple(sorted(out))


def _patron_units(units) -> dict[int, int]:
    """Map each apportioned background index to its patron unit's index."""
    out: dict[int, int] = {}
    for k, unit in enumerate(units):
        for w in unit.background_apportionment:
            out[w] = k
    return out


def _discounted_on_probabilities(
    factored: FactoredTPM, units: tuple[MacroUnit, ...], j: int
) -> np.ndarray:
    """Step 1 (Eqs. 26-30): modified ON probabilities for updating unit ``j``.

    Returns:
        np.ndarray: ``(2**n, n)`` — for each universe state (little-endian
        row index) and micro unit, the modified probability that the unit
        is ON at the next micro update.
    """
    n = factored.n_nodes
    constituents = set(units[j].micro_constituents)
    patron = _patron_units(units)
    columns = []
    for i in range(n):
        p_on = factored.factor(i)[..., 1]
        if i in constituents:
            out = p_on  # Eq. 27: connections among U^J kept intact
        elif i in patron:
            k = patron[i]
            keep = set(units[k].micro_constituents) | set(
                units[k].background_apportionment
            )
            axes = tuple(a for a in range(n) if a not in keep)
            if axes:
                out = np.broadcast_to(p_on.mean(axis=axes, keepdims=True), p_on.shape)
            else:
                out = p_on  # Eq. 29 with nothing to noise
        else:
            # Eq. 28: other system units and unapportioned background
            out = np.full(p_on.shape, p_on.mean())
        columns.append(np.asarray(out).reshape(-1, order="F"))
    return np.stack(columns, axis=1)


def _full_transition_matrix(on_probabilities: np.ndarray) -> np.ndarray:
    """Row-stochastic ``(2**n, 2**n)`` matrix from ON probabilities (Eq. 30).

    Rows and columns are little-endian universe state indices.
    """
    num_states, n = on_probabilities.shape
    transition = np.ones((num_states, num_states))
    column_bits = np.arange(num_states)
    for i in range(n):
        bit = (column_bits >> i) & 1
        p = on_probabilities[:, i][:, np.newaxis]
        transition *= np.where(bit[np.newaxis, :] == 1, p, 1.0 - p)
    return transition


def _unit_sequence_distributions(transition: np.ndarray, unit: MacroUnit) -> np.ndarray:
    """Steps 2+4a fused (Eqs. 31, 35-36).

    Chains ``tau_J`` micro updates of the discounted transition matrix,
    accumulating probability into per-update state classes of ``U^J``.
    With the pinned digit convention, a sequence-class index is directly
    an index into ``unit.micro_mapping``.

    Returns:
        np.ndarray: ``(2**n, 2**(m * tau_J))`` — for each starting
        universe state, the probability of each ``U^J`` state-sequence.
    """
    num_states = transition.shape[0]
    m = len(unit.micro_constituents)
    num_classes = 2**m
    idx = np.arange(num_states)
    state_class = np.zeros(num_states, dtype=np.int64)
    for k, u in enumerate(unit.micro_constituents):
        state_class |= ((idx >> u) & 1) << k
    tau = unit.micro_grain
    dist = np.eye(num_states)[:, np.newaxis, :]  # (start, seq, current)
    place = 1
    for step in range(tau):
        seq_dim = dist.shape[1]
        advanced = np.einsum("xsu,uv->xsv", dist, transition)
        if step == tau - 1:
            out = np.zeros((num_states, seq_dim * num_classes))
            for a in range(num_classes):
                selected = state_class == a
                block = advanced[:, :, selected].sum(axis=2)
                out[:, a * place : a * place + seq_dim] += block
            return out
        out = np.zeros((num_states, seq_dim * num_classes, num_states))
        for a in range(num_classes):
            selected = state_class == a
            out[:, a * place : a * place + seq_dim, selected] = advanced[:, :, selected]
        dist = out
        place *= num_classes
    raise AssertionError("unreachable: tau >= 1 returns inside the loop")


def _unit_macro_probabilities(transition: np.ndarray, unit: MacroUnit) -> np.ndarray:
    """Eq. 35: probability of each macro state of ``J`` per starting state.

    Returns:
        np.ndarray: ``(2**n, 2)``.
    """
    sequence_dist = _unit_sequence_distributions(transition, unit)
    table = np.asarray(unit.micro_mapping)
    return np.stack(
        [
            sequence_dist[:, table == 0].sum(axis=1),
            sequence_dist[:, table == 1].sum(axis=1),
        ],
        axis=1,
    )


def _unit_final_state_proportions(unit: MacroUnit, j: int) -> np.ndarray:
    """Per-unit factor of ``r(u^S, s)`` (Eqs. 37-39).

    The proportion of ``g_J``-preimage sequences for macro state ``j``
    that end in each final ``U^J`` state. Counting is uniform over
    sequences (Eq. 38), not probability-weighted.
    """
    m = len(unit.micro_constituents)
    tau = unit.micro_grain
    table = np.asarray(unit.micro_mapping)
    idx = np.arange(len(table))
    final_state = idx >> (m * (tau - 1))
    counts = np.array(
        [np.sum((table == j) & (final_state == f)) for f in range(2**m)],
        dtype=np.float64,
    )
    return counts / counts.sum()


def _state_weights(units, system_indices, macro_state) -> np.ndarray:
    """``r(u^S, s)`` over system micro states (Eqs. 37-39).

    Factorizes as the product of per-unit final-state proportions
    because the ``U^J`` are disjoint (Eq. 18) and exactly cover ``U^S``
    (Eq. 23).
    """
    num_system_states = 2 ** len(system_indices)
    position = {u: k for k, u in enumerate(system_indices)}
    idx = np.arange(num_system_states)
    weights = np.ones(num_system_states)
    for unit, j in zip(units, macro_state, strict=True):
        local = np.zeros(num_system_states, dtype=np.int64)
        for b, u in enumerate(unit.micro_constituents):
            local |= ((idx >> position[u]) & 1) << b
        weights *= _unit_final_state_proportions(unit, j)[local]
    return weights
