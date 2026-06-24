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

from pyphi import exceptions
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import _mixed_radix_digits


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
    factored: FactoredTPM,
    units: tuple[MacroUnit, ...],
    j: int,
    patron: dict[int, int] | None = None,
) -> np.ndarray:
    """Step 1 (Eqs. 26-30): modified ON probabilities for updating unit ``j``.

    ``patron`` is the ``_patron_units(units)`` map; it depends only on
    ``units`` (not ``j``), so a caller updating every unit may build it once
    and pass it in. If ``None``, it is computed here.

    Returns:
        np.ndarray: ``(2**n, n)`` — for each universe state (little-endian
        row index) and micro unit, the modified probability that the unit
        is ON at the next micro update.
    """
    n = factored.n_nodes
    constituents = set(units[j].micro_constituents)
    if patron is None:
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


def _background_weights_cause(
    factored: FactoredTPM, system_indices, earliest
) -> np.ndarray:
    """``q_c`` (Eq. 34): Bayes posterior over the pre-window background.

    The posterior over the background state one micro update before the
    earliest state of the current window, given that earliest universe
    state, with a uniform prior over the full prior state. Computed from
    the ORIGINAL (undiscounted) TPM.

    Returns:
        np.ndarray: ``(2**|W|,)`` over little-endian background states.
    """
    n = factored.n_nodes
    likelihood = np.ones((2,) * n)
    for i in range(n):
        likelihood = likelihood * factored.factor(i)[..., earliest[i]]
    total = likelihood.sum()
    if total <= 0.0:
        raise exceptions.StateUnreachableBackwardsError(tuple(earliest))
    system_axes = tuple(sorted(system_indices))
    if system_axes:
        posterior = likelihood.sum(axis=system_axes)
    else:
        posterior = likelihood
    return (posterior / total).reshape(-1, order="F")


def _background_weights_effect(background_indices, current_state) -> np.ndarray:
    """``q_e`` (Eq. 33): delta on the current background micro state."""
    weights = np.zeros(2 ** len(background_indices))
    index = 0
    for k, i in enumerate(background_indices):
        index |= current_state[i] << k
    weights[index] = 1.0
    return weights


def _initial_distribution_indices(n: int, system_indices):
    """Precompute the (loop-invariant) index vectors for the initial
    universe-state distributions.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(system_part,
        background_part, idx)`` mapping each universe state to its system
        and background sub-state indices.
    """
    background_indices = tuple(i for i in range(n) if i not in set(system_indices))
    idx = np.arange(2**n)
    system_part = np.zeros(2**n, dtype=np.int64)
    for k, i in enumerate(system_indices):
        system_part |= ((idx >> i) & 1) << k
    background_part = np.zeros(2**n, dtype=np.int64)
    for k, i in enumerate(background_indices):
        background_part |= ((idx >> i) & 1) << k
    return system_part, background_part, idx


def _initial_distributions(
    num_system_states: int,
    system_part: np.ndarray,
    background_part: np.ndarray,
    idx: np.ndarray,
    background_weights: np.ndarray,
) -> np.ndarray:
    """Initial universe-state distribution per system micro state.

    The index vectors (``system_part``, ``background_part``, ``idx``)
    depend only on ``(n, system_indices)`` and are precomputed once by
    :func:`_initial_distribution_indices`; only ``background_weights``
    varies per direction.

    Returns:
        np.ndarray: ``(2**|U^S|, 2**n)`` — row ``u^S`` is the
        distribution with the system part pinned to ``u^S`` and the
        background part distributed per ``background_weights``.
    """
    init = np.zeros((num_system_states, len(idx)))
    init[system_part, idx] = background_weights[background_part]
    return init


def macro_tpms(substrate, units, micro_history):
    """The macro cause and effect TPMs ``(T_c, T_e)`` (Eqs. 26-42).

    Args:
        substrate: A binary :class:`~pyphi.substrate.Substrate` for the
            micro universe.
        units: The system's macro units. Their ``U^J union W^J`` must be
            pairwise disjoint (Eq. 18).
        micro_history: Universe micro states, oldest first, of length
            ``max(tau_J)``; the last entry is the current state.

    Returns:
        tuple[FactoredTPM, FactoredTPM]: ``(T_c, T_e)`` with one factor
        per macro unit over the macro system's states.
    """
    factored = substrate.factored_tpm
    n = factored.n_nodes
    units = tuple(units)
    micro_history = tuple(tuple(s) for s in micro_history)
    system_indices = _system_micro_indices(units)
    background_indices = tuple(i for i in range(n) if i not in set(system_indices))
    current_state = micro_history[-1]
    num_macro = len(units)
    macro_shape = (2,) * num_macro
    num_system_states = 2 ** len(system_indices)
    factors_cause = []
    factors_effect = []
    effect_weights = _background_weights_effect(background_indices, current_state)

    # These depend only on (units, system_indices, n) — not on the unit ``j``
    # being updated or the cause/effect direction — so build them once.
    patron = _patron_units(units)
    system_part, background_part, idx = _initial_distribution_indices(n, system_indices)
    weight_table = [
        _state_weights(units, system_indices, _mixed_radix_digits(s, macro_shape))
        for s in range(2**num_macro)
    ]

    for j, unit in enumerate(units):
        on_probabilities = _discounted_on_probabilities(factored, units, j, patron)
        transition = _full_transition_matrix(on_probabilities)
        macro_prob_full = _unit_macro_probabilities(transition, unit)
        earliest = micro_history[len(micro_history) - unit.micro_grain]
        cause_weights = _background_weights_cause(factored, system_indices, earliest)
        unit_factors = []
        for background_weights in (cause_weights, effect_weights):
            init = _initial_distributions(
                num_system_states, system_part, background_part, idx, background_weights
            )
            prob_given_system_state = init @ macro_prob_full  # (2**|S|, 2)
            factor = np.zeros((*macro_shape, 2))
            for s_index in range(2**num_macro):
                macro_state = _mixed_radix_digits(s_index, macro_shape)
                factor[macro_state] = weight_table[s_index] @ prob_given_system_state
            unit_factors.append(factor)
        factors_cause.append(unit_factors[0])
        factors_effect.append(unit_factors[1])
    return (
        FactoredTPM(factors=factors_cause),
        FactoredTPM(factors=factors_effect),
    )
