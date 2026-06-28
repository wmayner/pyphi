# substrate_generator/mechanisms.py
"""Library of unit mechanisms ported from Bjørn Juel's ``substrate_modeler``.

Each mechanism is a unit function with the signature
``f(element, weights, state, **params) -> float`` returning the probability that
``element`` is ON at the next step, given the substrate ``state`` at the current
step. This is the contract consumed by :func:`pyphi.substrate_generator.build_tpm`
(it is called once per from-state), so the assembled substrate TPM equals the
``dynamic_tpm`` of the original ``substrate_modeler`` (present state = past
state).

State-dependent mechanisms (the "endorsement" family — :func:`resonnator`,
:func:`mismatch_corrector`, :func:`stabilized_sigmoid`, :func:`biased_sigmoid`,
:func:`modulated_sigmoid`) read the unit's own current state, ``state[element]``.
They therefore require a self-loop (a nonzero ``weights[element, element]``) so
that the dependency is represented in the connectivity matrix;
:func:`pyphi.substrate_generator.create_substrate` inserts it automatically.

``resonnator`` realizes the state-dependent coupling of the matching paper
(Mayner, Juel & Tononi), whose default ``weight_scale_mapping`` is the coupling
factor ``g(I_j, s_k)`` of that paper's appendix.
"""

import numpy as np

from .utils import map_to_floor_and_ceil

# Default state-dependent coupling for ``resonnator``, keyed by
# ``(unit_state, input_state)`` — the matching paper's coupling factor g.
DEFAULT_WEIGHT_SCALE_MAPPING = {
    (0, 0): 1.0,
    (1, 0): 0.5,
    (0, 1): 0.75,
    (1, 1): 1.5,
}


def _ordered_inputs(element, weights, inputs):
    """Return the unit's input indices, preserving caller order when given.

    With no explicit ``inputs``, the inputs are the nonzero entries of the
    element's weight column, in ascending index order.
    """
    if inputs is not None:
        return tuple(inputs)
    return tuple(int(i) for i in np.nonzero(weights[:, element])[0])


def _logistic(total_input, determinism, threshold):
    return 1.0 / (1.0 + np.e ** (-determinism * (total_input - threshold)))


def sigmoid(
    element,
    weights,
    state,
    *,
    floor=0.0,
    ceiling=1.0,
    determinism=5.0,
    threshold=0.0,
    ising=True,
    inputs=None,
    **kwargs,
):
    """Logistic activation of the weighted input."""
    ins = _ordered_inputs(element, weights, inputs)
    s = np.array([state[i] for i in ins], dtype=float)
    if ising:
        s = s * 2 - 1
    w = np.array([weights[i, element] for i in ins], dtype=float)
    y = _logistic(float(np.sum(s * w)), determinism, threshold)
    return map_to_floor_and_ceil(y, floor, ceiling)


def resonnator(
    element,
    weights,
    state,
    *,
    determinism,
    threshold,
    weight_scale_mapping=None,
    floor=0.0,
    ceiling=1.0,
    inputs=None,
    **kwargs,
):
    """State-dependent endorsement: inputs that agree with the unit's own state
    are excitatory and amplified; disagreeing inputs are inhibitory.

    Requires a self-loop so ``state[element]`` is part of the connectivity.
    """
    if weight_scale_mapping is None:
        weight_scale_mapping = DEFAULT_WEIGHT_SCALE_MAPPING
    ins = _ordered_inputs(element, weights, inputs)
    unit_state = state[element]
    w = []
    for i in ins:
        input_state = state[i]
        scale = weight_scale_mapping[(unit_state, input_state)]
        base = weights[i, element]
        w.append(base * scale if input_state == unit_state else -base * scale)
    spin = np.array([2 * state[i] - 1 for i in ins], dtype=float)
    y = _logistic(float(np.sum(spin * np.array(w))), determinism, threshold)
    return map_to_floor_and_ceil(y, floor, ceiling)


def sor_gate(
    element,
    weights,
    state,
    *,
    pattern_selection=(),
    ceiling=1.0,
    floor=0.0,
    selectivity=2.0,
    inputs=None,
    **kwargs,
):
    """Selective-OR detector: ON with probability ``ceiling`` iff the input
    pattern is one of ``pattern_selection``, otherwise OFF.

    (``floor`` and ``selectivity`` parameterize the off-diagonal entries of the
    original per-unit TPM, which the present=past substrate TPM never reads; they
    are accepted for signature parity.)
    """
    patterns = [tuple(p) for p in pattern_selection]
    ins = _ordered_inputs(element, weights, inputs)
    present = tuple(state[i] for i in ins)
    return ceiling if present in patterns else 0.0


def mismatch_pattern_detector(
    element,
    weights,
    state,
    *,
    pattern_selection=(),
    ceiling=1.0,
    floor=0.0,
    selectivity=1.0,
    inputs=None,
    **kwargs,
):
    """Pattern detector that responds strongly to *mismatched* inputs: weakly
    coupled to inputs matching its current state, strongly to unexpected ones.

    Requires a self-loop (reads ``state[element]``).
    """
    if not selectivity > 1:
        selectivity = 1 / selectivity
    patterns = [tuple(p) for p in pattern_selection]
    ins = _ordered_inputs(element, weights, inputs)
    present = tuple(state[i] for i in ins)
    if state[element] == 1:
        p_pattern = 0.5 + (ceiling - 0.5) / selectivity
        p_no_pattern = floor
    else:
        p_pattern = ceiling
        p_no_pattern = 0.5 - (0.5 - floor) / selectivity
    return p_pattern if present in patterns else p_no_pattern


def gabor_gate(
    element,
    weights,
    state,
    *,
    preferred_states=(),
    ceiling=1.0,
    floor=0.0,
    inputs=None,
    **kwargs,
):
    """Gabor-like detector: ``ceiling`` on a preferred input pattern,
    ``floor`` on its complement (anti-pattern), and ``0.5`` otherwise.
    """
    preferred = [tuple(p) for p in preferred_states]
    anti = [tuple(int(1 - s) for s in p) for p in preferred]
    ins = _ordered_inputs(element, weights, inputs)
    present = tuple(state[i] for i in ins)
    if present in preferred:
        return ceiling
    if present in anti:
        return floor
    return 0.5


def copy_gate(element, weights, state, *, floor=0.0, ceiling=1.0, inputs=None, **kwargs):
    """Single-input copy: output follows the input."""
    ins = _ordered_inputs(element, weights, inputs)
    return ceiling if state[ins[0]] == 1 else floor


def and_gate(element, weights, state, *, floor=0.0, ceiling=1.0, inputs=None, **kwargs):
    """Two-input AND truth table (``ceiling`` only when both inputs are ON)."""
    ins = _ordered_inputs(element, weights, inputs)
    present = (state[ins[0]], state[ins[1]])
    return ceiling if present == (1, 1) else floor


def or_gate(element, weights, state, *, floor=0.0, ceiling=1.0, inputs=None, **kwargs):
    """Two-input OR truth table (``floor`` only when both inputs are OFF)."""
    ins = _ordered_inputs(element, weights, inputs)
    present = (state[ins[0]], state[ins[1]])
    return floor if present == (0, 0) else ceiling


def xor_gate(element, weights, state, *, floor=0.0, ceiling=1.0, inputs=None, **kwargs):
    """Two-input XOR truth table (``ceiling`` iff the inputs differ)."""
    ins = _ordered_inputs(element, weights, inputs)
    present = (state[ins[0]], state[ins[1]])
    return ceiling if present in {(0, 1), (1, 0)} else floor


def democracy(element, weights, state, *, floor=0.0, ceiling=1.0, inputs=None, **kwargs):
    """Activation equal to the mean of the (binary) input states."""
    ins = _ordered_inputs(element, weights, inputs)
    avg_vote = float(np.mean([state[i] for i in ins]))
    return avg_vote * (ceiling - floor) + floor


def majority(element, weights, state, *, floor=0.0, ceiling=1.0, inputs=None, **kwargs):
    """Activation equal to the rounded mean of the (binary) input states."""
    ins = _ordered_inputs(element, weights, inputs)
    avg_vote = round(float(np.mean([state[i] for i in ins])))
    return avg_vote * (ceiling - floor) + floor


def weighted_mean(
    element, weights, state, *, floor=0.0, ceiling=1.0, inputs=None, **kwargs
):
    """Spin-weighted mean of the inputs, using the (normalized) input weights."""
    ins = _ordered_inputs(element, weights, inputs)
    w = np.array([weights[i, element] for i in ins], dtype=float)
    total = np.sum(w)
    if total == 0:
        return floor
    w = w / total
    n = len(ins)
    wm = sum((1 + wi * (state[i] * 2 - 1)) / 2 for wi, i in zip(w, ins, strict=True)) / n
    return wm * (ceiling - floor) + floor


def mismatch_corrector(
    element, weights, state, *, floor=0.0, ceiling=1.0, bias=0.0, inputs=None, **kwargs
):
    """Single-input corrector: when the unit and its input match, it relaxes
    toward 0.5 (biased by ``bias``); when they mismatch, it copies the input.

    Requires a self-loop (reads ``state[element]``).
    """
    ins = _ordered_inputs(element, weights, inputs)
    unit_state = state[element]
    input_state = state[ins[0]]
    if unit_state == input_state:
        return 0.5 - (unit_state * 2 - 1) * bias * 0.5
    return ceiling if input_state == 1 else floor


def modulated_sigmoid(
    element,
    weights,
    state,
    *,
    input_weights,
    modulation,
    floor=0.0,
    ceiling=1.0,
    determinism=2.0,
    threshold=0.0,
    inputs=None,
    **kwargs,
):
    """Sigmoid whose threshold and determinism are shifted by the number of
    active modulator inputs, scaled by the unit's own (Ising) state.

    ``modulation`` is ``{'modulator': tuple(indices), 'threshold': float,
    'determinism': float}``. Requires a self-loop (reads ``state[element]``).
    """
    modulator = tuple(modulation["modulator"])
    ins = _ordered_inputs(element, weights, inputs)
    true_inputs = [i for i in ins if i not in modulator]
    total_input = sum(
        state[i] * w for i, w in zip(true_inputs, input_weights, strict=False)
    )
    mods_on = sum(state[i] for i in modulator)
    unit_state = state[element] * 2 - 1
    new_threshold = threshold + unit_state * mods_on * modulation["threshold"]
    new_determinism = determinism + unit_state * mods_on * modulation["determinism"]
    return ceiling * (
        floor + (1 - floor) * _logistic(total_input, new_determinism, new_threshold)
    )


def stabilized_sigmoid(
    element,
    weights,
    state,
    *,
    input_weights,
    determinism,
    threshold,
    modulation,
    floor=0.0,
    ceiling=1.0,
    inputs=None,
    **kwargs,
):
    """Sigmoid whose modulators stabilize the unit's current state, more
    strongly the more modulators are active.

    ``modulation`` is ``{'modulator': tuple(indices), 'threshold': float,
    'determinism': float, 'selectivity': float}``. Requires a self-loop.

    The original ``substrate_modeler`` implementation built its per-unit TPM with
    the input and modulator axes in swapped order (a Fortran-reshape artifact),
    so the declared ``modulator`` was not the axis actually treated as the
    modulator. This port uses the documented convention consistently (the
    modulators are exactly ``modulation['modulator']``), so it does not
    reproduce that artifact.
    """
    modulator = tuple(modulation["modulator"])
    ins = _ordered_inputs(element, weights, inputs)
    true_inputs = [i for i in ins if i not in modulator]
    total_input = sum(
        state[i] * w for i, w in zip(true_inputs, input_weights, strict=False)
    )
    mods_on = sum(state[i] for i in modulator)
    if mods_on == 0:
        mods_on = 1 / modulation["selectivity"]
    ising_state = state[element] * 2 - 1
    new_threshold = threshold - ising_state * mods_on * modulation["threshold"]
    if mods_on == 0 or modulation["determinism"] == 0:
        new_determinism = determinism
    else:
        new_determinism = determinism * float(mods_on * modulation["determinism"]) ** (
            ising_state
        )
    return ceiling * (
        floor + (1 - floor) * _logistic(total_input, new_determinism, new_threshold)
    )


def biased_sigmoid(
    element,
    weights,
    state,
    *,
    input_weights,
    floor=0.0,
    ceiling=1.0,
    determinism=2.0,
    threshold=0.0,
    inputs=None,
    **kwargs,
):
    """Sigmoid biased toward the last input unit's state: when the bias unit is
    OFF the activation is divided by the bias factor; when ON, the inactivation
    is divided by it. The bias unit is the last of ``inputs``; the bias factor
    is the last of ``input_weights``.
    """
    ins = _ordered_inputs(element, weights, inputs)
    bias_index = ins[-1]
    true_inputs = ins[:-1]
    true_weights = input_weights[:-1]
    bias_factor = input_weights[-1]
    total_input = sum(
        state[i] * w for i, w in zip(true_inputs, true_weights, strict=False)
    )
    y = ceiling * (floor + (1 - floor) * _logistic(total_input, determinism, threshold))
    if state[bias_index] == 0:
        return y / bias_factor
    return 1 - (1 - y) / bias_factor


# Mechanisms that read the unit's own state and therefore need a self-loop.
STATE_DEPENDENT = frozenset(
    {
        "resonnator",
        "mismatch_corrector",
        "mismatch_pattern_detector",
        "modulated_sigmoid",
        "stabilized_sigmoid",
    }
)

# Mechanisms that weight their inputs (so ``input_weights`` map to real edge
# weights); the rest are connectivity-only (edges carry a marker weight of 1.0).
WEIGHTED = frozenset(
    {
        "sigmoid",
        "resonnator",
        "weighted_mean",
        "modulated_sigmoid",
        "stabilized_sigmoid",
        "biased_sigmoid",
    }
)

MECHANISMS = {
    "sigmoid": sigmoid,
    "resonnator": resonnator,
    "sor": sor_gate,
    "mismatch_pattern_detector": mismatch_pattern_detector,
    "gabor": gabor_gate,
    "copy": copy_gate,
    "and": and_gate,
    "or": or_gate,
    "xor": xor_gate,
    "democracy": democracy,
    "majority": majority,
    "weighted_mean": weighted_mean,
    "mismatch_corrector": mismatch_corrector,
    "modulated_sigmoid": modulated_sigmoid,
    "stabilized_sigmoid": stabilized_sigmoid,
    "biased_sigmoid": biased_sigmoid,
}
