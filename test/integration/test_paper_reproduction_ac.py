"""Actual-causation paper reproductions -- Albantakis, Marshall, Hoel & Tononi
(2019), "What caused what?", Entropy 21(5):459.

Paper-sourced pins of the causal accounts in Figs 7-16. Values are quoted to
the paper's precision (three decimals, or two where the figure gives two).
Input units are modeled as self-copying (their next state repeats their
current state), the convention the Fig 8B reproduction in
``test_paper_reproduction.py`` uses; input dynamics do not enter links whose
effect set is the output unit.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import actual
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.substrate import Substrate


@pytest.fixture
def _iit3():
    with config.override(
        **presets.iit3, validate_system_states=False, progress_bars=False
    ):
        yield


def _gate_substrate(n_inputs, outputs, labels=None, alphabet=None):
    """A substrate of ``n_inputs`` self-copying input units followed by output
    units. ``outputs`` maps an output name to ``f(*input_states) -> state``
    (deterministic) or ``-> dict[state, probability]`` (stochastic). Output
    units read only the inputs. ``alphabet`` gives per-unit alphabet sizes
    (default binary)."""
    names = list(outputs)
    n = n_inputs + len(names)
    sizes = tuple(alphabet) if alphabet is not None else (2,) * n
    marginals = []
    for i in range(n):
        factor = np.zeros((*sizes, sizes[i]))
        for state in np.ndindex(*sizes):
            if i < n_inputs:
                factor[(*state, state[i])] = 1.0
            else:
                result = outputs[names[i - n_inputs]](*state[:n_inputs])
                if isinstance(result, dict):
                    for target, p in result.items():
                        factor[(*state, target)] = p
                else:
                    factor[(*state, result)] = 1.0
        marginals.append(factor)
    if labels is None:
        labels = tuple("ABCDEFGHIJ"[:n_inputs]) + tuple(names)
    return Substrate(
        marginals=marginals,
        state_space=tuple(tuple(range(k)) for k in sizes),
        node_labels=labels,
    )


def _account(substrate, before, after, cause_indices, effect_indices):
    """{(direction, mechanism): (purview, alpha rounded to 3 dp)}, plus the links."""
    transition = actual.Transition(
        substrate, before, after, cause_indices, effect_indices
    )
    links = list(actual.account(transition))
    return {
        (link.direction, tuple(link.mechanism)): (
            tuple(link.purview),
            round(float(link.alpha), 3),
        )
        for link in links
    }, links


# --------------------------------------------------------------------------- #
# Fig 7 -- four gates, one transition {AB = 11} -> {out = 1}
# --------------------------------------------------------------------------- #


def test_ac_fig7a_disjunction(_iit3):
    """Fig 7A (OR): {A} -> {C} and {B} -> {C} at 0.415 each; the actual cause
    of {C = 1} is undetermined between {A} and {B} at 0.415 (symmetric
    over-determination); {AB} is reducible on the effect side."""
    s = _gate_substrate(2, {"C": lambda a, b: a | b})
    account, links = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    assert account[(Direction.EFFECT, (0,))] == ((2,), 0.415)
    assert account[(Direction.EFFECT, (1,))] == ((2,), 0.415)
    assert (Direction.EFFECT, (0, 1)) not in account
    cause = next(
        link
        for link in links
        if link.direction == Direction.CAUSE and tuple(link.mechanism) == (2,)
    )
    assert round(float(cause.alpha), 3) == 0.415
    assert cause.purview_ties is not None and len(cause.purview_ties) == 2


def test_ac_fig7b_conjunction(_iit3):
    """Fig 7B (AND): {A}, {B}, {AB} each -> {D} at 1.0; the one actual cause
    of {D = 1} is {AB = 11} at 2.0."""
    s = _gate_substrate(2, {"D": lambda a, b: a & b})
    account, _ = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    for mech in ((0,), (1,), (0, 1)):
        assert account[(Direction.EFFECT, mech)] == ((2,), 1.0)
    assert account[(Direction.CAUSE, (2,))] == ((0, 1), 2.0)


def test_ac_fig7c_biconditional(_iit3):
    """Fig 7C (XNOR): only the second-order occurrence links, {AB} <-> {E}
    at 1.0 each way; the parts have zero effect information."""
    s = _gate_substrate(2, {"E": lambda a, b: int(a == b)})
    account, _ = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    assert account[(Direction.EFFECT, (0, 1))] == ((2,), 1.0)
    assert account[(Direction.CAUSE, (2,))] == ((0, 1), 1.0)
    assert (Direction.EFFECT, (0,)) not in account
    assert (Direction.EFFECT, (1,)) not in account


def test_ac_fig7d_prevention(_iit3):
    """Fig 7D (prevention: F = 0 only for AB = 10): {B} <-> {F} at 0.415;
    {A} has no effect and is not a cause."""
    s = _gate_substrate(2, {"F": lambda a, b: 0 if (a, b) == (1, 0) else 1})
    account, _ = _account(s, (1, 1, 0), (1, 1, 1), (0, 1), (2,))
    assert account[(Direction.EFFECT, (1,))] == ((2,), 0.415)
    assert account[(Direction.CAUSE, (2,))] == ((1,), 0.415)
    assert (Direction.EFFECT, (0,)) not in account


# --------------------------------------------------------------------------- #
# Fig 8A -- majority gate, ABCD = 1110 -> M = 1
# --------------------------------------------------------------------------- #


def test_ac_fig8a_majority(_iit3):
    """Fig 8A: singleton effects 0.678, pairs 0.585, {ABC} 0.415; the actual
    cause {ABC = 111} <- {M = 1} at 1.678; D = 0 links nowhere."""
    s = _gate_substrate(4, {"M": lambda a, b, c, d: int(a + b + c + d >= 3)})
    account, _ = _account(s, (1, 1, 1, 0, 0), (1, 1, 1, 0, 1), (0, 1, 2, 3), (4,))
    for mech in ((0,), (1,), (2,)):
        assert account[(Direction.EFFECT, mech)] == ((4,), 0.678)
    for mech in ((0, 1), (0, 2), (1, 2)):
        assert account[(Direction.EFFECT, mech)] == ((4,), 0.585)
    assert account[(Direction.EFFECT, (0, 1, 2))] == ((4,), 0.415)
    assert account[(Direction.CAUSE, (4,))] == ((0, 1, 2), 1.678)
    assert not any(3 in mech for (d, mech) in account if d == Direction.EFFECT)


# --------------------------------------------------------------------------- #
# Fig 9 -- disjunction of conjunctions (A and B) or C, ABC = 101 -> D = 1
# --------------------------------------------------------------------------- #
# The example substrate's inputs go to OFF at the next step, so the after
# state of the inputs is (0, 0, 0); the paper says nothing about it.


def test_ac_fig9a_disjunction_of_conjunctions(_iit3):
    """Fig 9A: {A} -> {D} at 0.263, {C} -> {D} at 0.678; the actual cause of
    {D = 1} is {C} at 0.678."""
    s = examples.disjunction_conjunction_substrate()
    account, _ = _account(s, (1, 0, 1, 0), (0, 0, 0, 1), (0, 1, 2), (3,))
    assert account[(Direction.EFFECT, (0,))] == ((3,), 0.263)
    assert account[(Direction.EFFECT, (2,))] == ((3,), 0.678)
    assert account[(Direction.CAUSE, (3,))] == ((2,), 0.678)


def test_ac_fig9b_background_b(_iit3):
    """Fig 9B: with B = 0 a fixed background condition, {C} <-> {D} at 1.0 and
    {A} has no actual effect."""
    s = examples.disjunction_conjunction_substrate()
    account, _ = _account(s, (1, 0, 1, 0), (0, 0, 0, 1), (0, 2), (3,))
    assert account[(Direction.EFFECT, (2,))] == ((3,), 1.0)
    assert account[(Direction.CAUSE, (3,))] == ((2,), 1.0)
    assert (Direction.EFFECT, (0,)) not in account


# --------------------------------------------------------------------------- #
# Fig 12 -- a noisy COPY
# --------------------------------------------------------------------------- #


def test_ac_fig12_noisy_copy(_iit3):
    """Fig 12: N copies A with probability 0.9. {A = 1} -> {N = 1} and
    {A = 1} <- {N = 1} at 0.848 (12A); {A = 1} -> {N = 0} has no causal links
    (12B)."""
    s = _gate_substrate(1, {"N": lambda a: {1: 0.9, 0: 0.1} if a else {1: 0.1, 0: 0.9}})
    account, _ = _account(s, (1, 0), (1, 1), (0,), (1,))
    assert account[(Direction.EFFECT, (0,))] == ((1,), 0.848)
    assert account[(Direction.CAUSE, (1,))] == ((0,), 0.848)
    account_b, _ = _account(s, (1, 0), (1, 0), (0,), (1,))
    assert account_b == {}
