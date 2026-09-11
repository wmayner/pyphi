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

import itertools

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


# --------------------------------------------------------------------------- #
# Fig 10 -- complicated voting, ABCDE = 11000 -> F = 1
# --------------------------------------------------------------------------- #


def _complicated_vote(a, b, c, d, e):
    if a == b:
        return a
    if b == c == d == e:
        return a
    return int(a + b + c + d + e >= 3)


def test_ac_fig10_complicated_voting(_iit3):
    """Fig 10: effects {A} 0.70, {B} 0.46, {AB} 0.30, {ACDE} 0.30; the actual
    cause of {F = 1} is undetermined between {AB = 11} and {ACDE = 1000},
    both at 1.0."""
    s = _gate_substrate(5, {"F": _complicated_vote})
    account, links = _account(
        s, (1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 0, 1), (0, 1, 2, 3, 4), (5,)
    )
    two = {k: (p, round(a, 2)) for k, (p, a) in account.items()}
    assert two[(Direction.EFFECT, (0,))] == ((5,), 0.70)
    assert two[(Direction.EFFECT, (1,))] == ((5,), 0.46)
    assert two[(Direction.EFFECT, (0, 1))] == ((5,), 0.30)
    assert two[(Direction.EFFECT, (0, 2, 3, 4))] == ((5,), 0.30)
    cause = next(link for link in links if link.direction == Direction.CAUSE)
    assert round(float(cause.alpha), 3) == 1.0
    tied = {tuple(ria.purview) for ria in (cause.purview_ties or ())}
    assert tied == {(0, 1), (0, 2, 3, 4)}


# --------------------------------------------------------------------------- #
# Fig 11 -- three candidates, seven voters (multi-valued)
# --------------------------------------------------------------------------- #


def test_ac_fig11_three_candidate_election(_iit3):
    """Fig 11: five votes for "1" (state 0) and two for "2" (state 1) elect
    "1" (W = 1). Effects of the "1" voters by occurrence order: 0.718 (one),
    0.581 (two), 0.404 (three), 0.190 (four); the actual cause of {W = 1} is
    an undetermined set of four "1" voters at 1.893; the "2" votes {F}, {G}
    have alpha = 0 (no links). The suite's first multi-valued AC pin."""
    s = examples.ac_2019_three_candidate_election_substrate()
    before = (0, 0, 0, 0, 0, 1, 1, 0)
    after = (0, 0, 0, 0, 0, 1, 1, 1)
    account, links = _account(s, before, after, tuple(range(7)), (7,))
    # PyPhi gives 0.5805 for the pairs, which the paper prints as 0.581.
    effects = {
        tuple(link.mechanism): (tuple(link.purview), float(link.alpha))
        for link in links
        if link.direction == Direction.EFFECT
    }
    for mech, alpha in (
        ((0,), 0.718),
        ((0, 1), 0.581),
        ((0, 1, 2), 0.404),
        ((0, 1, 2, 3), 0.190),
    ):
        purview, value = effects[mech]
        assert purview == (7,)
        assert value == pytest.approx(alpha, abs=0.001)
    assert not any(5 in m or 6 in m for (d, m) in account if d == Direction.EFFECT)
    cause = next(link for link in links if link.direction == Direction.CAUSE)
    assert round(float(cause.alpha), 3) == 1.893
    tied = {tuple(ria.purview) for ria in (cause.purview_ties or ())}
    assert tied == set(itertools.combinations(range(5), 4))


# --------------------------------------------------------------------------- #
# Fig 13 -- dot / segment / line classifier, ABC = 001 -> DSL = 100
# --------------------------------------------------------------------------- #


def test_ac_fig13_classifier(_iit3):
    """Fig 13: effects {A=0} -> {DL=10} 0.608, {B=0} -> {DSL=100} 1.02,
    {ABC=001} -> {D=1} 1.0; causes {ABC=001} <- {D=1} 1.415, {B=0} <- {S=0}
    0.415, {{A=0},{B=0}} <- {L=0} 0.193, {B=0} <- {DS=10} 0.263,
    {{A=0},{B=0}} <- {DL=10} 0.126, {B=0} <- {SL=00} 0.126,
    {B=0} <- {DSL=100} 0.074."""
    s = _gate_substrate(
        3,
        {
            "D": lambda a, b, c: int(a + b + c == 1),
            "S": lambda a, b, c: int((a, b, c) in ((1, 1, 0), (0, 1, 1))),
            "L": lambda a, b, c: int((a, b, c) == (1, 1, 1)),
        },
    )
    account, links = _account(
        s, (0, 0, 1, 0, 0, 0), (0, 0, 1, 1, 0, 0), (0, 1, 2), (3, 4, 5)
    )
    assert account[(Direction.EFFECT, (0,))] == ((3, 5), 0.608)
    purview, value = account[(Direction.EFFECT, (1,))]
    assert (purview, round(value, 2)) == ((3, 4, 5), 1.02)  # the figure gives 2 dp
    assert account[(Direction.EFFECT, (0, 1, 2))] == ((3,), 1.0)
    assert account[(Direction.CAUSE, (3,))] == ((0, 1, 2), 1.415)
    assert account[(Direction.CAUSE, (4,))] == ((1,), 0.415)
    assert account[(Direction.CAUSE, (3, 4))] == ((1,), 0.263)
    assert account[(Direction.CAUSE, (4, 5))] == ((1,), 0.126)
    assert account[(Direction.CAUSE, (3, 4, 5))] == ((1,), 0.074)
    # Undetermined causes: {A=0} or {B=0} for {L=0} (0.193) and {DL=10} (0.126).
    by_mech = {
        tuple(link.mechanism): link
        for link in links
        if link.direction == Direction.CAUSE
    }
    for mech, alpha in (((5,), 0.193), ((3, 5), 0.126)):
        link = by_mech[mech]
        assert round(float(link.alpha), 3) == alpha
        assert {tuple(r.purview) for r in (link.purview_ties or ())} == {(0,), (1,)}


# --------------------------------------------------------------------------- #
# Fig 15 -- double bi-conditional, ABC = 111 -> DE = 11
# --------------------------------------------------------------------------- #


def test_ac_fig15_double_biconditional(_iit3):
    """Fig 15: {AB} -> {D}, {BC} -> {E}, {ABC} -> {DE} and the three reverse
    causes all at 1.0 bits; no first-order links."""
    s = _gate_substrate(
        3, {"D": lambda a, b, _c: int(a == b), "E": lambda _a, b, c: int(b == c)}
    )
    account, _ = _account(s, (1, 1, 1, 0, 0), (1, 1, 1, 1, 1), (0, 1, 2), (3, 4))
    assert account[(Direction.EFFECT, (0, 1))] == ((3,), 1.0)
    assert account[(Direction.EFFECT, (1, 2))] == ((4,), 1.0)
    assert account[(Direction.EFFECT, (0, 1, 2))] == ((3, 4), 1.0)
    assert account[(Direction.CAUSE, (3,))] == ((0, 1), 1.0)
    assert account[(Direction.CAUSE, (4,))] == ((1, 2), 1.0)
    assert account[(Direction.CAUSE, (3, 4))] == ((0, 1, 2), 1.0)
    assert not any(len(m) == 1 for (d, m) in account if d == Direction.EFFECT)


# --------------------------------------------------------------------------- #
# Fig 16 -- irreducible vs reducible second-order occurrence
# --------------------------------------------------------------------------- #


def test_ac_fig16a_shared_inputs_irreducible(_iit3):
    """Fig 16A: OR and AND share inputs A, B; {AB = 10} <- {(OR,AND) = 10} at
    0.170 in addition to the four first-order links at 0.415; the
    transition's irreducibility is 0.17 bits."""
    s = _gate_substrate(2, {"OR": lambda a, b: a | b, "AND": lambda a, b: a & b})
    account, _ = _account(s, (1, 0, 0, 0), (1, 0, 1, 0), (0, 1), (2, 3))
    assert account[(Direction.EFFECT, (0,))] == ((2,), 0.415)
    assert account[(Direction.EFFECT, (1,))] == ((3,), 0.415)
    assert account[(Direction.CAUSE, (2,))] == ((0,), 0.415)
    assert account[(Direction.CAUSE, (3,))] == ((1,), 0.415)
    assert account[(Direction.CAUSE, (2, 3))] == ((0, 1), 0.170)
    transition = actual.Transition(s, (1, 0, 0, 0), (1, 0, 1, 0), (0, 1), (2, 3))
    assert round(float(actual.sia(transition).alpha), 2) == 0.17


def test_ac_fig16c_independent_inputs_reducible(_iit3):
    """Fig 16C: with independent inputs (A, B -> OR; C, D -> AND) the
    second-order link is absent and the transition is reducible (0 bits)."""
    s = _gate_substrate(
        4, {"OR": lambda a, b, _c, _d: a | b, "AND": lambda _a, _b, c, d: c & d}
    )
    before, after = (1, 0, 1, 0, 0, 0), (1, 0, 1, 0, 1, 0)
    account, _ = _account(s, before, after, (0, 1, 2, 3), (4, 5))
    assert (Direction.CAUSE, (4, 5)) not in account
    transition = actual.Transition(s, before, after, (0, 1, 2, 3), (4, 5))
    assert float(actual.sia(transition).alpha) == 0.0
