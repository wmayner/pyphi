"""Tests for enforcement of the Realization principle (Albantakis et al. 2019)."""

import numpy as np
import pytest

from pyphi import actual
from pyphi import examples
from pyphi import exceptions
from pyphi import validate
from pyphi.direction import Direction
from pyphi.models import NullCut
from pyphi.substrate import Substrate


@pytest.fixture
def swap_substrate():
    # Each unit copies the other's previous state: unit 0 next = unit 1
    # previous, unit 1 next = unit 0 previous. Deterministic, so from
    # (0, 0) the only successor is (0, 0).
    tpm = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    cm = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    return Substrate(tpm, cm)


def test_transition_unreachable_error_attributes():
    err = exceptions.TransitionUnreachableError((0, 0), (1, 0))
    assert isinstance(err, exceptions.StateUnreachableError)
    assert err.before_state == (0, 0)
    assert err.after_state == (1, 0)
    assert err.state == (1, 0)


def test_transition_states_rejects_impossible_pair(swap_substrate):
    # Unit 0 cannot turn on from (0, 0).
    with pytest.raises(exceptions.TransitionUnreachableError):
        validate.transition_states(swap_substrate, (0, 0), (1, 0))


def test_transition_states_accepts_possible_pair(swap_substrate):
    validate.transition_states(swap_substrate, (0, 0), (0, 0))
    validate.transition_states(swap_substrate, (1, 0), (0, 1))


def test_transition_states_rejects_malformed_states(swap_substrate):
    with pytest.raises(ValueError):
        validate.transition_states(swap_substrate, (0, 0, 0), (0, 0))
    with pytest.raises(ValueError):
        validate.transition_states(swap_substrate, (0, 0), (0, 2))


def test_unrealizable_transition_raises(swap_substrate):
    # Effect side contains unit 0, whose observed after-state (on) is
    # impossible from (0, 0).
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.Transition(swap_substrate, (0, 0), (1, 0), (1,), (0,))


def test_realized_candidate_within_impossible_pair_constructs(swap_substrate):
    # Unit 1's observed after-state (off) has probability 1 from (0, 0),
    # so this candidate transition satisfies Realization even though the
    # full observed pair is impossible. Rejecting the pair is the job of
    # the analysis entry points, not the Transition object.
    t = actual.Transition(swap_substrate, (0, 0), (1, 0), (0,), (1,))
    assert t.effect_indices == (1,)


def test_realizable_transition_constructs(swap_substrate):
    t = actual.Transition(swap_substrate, (1, 0), (0, 1), (0, 1), (0, 1))
    assert t.node_indices == (0, 1)


def test_null_transition_constructs(swap_substrate):
    # An empty effect side is trivially realized (empty product = 1),
    # even within an impossible observed pair.
    t = actual.Transition(swap_substrate, (0, 0), (1, 0), (), ())
    assert len(t) == 0


def test_explicit_partition_bypasses_check(swap_substrate):
    # An explicit partition marks a derived copy (apply_cut path); the
    # unpartitioned parent is where validation happens.
    t = actual.Transition(
        swap_substrate,
        (0, 0),
        (1, 0),
        (1,),
        (0,),
        partition=NullCut((0, 1), swap_substrate.node_labels),
    )
    assert t.effect_indices == (0,)


def test_apply_cut_does_not_recheck(swap_substrate):
    t = actual.Transition(swap_substrate, (1, 0), (0, 1), (0,), (1,))
    cut = NullCut((0, 1), swap_substrate.node_labels)
    assert t.apply_cut(cut).partition is cut


def test_noised_background_can_realize(swap_substrate):
    # Unit 0's next state copies unit 1, which is background here (the
    # transition is over unit 0 alone). Frozen at 0, after-state 1 is
    # impossible; noised, it has probability 1/2.
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.Transition(swap_substrate, (0, 0), (1, 0), (0,), (0,))
    t = actual.Transition(
        swap_substrate, (0, 0), (1, 0), (0,), (0,), noise_background=True
    )
    assert t.probability(Direction.EFFECT, (0,), (0,)) == pytest.approx(0.5)


def test_transitions_raises_eagerly(swap_substrate):
    # No iteration: the pair is validated at call time.
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.transitions(swap_substrate, (0, 0), (1, 0))


def test_transitions_yields_all_candidates_for_valid_pair(swap_substrate):
    # For a realizable pair, every cm-supported candidate is realized
    # (each unit's factor is positive, so every subset product is), and
    # the sweep yields all (2^2 - 1)^2 of them.
    assert len(list(actual.transitions(swap_substrate, (1, 0), (0, 1)))) == 9


def test_causal_nexus_rejects_impossible_pair(swap_substrate):
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.causal_nexus(swap_substrate, (0, 0), (1, 0))


def test_nexus_rejects_impossible_pair(swap_substrate):
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.nexus(swap_substrate, (0, 0), (1, 0))


def test_causal_nexus_valid_pair_still_works(swap_substrate):
    result = actual.causal_nexus(swap_substrate, (1, 0), (0, 1))
    assert result is not None


def test_events_rejects_impossible_first_pair(swap_substrate):
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.events(swap_substrate, (0, 0), (1, 0), (0, 1), (0, 1))


def test_events_rejects_impossible_second_pair(swap_substrate):
    with pytest.raises(exceptions.TransitionUnreachableError):
        actual.events(swap_substrate, (0, 0), (0, 0), (1, 0), (0, 1))


def test_events_valid_triplet_still_works(swap_substrate):
    result = actual.events(swap_substrate, (1, 0), (0, 1), (1, 0), (0, 1))
    assert isinstance(result, tuple)


def test_paper_example_transitions_construct():
    """The shipped disjunction-conjunction example's own paper transitions
    are accepted: Realization constrains only p(after | before) > 0, not
    reachability of the before-state itself (in this substrate the input
    units reset to 0, so any before-state with an input ON has no
    predecessor -- and is still a valid occurrence).
    """
    substrate = examples.disjunction_conjunction_substrate()
    for before in [(1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1)]:
        transition = actual.Transition(substrate, before, (0, 0, 0, 1), (0, 1, 2), (3,))
        assert actual.account(transition)
