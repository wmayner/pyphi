"""Purview labels are written in the specified state, and the per-direction
specification value is called intrinsic specification."""

import numpy as np
import pytest

import pyphi
from pyphi import examples


def _labels(analysis):
    return {
        d.mechanism: (d.cause_purview_label, d.effect_purview_label)
        for d in analysis.ces.distinctions
    }


def test_purview_labels_use_the_specified_state():
    # basic_substrate in (1, 1, 0): current and specified states differ.
    with pyphi.config.override(**pyphi.iit4_2026):
        a = pyphi.analyze(examples.basic_substrate(), (1, 1, 0))
    assert _labels(a) == {
        (1,): ("C", "A"),
        (2,): ("ab", "b"),
        (0, 1): ("C", "Ac"),
    }
    for d in a.ces.distinctions:
        assert d.cause_purview_label == d.cause.specified_state.purview_label
        assert d.effect_purview_label == d.effect.specified_state.purview_label


def _kary_substrate():
    """P has three values and Q two. Q = 1 drives P to 2, and P's value
    raises the chance that Q turns on."""
    p_given_q = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]).reshape(1, 2, 3)
    q_on = np.array([0.1, 0.5, 0.9])
    q_given_p = np.stack([1 - q_on, q_on], axis=-1).reshape(3, 1, 2)
    return pyphi.Substrate(
        marginals=[p_given_q, q_given_p],
        state_space=((0, 1, 2), (0, 1)),
        node_labels=("P", "Q"),
    )


def test_kary_purview_label_uses_the_specified_value():
    # P is currently 0, but the mechanism Q specifies P = 2 in both directions.
    with pyphi.config.override(**pyphi.iit4_2026):
        a = pyphi.analyze(_kary_substrate(), (0, 1))
    q = next(d for d in a.ces.distinctions if d.mechanism == (1,))
    assert q.cause.current_state == (0,)
    assert q.cause.specified_state.state == (2,)
    assert q.effect.specified_state.state == (2,)
    assert (q.cause_purview_label, q.effect_purview_label) == ("P₂", "P₂")
    assert q.cause_purview_label == q.cause.specified_state.purview_label
    assert q.effect_purview_label == q.effect.specified_state.purview_label


def test_ces_plot_endpoints_use_the_specified_state():
    from pyphi.visualize.projection import project_ces

    with pyphi.config.override(**pyphi.iit4_2026):
        a = pyphi.analyze(examples.basic_substrate(), (1, 1, 0))
    endpoints = project_ces(a.ces).endpoints
    labels = {(e.distinction_id, e.direction): e.label for e in endpoints}
    distinctions = list(a.ces.distinctions)
    for (i, direction), label in labels.items():
        mice = distinctions[i].cause if direction == "cause" else distinctions[i].effect
        assert label == mice.specified_state.purview_label


def test_per_direction_quantity_is_called_intrinsic_specification():
    with pyphi.config.override(**pyphi.iit4_2026):
        sia = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(),
            (0, 1, 1),
            subset=(0, 1),
            compute="sia",
        )
    cause = sia.system_state.cause
    assert cause.intrinsic_specification == pytest.approx(0.843228, abs=1e-6)
    assert not hasattr(cause, "intrinsic_information")
    assert not hasattr(cause, "runner_up_intrinsic_information")
    assert sia.intrinsic_information == pytest.approx(sia.phi)  # ii(s) binds here


def test_current_state_replaces_purview_state():
    with pyphi.config.override(**pyphi.iit4_2026):
        a = pyphi.analyze(examples.basic_substrate(), (1, 1, 0))
    b = next(d for d in a.ces.distinctions if d.mechanism == (1,))
    assert b.cause.current_state == (0,)
    assert not hasattr(b.cause, "purview_state")


def test_2023_analysis_card_has_no_ii_row():
    a = pyphi.analyze(
        examples.iit4_2023_fig1a_substrate(),
        (0, 1, 1),
        subset=(0, 1),
        formalism="IIT_4_0_2023",
    )
    assert "ii(s)" not in str(a)
    with pyphi.config.override(**pyphi.iit4_2026):
        a = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1))
    assert "ii(s)" in str(a)


@pytest.mark.parametrize("formalism", ["IIT_4_0_2023", "IIT_4_0_2026"])
def test_cards_call_the_per_direction_value_intrinsic_specification(formalism):
    a = pyphi.analyze(
        examples.iit4_2023_fig1a_substrate(),
        (0, 1, 1),
        subset=(0, 1),
        formalism=formalism,
    )
    cards = str(a.sia) + str(a.sia.system_state) + str(a.ces.distinctions[0].cause)
    assert "Intrinsic specification" in cards
    assert "Intrinsic information" not in cards
    margin_labels = [f.label for f in a.sia.explain().findings]
    assert "Specified-cause-state margin (i_spec)" in margin_labels
