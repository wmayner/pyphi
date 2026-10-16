"""Mechanisms and purviews may be given in any order, as labels or as numpy
arrays: every query returns what it returns for the sorted indices."""

import itertools

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi import formalism as F
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction

FIG1A_STATE = (0, 1, 1)
FORMALISMS_4 = ["IIT_4_0_2026", "IIT_4_0_2023"]
GID = "GENERALIZED_INTRINSIC_DIFFERENCE"


def _fig1a_system():
    return pyphi.System(examples.iit4_2023_fig1a_substrate(), FIG1A_STATE)


@pytest.mark.parametrize("formalism", FORMALISMS_4)
def test_free_functions_do_not_depend_on_mechanism_order(formalism):
    with config.override(**presets.by_name[formalism]):
        system = _fig1a_system()
        assert F.distinction(system, (0, 1)).phi > 0
        for mechanism in [(0, 1), (0, 2), (1, 2), (0, 1, 2)]:
            for shuffled in itertools.permutations(mechanism):
                assert F.distinction(system, shuffled).phi == pytest.approx(
                    F.distinction(system, mechanism).phi
                )
                mic, ref = F.mic(system, shuffled), F.mic(system, mechanism)
                assert (mic.purview, mic.phi) == (ref.purview, pytest.approx(ref.phi))
                for purview in [(0, 1, 2), (2, 1, 0)]:
                    assert F.find_mip(
                        system, Direction.CAUSE, shuffled, purview
                    ).phi == pytest.approx(
                        F.find_mip(system, Direction.CAUSE, mechanism, (0, 1, 2)).phi
                    )


def test_unsorted_mechanism_on_a_kary_substrate():
    with config.override(**presets.iit4_2026):
        system = pyphi.System(examples.gomez_p53_mdm2_substrate(), (2, 0, 1))
        for direction in Direction.both():
            assert F.find_mip(system, direction, (1, 0), (0,)).phi == pytest.approx(
                F.find_mip(system, direction, (0, 1), (0,)).phi
            )


def test_forward_cause_probability_follows_the_given_order():
    """States given with a shuffled mechanism or purview follow that order."""
    with config.override(**presets.iit4_2026):
        system = _fig1a_system()
        mechanism, purview, purview_state = (0, 2), (0, 1, 2), (1, 0, 1)
        mechanism_state = (FIG1A_STATE[0], FIG1A_STATE[2])
        ref = system.forward_cause_probability(mechanism, purview, purview_state)
        assert ref > 0
        assert system.forward_cause_probability(
            (2, 0), purview, purview_state
        ) == pytest.approx(ref)
        assert system.forward_cause_probability(
            (2, 0), (2, 1, 0), purview_state[::-1], mechanism_state[::-1]
        ) == pytest.approx(ref)
        assert np.allclose(
            system.forward_cause_repertoire((2, 0), purview),
            system.forward_cause_repertoire(mechanism, purview),
        )


@pytest.mark.parametrize("direction", list(Direction.both()))
def test_intrinsic_specification_pairs_the_state_with_its_units(direction):
    with config.override(**presets.iit4_2026):
        system = _fig1a_system()
        ref = system.intrinsic_specification(
            direction, (0, 1), (0, 1, 2), specification_measure=GID
        )
        got = system.intrinsic_specification(
            direction, (1, 0), (2, 0, 1), specification_measure=GID
        )
    assert got.intrinsic_specification == pytest.approx(ref.intrinsic_specification)
    assert dict(zip(got.purview, got.state, strict=True)) == dict(
        zip(ref.purview, ref.state, strict=True)
    )


SPELLINGS = [
    lambda units: units,
    lambda units: tuple(reversed(units)),
    lambda units: list(units),
    lambda units: np.array(units),
    lambda units: tuple("ABC"[u] for u in units),
    lambda units: tuple("ABC"[u] for u in reversed(units)),
]


@pytest.mark.parametrize("spell", SPELLINGS)
def test_repertoire_methods_accept_any_spelling(spell):
    with config.override(**presets.iit3):
        system = _fig1a_system()
        mechanism, purview = (0, 1), (0, 2)
        assert system.cause_repertoire(mechanism, purview).size == 4
        for method in ("cause_repertoire", "effect_repertoire", "cause_info"):
            query = getattr(system, method)
            np.testing.assert_allclose(
                query(spell(mechanism), spell(purview)), query(mechanism, purview)
            )
        for method in (
            "unconstrained_cause_repertoire",
            "unconstrained_effect_repertoire",
        ):
            query = getattr(system, method)
            np.testing.assert_allclose(query(spell(purview)), query(purview))
        np.testing.assert_allclose(
            system.repertoire(Direction.CAUSE, spell(mechanism), spell(purview)),
            system.repertoire(Direction.CAUSE, mechanism, purview),
        )


def test_a_state_of_the_wrong_length_is_rejected_in_any_spelling():
    with config.override(**presets.iit4_2026):
        system = _fig1a_system()
        for mechanism in [(0, 2), [0, 2], (2, 0), ("A", "C")]:
            with pytest.raises(ValueError, match="one entry per unit"):
                system.effect_repertoire(mechanism, (1,), mechanism_state=(1, 0, 1))
        for purview in [(1, 2), (2, 1)]:
            with pytest.raises(ValueError, match="one entry per unit"):
                system.forward_cause_probability((0,), purview, (1, 0, 1))
        assert np.allclose(
            system.effect_repertoire(iter((2, 0)), (1,), mechanism_state=(1, 0)),
            system.effect_repertoire((0, 2), (1,), mechanism_state=(0, 1)),
        )
