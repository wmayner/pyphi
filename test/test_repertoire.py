import numpy as np

from pyphi import repertoire
from pyphi.distribution import repertoire_shape
from pyphi.utils import all_states


def test_forward_effect_repertoire_matches_subsystem(s):
    mechanism = (0,)
    purview = (1, 2)
    expected = s.effect_repertoire(mechanism, purview)
    actual = repertoire.forward_effect_repertoire(s, mechanism, purview)
    assert np.allclose(actual, expected)


def test_forward_effect_probability_matches_repertoire(s):
    mechanism = (0,)
    purview = (1, 2)
    purview_state = (1, 0)
    expected = repertoire.forward_effect_repertoire(s, mechanism, purview).squeeze()[
        purview_state
    ]
    actual = repertoire.forward_effect_probability(
        s, mechanism, purview, purview_state
    )
    assert np.isclose(actual, expected)


def test_forward_cause_probability_matches_repertoire(s):
    mechanism = (0,)
    purview = (1, 2)
    purview_state = (1, 0)
    expected = repertoire.forward_cause_repertoire(
        s, mechanism, purview, purview_state=purview_state
    ).squeeze()[purview_state]
    actual = repertoire.forward_cause_probability(
        s, mechanism, purview, purview_state
    )
    assert np.isclose(actual, expected)


def test_forward_cause_repertoire_shape_empty_purview(s):
    mechanism = (0,)
    purview = ()
    actual = repertoire.forward_cause_repertoire(s, mechanism, purview)
    expected_shape = tuple(repertoire_shape(s.network.node_indices, purview))
    assert actual.shape == expected_shape


def test_unconstrained_forward_effect_repertoire_is_mean(s):
    mechanism = (0, 1)
    purview = (2,)
    expected = np.stack(
        [
            s.forward_effect_repertoire(mechanism, purview, mechanism_state=state)
            for state in all_states(len(mechanism))
        ]
    ).mean(axis=0)
    actual = repertoire.unconstrained_forward_effect_repertoire(
        s, mechanism, purview
    )
    assert np.allclose(actual, expected)


def test_unconstrained_forward_cause_repertoire_is_uniform(s):
    mechanism = (0,)
    purview = (1, 2)
    actual = repertoire.unconstrained_forward_cause_repertoire(
        s, mechanism, purview
    )
    assert np.allclose(actual, actual.flat[0])
