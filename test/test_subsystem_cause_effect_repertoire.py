#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_subsystem_cause_and_effect_repertoire.py

import numpy as np
import pytest

import example_networks
from pyphi import Direction, Subsystem
from pyphi.models import Cut

# Get example networks
standard = example_networks.standard()

# Get example subsystems
standard_subsystem = example_networks.s()
simple_all_off = example_networks.simple_subsys_all_off()
simple_a_just_on = example_networks.simple_subsys_all_a_just_on()


full = tuple(range(standard.size))


# Set up test scenarios
# =====================
# Scenario structure:
# (
#     function to test,
#     subsystem,
#     mechanism,
#     purview,
#     expected result
# )
scenarios = [
# Cause repertoire {{{
# ====================
    # Default Matlab network {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Full network, no cut {{{
        # ------------------------
    (
        'cause_repertoire',
        standard_subsystem,
        [0],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        standard_subsystem,
        [0],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        standard_subsystem,
        [0, 1],
        [0, 2],
        np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2, order="F")
    ), (
        'cause_repertoire',
        standard_subsystem,
        [1],
        [2],
        np.array([1.0, 0.0]).reshape(1, 1, 2, order="F")
    ), (
        'cause_repertoire',
        standard_subsystem,
        [],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'cause_repertoire',
        standard_subsystem,
        [1],
        [],
        np.array([1])
    ),
        # }}}
        # Full network, with cut {{{
        # --------------------------
    (
        'cause_repertoire',
        Subsystem(standard, standard_subsystem.state, full,
                  cut=Cut((2,), (0, 1))),
        [0],
        [1],
        np.array([1 / 3, 2 / 3]).reshape(1, 2, 1, order="F")
    ),
        # }}}
        # Subset, with cut {{{
        # --------------------
    (
        'cause_repertoire',
        Subsystem(standard, standard_subsystem.state, (1, 2),
                  cut=Cut((1,), (2,))),
        [2],
        [1, 2],
        np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2, order="F")
    ), (
        'cause_repertoire',
        Subsystem(standard, standard_subsystem.state, (1, 2),
                  cut=Cut((1,), (2,))),
        [2],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'cause_repertoire',
        Subsystem(standard, standard_subsystem.state, (0, 2),
                  cut=Cut((0,), (2,))),
        [2],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
    ),
        # }}}
    # }}}
    # Simple 'AND' network {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~
        # State: 'A' just turned on {{{
        # -----------------------------
    (
        'cause_repertoire',
        simple_a_just_on,
        [0],
        [0],
        # Cause repertoire is maximally selective; the previous state must have
        # been {0,1,1}, so `expected[(0,1,1)]` should be 1 and everything else
        # should be 0
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        simple_a_just_on,
        [],
        [0],
        # No matter the state of the purview (m0), the probability it will be
        # on in the next timestep is 1/8
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        simple_a_just_on,
        [1],
        [0, 1, 2],
        np.ones((2, 2, 2)) / 8
    ), (
        'cause_repertoire',
        simple_a_just_on,
        [0, 1],
        [0, 2],
        np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2, order="F")
    ),
        # }}}
        # State: all nodes off {{{
        # ------------------------
    (
        'cause_repertoire',
        simple_all_off,
        [0],
        [0],
        np.array([(3 / 7), (4 / 7)]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        simple_all_off,
        [0],
        [0, 1, 2],
        # Cause repertoire is minimally selective; only {0,1,1} is ruled out,
        # so probability density should be uniformly distributed among all
        # states not including {0,1,1} when purview is whole network
        np.array([1., 1., 1., 0., 1., 1., 1., 1.]).reshape(2, 2, 2) / 7
    ),
        # }}}
    # }}}
# }}}
# Effect repertoire {{{
# =====================
    # Default Matlab network {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Full network, no cut {{{
        # ------------------------

    (
        'effect_repertoire',
        standard_subsystem,
        [0],
        [0],
        np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        standard_subsystem,
        [0, 1],
        [0, 2],
        np.array([0.0, 0.0, 0.5, 0.5]).reshape(2, 1, 2, order="F")
    ), (
        'effect_repertoire',
        standard_subsystem,
        [1],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'effect_repertoire',
        standard_subsystem,
        [],
        [1],
        np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
    ), (
        'effect_repertoire',
        standard_subsystem,
        [2],
        [],
        np.array([1])
    ), (
        'effect_repertoire',
        standard_subsystem,
        [],
        [0],
        np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        standard_subsystem,
        [0],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'effect_repertoire',
        standard_subsystem,
        [1, 2],
        [0],
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        standard_subsystem,
        [1],
        [],
        np.array([1])
    ),
        # }}}
        # Full network, with cut {{{
        # --------------------------
    (
        'effect_repertoire',
        Subsystem(standard, standard_subsystem.state, full,
                  cut=Cut((0, 2), (1,))),
        [0],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'effect_repertoire',
        Subsystem(standard, standard_subsystem.state, full,
                  cut=Cut((0, 2), (1,))),
        [0, 1, 2],
        [0, 2],
        np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2, order="F")
    ),
        # }}}
        # Subset, with cut {{{
        # --------------------
    (
        'effect_repertoire',
        Subsystem(standard, standard_subsystem.state, (1, 2),
                  cut=Cut((1,), (2,))),
        [1],
        [1, 2],
        np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2, order="F")
    ), (
        'effect_repertoire',
        Subsystem(standard, standard_subsystem.state, (1, 2),
                  cut=Cut((1,), (2,))),
        [],
        [1],
        np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
    ), (
        'effect_repertoire',
        Subsystem(standard, standard_subsystem.state, (1, 2),
                  cut=Cut((1,), (2,))),
        [1],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ),
        # }}}
    # }}}
    # Simple 'AND' network {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~
        # State: 'A' just turned on {{{
        # -----------------------------
    (
        'effect_repertoire',
        simple_a_just_on,
        [0],
        [0],
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        simple_a_just_on,
        [],
        [0],
        # No matter the state of the purview {m0}, the probability it will
        # be on in the next timestep is 1/8
        np.array([0.875, 0.125]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        simple_a_just_on,
        [1],
        [0, 1, 2],
        np.array([1., 0., 0., 0., 0., 0., 0., 0.]).reshape(2, 2, 2)
    ), (
        'effect_repertoire',
        simple_a_just_on,
        [1],
        [0, 2],
        np.array([1.0, 0.0, 0.0, 0.0]).reshape(2, 1, 2, order="F")
    ),
        # }}}
        # State: all nodes off {{{
        # ------------------------
    (
        'effect_repertoire',
        simple_all_off,
        [0],
        [0],
        np.array([0.75, 0.25]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        simple_all_off,
        [0],
        [0, 1, 2],
        np.array([0.75, 0., 0., 0., 0.25, 0., 0., 0.]).reshape(2, 2, 2)
    )
        # }}}
    # }}}
# }}}
]


parameter_string = "function,subsystem,mechanism,purview,expected"


@pytest.mark.parametrize(parameter_string, scenarios)
def test_cause_and_effect_repertoire(function, subsystem, mechanism, purview,
                                     expected):
    """Test ``effect_repertoire`` or ``cause_repertoire``."""

    print("\nTesting " + function + " with subsystem \n" + str(subsystem))

    # Set up testing parameters from scenario
    compute_repertoire = getattr(subsystem, function)
    mechanism = tuple(mechanism)
    purview = tuple(purview)

    result = compute_repertoire(mechanism, purview)

    print("\nMechanism:".rjust(12), mechanism, "\nPurview:".rjust(12), purview,
          "\nCut:".rjust(12), subsystem.cut, "\n")

    print('-' * 40, "Result:", result, "\nResult Shape:", result.shape,
          '-' * 40, "Expected:", expected, "\nExpected Shape:", expected.shape,
          '-' * 40, sep="\n")

    assert np.array_equal(result, expected)


def test_repertoire_wrong_direction_error(s):
    with pytest.raises(ValueError):
        s.repertoire(Direction.BIDIRECTIONAL, (0,), (0, 1))

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
