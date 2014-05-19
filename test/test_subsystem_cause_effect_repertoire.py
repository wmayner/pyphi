#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from cyphi.models import Cut

import example_networks


# Get example subsystems
s = example_networks.s()
s_all_a_just_on = example_networks.s_subsys_all_a_just_on()
s_all_off = example_networks.s_subsys_all_off()
subsys_n0n2 = example_networks.subsys_n0n2()
subsys_n1n2 = example_networks.subsys_n1n2()


# Set up test scenarios
# =====================
# Scenario structure:
# (
#     function to test,
#     subsystem, cut,
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
        s, None,
        [0],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        s, None,
        [0],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        s, None,
        [0, 1],
        [0, 2],
        np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2, order="F")
    ), (
        'cause_repertoire',
        s, None,
        [1],
        [2],
        np.array([1.0, 0.0]).reshape(1, 1, 2, order="F")
    ), (
        'cause_repertoire',
        s, None,
        [],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'cause_repertoire',
        s, None,
        [1],
        [],
        np.array([1])
    ),
        # }}}
        # Full network, with cut {{{
        # --------------------------
    (
        'cause_repertoire',
        s, (2, (0, 1)),
        [0],
        [1],
        np.array([1/3, 2/3]).reshape(1, 2, 1, order="F")
    ),
        # }}}
        # Subset, with cut {{{
        # --------------------
    (
        'cause_repertoire',
        subsys_n1n2, (1, 2),
        [2],
        [1, 2],
        np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2, order="F")
    ), (
        'cause_repertoire',
        subsys_n1n2, (1, 2),
        [2],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'cause_repertoire',
        subsys_n0n2, (0, 2),
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
        s_all_a_just_on, None,
        [0],
        [0],
        # Cause repertoire is maximally selective; the past state must have
        # been {0,1,1}, so `expected[(0,1,1)]` should be 1 and everything else
        # should be 0
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        s_all_a_just_on, None,
        [],
        [0],
        # No matter the state of the purview (m0), the probability it will be
        # on in the next timestep is 1/8
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        s_all_a_just_on, None,
        [1],
        [0, 1, 2],
        np.ones((2, 2, 2)) / 8
    ), (
        'cause_repertoire',
        s_all_a_just_on, None,
        [0, 1],
        [0, 2],
        np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2, order="F")
    ),
        # }}}
        # State: all nodes off {{{
        # ------------------------
    (
        'cause_repertoire',
        s_all_off, None,
        [0],
        [0],
        np.array([(3 / 7), (4 / 7)]).reshape(2, 1, 1, order="F")
    ), (
        'cause_repertoire',
        s_all_off, None,
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
        s, None,
        [0],
        [0],
        np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        s, None,
        [0, 1],
        [0, 2],
        np.array([0.0, 0.0, 0.5, 0.5]).reshape(2, 1, 2, order="F")
    ), (
        'effect_repertoire',
        s, None,
        [1],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'effect_repertoire',
        s, None,
        [],
        [1],
        np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
    ), (
        'effect_repertoire',
        s, None,
        [2],
        [],
        np.array([1])
    ), (
        'effect_repertoire',
        s, None,
        [],
        [0],
        np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        s, None,
        [0],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'effect_repertoire',
        s, None,
        [1, 2],
        [0],
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        s, None,
        [1],
        [],
        np.array([1])
    ),
        # }}}
        # Full network, with cut {{{
        # --------------------------
    (
        'effect_repertoire',
        s, ((0, 2), 1),
        [0],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
    ), (
        'effect_repertoire',
        s, ((0, 2), 1),
        [0, 1, 2],
        [0, 2],
        np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2, order="F")
    ),
        # }}}
        # Subset, with cut {{{
        # --------------------
    (
        'effect_repertoire',
        subsys_n1n2, (1, 2),
        [1],
        [1, 2],
        np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2, order="F")
    ), (
        'effect_repertoire',
        subsys_n1n2, (1, 2),
        [],
        [1],
        np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
    ), (
        'effect_repertoire',
        subsys_n1n2, (1, 2),
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
        s_all_a_just_on, None,
        [0],
        [0],
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        s_all_a_just_on, None,
        [],
        [0],
        # No matter the state of the purview {m0}, the probability it will
        # be on in the next timestep is 1/8
        np.array([0.875, 0.125]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        s_all_a_just_on, None,
        [1],
        [0, 1, 2],
        np.array([1., 0., 0., 0., 0., 0., 0., 0.]).reshape(2,2,2)
    ), (
        'effect_repertoire',
        s_all_a_just_on, None,
        [1],
        [0, 2],
        np.array([1.0, 0.0, 0.0, 0.0]).reshape(2, 1, 2, order="F")
    ),
        # }}}
        # State: all nodes off {{{
        # ------------------------
    (
        'effect_repertoire',
        s_all_off, None,
        [0],
        [0],
        np.array([0.75, 0.25]).reshape(2, 1, 1, order="F")
    ), (
        'effect_repertoire',
        s_all_off, None,
        [0],
        [0, 1, 2],
        np.array([0.75, 0., 0., 0., 0.25, 0., 0., 0.]).reshape(2,2,2)
    )
        # }}}
    # }}}
# }}}
]
parameter_string = "function,subsystem,cut,mechanism,purview,expected"


@pytest.mark.parametrize(parameter_string, scenarios)
def test_cause_and_effect_repertoire(function, subsystem, cut, mechanism,
                                     purview, expected):
    """Test ``effect_repertoire`` or ``cause_repertoire``."""

    print("\nTesting " + function + " with subsystem \n" + str(subsystem))

    # Set up testing parameters from scenario
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mechanism = tuple(subsystem.network.nodes[index] for index in mechanism)
    purview = tuple(subsystem.network.nodes[index] for index in purview)
    compute_repertoire = getattr(subsystem, function)
    if cut:
        severed = cut[0]
        intact = cut[1]
        # Convert single nodes to singleton tuples
        if not isinstance(severed, type(())):
            severed = (severed,)
        if not isinstance(intact, type(())):
            intact = (intact,)
        severed = tuple(subsystem.network.nodes[index] for index in severed)
        intact = tuple(subsystem.network.nodes[index] for index in intact)
        cut = Cut(severed, intact)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    result = compute_repertoire(mechanism, purview, cut)

    print("\nMechanism:".rjust(12), mechanism, "\nPurview:".rjust(12), purview,
          "\nCut:".rjust(12), cut, "\n")

    print('-'*40, "Result:", result, "\nResult Shape:", result.shape,
            '-'*40, "Expected:", expected, "\nExpected Shape:",
            expected.shape, '-'*40, sep="\n")

    assert np.array_equal(result, expected)


# Test validation
def test_cause_and_effect_repertoire_validation(s):
    with pytest.raises(ValueError):
        s.cause_repertoire((0,), (1,), s.null_cut)
    with pytest.raises(ValueError):
        s.effect_repertoire((0,1), (2,), s.null_cut)
    with pytest.raises(ValueError):
        s.effect_repertoire(0, (2), s.null_cut)


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
