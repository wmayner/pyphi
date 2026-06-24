import numpy as np
import pytest

from pyphi import Direction
from pyphi import System
from pyphi.models import DirectedBipartition

from . import example_substrates

# Get example substrates
standard = example_substrates.standard()

# Get example systems
standard_system = example_substrates.s()
simple_all_off = example_substrates.simple_subsys_all_off()
simple_a_just_on = example_substrates.simple_subsys_all_a_just_on()


full = tuple(range(standard.size))


# Set up test scenarios
# =====================
# Scenario structure:
# (
#     function to test,
#     system,
#     mechanism,
#     purview,
#     expected result
# )
scenarios = [
    # Cause repertoire {{{
    # ====================
    # Default Matlab substrate {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Full substrate, no cut {{{
    # ------------------------
    (
        "cause_repertoire",
        standard_system,
        [0],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F"),
    ),
    (
        "cause_repertoire",
        standard_system,
        [0],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F"),
    ),
    (
        "cause_repertoire",
        standard_system,
        [0, 1],
        [0, 2],
        np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2, order="F"),
    ),
    (
        "cause_repertoire",
        standard_system,
        [1],
        [2],
        np.array([1.0, 0.0]).reshape(1, 1, 2, order="F"),
    ),
    (
        "cause_repertoire",
        standard_system,
        [],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F"),
    ),
    ("cause_repertoire", standard_system, [1], [], np.array([1])),
    # }}}
    # Full substrate, with cut {{{
    # --------------------------
    (
        "cause_repertoire",
        System(
            standard,
            standard_system.state,
            full,
            partition=DirectedBipartition(Direction.EFFECT, (2,), (0, 1)),
        ),
        [0],
        [1],
        np.array([1 / 3, 2 / 3]).reshape(1, 2, 1, order="F"),
    ),
    # }}}
    # Subset, with cut {{{
    # --------------------
    (
        "cause_repertoire",
        System(
            standard,
            standard_system.state,
            (1, 2),
            partition=DirectedBipartition(Direction.EFFECT, (1,), (2,)),
        ),
        [2],
        [1, 2],
        np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2, order="F"),
    ),
    (
        "cause_repertoire",
        System(
            standard,
            standard_system.state,
            (1, 2),
            partition=DirectedBipartition(Direction.EFFECT, (1,), (2,)),
        ),
        [2],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F"),
    ),
    (
        "cause_repertoire",
        System(
            standard,
            standard_system.state,
            (0, 2),
            partition=DirectedBipartition(Direction.EFFECT, (0,), (2,)),
        ),
        [2],
        [0],
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F"),
    ),
    # }}}
    # }}}
    # Simple 'AND' substrate {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # State: 'A' just turned on {{{
    # -----------------------------
    (
        "cause_repertoire",
        simple_a_just_on,
        [0],
        [0],
        # Cause repertoire is maximally selective; the previous state must have
        # been {0,1,1}, so `expected[(0,1,1)]` should be 1 and everything else
        # should be 0
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F"),
    ),
    (
        "cause_repertoire",
        simple_a_just_on,
        [],
        [0],
        # No matter the state of the purview (m0), the probability it will be
        # on in the next timestep is 1/8
        np.array([0.5, 0.5]).reshape(2, 1, 1, order="F"),
    ),
    ("cause_repertoire", simple_a_just_on, [1], [0, 1, 2], np.ones((2, 2, 2)) / 8),
    (
        "cause_repertoire",
        simple_a_just_on,
        [0, 1],
        [0, 2],
        np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2, order="F"),
    ),
    # }}}
    # State: all nodes off {{{
    # ------------------------
    (
        "cause_repertoire",
        simple_all_off,
        [0],
        [0],
        np.array([(3 / 7), (4 / 7)]).reshape(2, 1, 1, order="F"),
    ),
    (
        "cause_repertoire",
        simple_all_off,
        [0],
        [0, 1, 2],
        # Cause repertoire is minimally selective; only {0,1,1} is ruled out,
        # so probability density should be uniformly distributed among all
        # states not including {0,1,1} when purview is whole substrate
        np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]).reshape(2, 2, 2) / 7,
    ),
    # }}}
    # }}}
    # }}}
    # Effect repertoire {{{
    # =====================
    # Default Matlab substrate {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Full substrate, no cut {{{
    # ------------------------
    (
        "effect_repertoire",
        standard_system,
        [0],
        [0],
        np.array([0.25, 0.75]).reshape(2, 1, 1, order="F"),
    ),
    (
        "effect_repertoire",
        standard_system,
        [0, 1],
        [0, 2],
        np.array([0.0, 0.0, 0.5, 0.5]).reshape(2, 1, 2, order="F"),
    ),
    (
        "effect_repertoire",
        standard_system,
        [1],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F"),
    ),
    (
        "effect_repertoire",
        standard_system,
        [],
        [1],
        np.array([0.5, 0.5]).reshape(1, 2, 1, order="F"),
    ),
    ("effect_repertoire", standard_system, [2], [], np.array([1])),
    (
        "effect_repertoire",
        standard_system,
        [],
        [0],
        np.array([0.25, 0.75]).reshape(2, 1, 1, order="F"),
    ),
    (
        "effect_repertoire",
        standard_system,
        [0],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F"),
    ),
    (
        "effect_repertoire",
        standard_system,
        [1, 2],
        [0],
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F"),
    ),
    ("effect_repertoire", standard_system, [1], [], np.array([1])),
    # }}}
    # Full substrate, with cut {{{
    # --------------------------
    (
        "effect_repertoire",
        System(
            standard,
            standard_system.state,
            full,
            partition=DirectedBipartition(Direction.EFFECT, (0, 2), (1,)),
        ),
        [0],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F"),
    ),
    (
        "effect_repertoire",
        System(
            standard,
            standard_system.state,
            full,
            partition=DirectedBipartition(Direction.EFFECT, (0, 2), (1,)),
        ),
        [0, 1, 2],
        [0, 2],
        np.array([0.0, 0.0, 1.0, 0.0]).reshape(2, 1, 2, order="F"),
    ),
    # }}}
    # Subset, with cut {{{
    # --------------------
    (
        "effect_repertoire",
        System(
            standard,
            standard_system.state,
            (1, 2),
            partition=DirectedBipartition(Direction.EFFECT, (1,), (2,)),
        ),
        [1],
        [1, 2],
        np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 2, 2, order="F"),
    ),
    (
        "effect_repertoire",
        System(
            standard,
            standard_system.state,
            (1, 2),
            partition=DirectedBipartition(Direction.EFFECT, (1,), (2,)),
        ),
        [],
        [1],
        np.array([0.5, 0.5]).reshape(1, 2, 1, order="F"),
    ),
    (
        "effect_repertoire",
        System(
            standard,
            standard_system.state,
            (1, 2),
            partition=DirectedBipartition(Direction.EFFECT, (1,), (2,)),
        ),
        [1],
        [2],
        np.array([0.5, 0.5]).reshape(1, 1, 2, order="F"),
    ),
    # }}}
    # }}}
    # Simple 'AND' substrate {{{
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    # State: 'A' just turned on {{{
    # -----------------------------
    (
        "effect_repertoire",
        simple_a_just_on,
        [0],
        [0],
        np.array([1.0, 0.0]).reshape(2, 1, 1, order="F"),
    ),
    (
        "effect_repertoire",
        simple_a_just_on,
        [],
        [0],
        # No matter the state of the purview {m0}, the probability it will
        # be on in the next timestep is 1/8
        np.array([0.875, 0.125]).reshape(2, 1, 1, order="F"),
    ),
    (
        "effect_repertoire",
        simple_a_just_on,
        [1],
        [0, 1, 2],
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(2, 2, 2),
    ),
    (
        "effect_repertoire",
        simple_a_just_on,
        [1],
        [0, 2],
        np.array([1.0, 0.0, 0.0, 0.0]).reshape(2, 1, 2, order="F"),
    ),
    # }}}
    # State: all nodes off {{{
    # ------------------------
    (
        "effect_repertoire",
        simple_all_off,
        [0],
        [0],
        np.array([0.75, 0.25]).reshape(2, 1, 1, order="F"),
    ),
    (
        "effect_repertoire",
        simple_all_off,
        [0],
        [0, 1, 2],
        np.array([0.75, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0]).reshape(2, 2, 2),
    ),
    # }}}
    # }}}
    # }}}
]


parameter_string = "function,system,mechanism,purview,expected"


@pytest.mark.parametrize(parameter_string, scenarios)
def test_cause_and_effect_repertoire(function, system, mechanism, purview, expected):
    """Test ``effect_repertoire`` or ``cause_repertoire``."""

    print("\nTesting " + function + " with system \n" + str(system))

    # Set up testing parameters from scenario
    compute_repertoire = getattr(system, function)
    mechanism = tuple(mechanism)
    purview = tuple(purview)

    result = compute_repertoire(mechanism, purview)

    print(
        "\nMechanism:".rjust(12),
        mechanism,
        "\nPurview:".rjust(12),
        purview,
        "\nCut:".rjust(12),
        system.partition,
        "\n",
    )

    print(
        "-" * 40,
        "Result:",
        result,
        "\nResult Shape:",
        result.shape,
        "-" * 40,
        "Expected:",
        expected,
        "\nExpected Shape:",
        expected.shape,
        "-" * 40,
        sep="\n",
    )

    assert np.array_equal(result, expected)


def test_repertoire_wrong_direction_error(s):
    with pytest.raises(ValueError):
        s.repertoire(Direction.BIDIRECTIONAL, (0,), (0, 1))


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
