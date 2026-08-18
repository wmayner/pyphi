import numpy as np
import pytest

from pyphi import examples
from pyphi.actual import Transition
from pyphi.substrate import Substrate
from pyphi.system import System


def _example_items(kind):
    return sorted(examples.EXAMPLES[kind].items())


def test_examples_registry_contains_expected_categories():
    expected = {"substrate", "system", "tpm", "transition"}
    assert expected.issubset(examples.EXAMPLES.keys())


def test_register_example_rejects_unknown_category():
    """A function whose name does not end in a known category raises, rather
    than registering under a spurious one.
    """
    with pytest.raises(ValueError, match="must end in one of"):

        @examples.register_example
        def something_bogus():  # 'bogus' is not a known category
            return None


@pytest.mark.parametrize("name, func", _example_items("substrate"))
def test_example_substrates_construct(name, func):
    substrate = func()
    assert isinstance(substrate, Substrate)


@pytest.mark.parametrize("name, func", _example_items("system"))
def test_example_systems_construct(name, func):
    system = func()
    assert isinstance(system, System)


@pytest.mark.parametrize("name, func", _example_items("tpm"))
def test_example_tpms_construct(name, func):
    tpm = func()
    assert isinstance(tpm, np.ndarray)
    assert tpm.ndim == 2
    assert tpm.shape[0] == tpm.shape[1]


@pytest.mark.parametrize("name, func", _example_items("transition"))
def test_example_transitions_construct(name, func):
    transition = func()
    assert isinstance(transition, Transition)


def test_frog_species_build_under_default_config():
    """Each frog species builds a substrate and transition of the expected
    size, with no special configuration required to construct them.
    """
    sizes = {"F1": 8, "F2": 7, "F3": 8}
    for species, size in sizes.items():
        substrate = examples.frog_substrate(species)
        assert isinstance(substrate, Substrate)
        assert substrate.size == size
        assert isinstance(examples.frog_transition(species), Transition)

    with pytest.raises(ValueError, match="unknown frog species"):
        examples.frog_substrate("bogus")


def test_frog_accounts_have_composite_causes():
    """The frogs' actual-causation accounts contain composite (multi-unit)
    causes — most in F3 (which has the composite super-bug detector CC) and
    fewest in the reduced F1 — reproducing the point of Grasso et al. (2021).
    """
    from dataclasses import replace

    from pyphi import actual
    from pyphi import config
    from pyphi import iit3

    composite_counts = {}
    with config.override(
        iit=replace(
            iit3["iit"],
            mechanism_partition_scheme="WEDGE_TRIPARTITION",
            mechanism_phi_measure="AID",
        ),
        validate_system_states=False,
        alpha_measure="WPMI",
        progress_bars=False,
    ):
        for species in ("F1", "F2", "F3"):
            account = actual.account(examples.frog_transition(species))
            composite_counts[species] = sum(
                1 for link in account if len(link.purview) >= 2
            )

    assert all(count >= 1 for count in composite_counts.values())
    assert composite_counts["F3"] > composite_counts["F1"]


def test_propagation_delay_d_gate_is_xor():
    """Unit D copies XOR(B, F) from the previous state, for every state."""
    import itertools

    substrate = examples.propagation_delay_substrate()
    tpm = substrate.tpm.to_array()
    for previous in itertools.product((0, 1), repeat=9):
        expected = (previous[1] == 1) ^ (previous[5] == 1)
        assert tpm[previous][3][1] == expected, previous


def test_basic_substrate_honors_passed_cm():
    full = np.ones((3, 3), dtype=int)
    substrate = examples.basic_substrate(cm=full)
    assert np.array_equal(substrate.cm, full)
    # The default connectivity is unchanged.
    default = examples.basic_substrate()
    assert np.array_equal(default.cm, np.array([[0, 0, 1], [1, 0, 1], [1, 1, 0]]))
    # An inconsistent matrix is rejected rather than silently replaced.
    cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    with pytest.raises(ValueError, match="under-specified"):
        examples.basic_substrate(cm=cycle)


def test_fig5b_gates_match_the_2014_paper():
    """Figure 5B of the 2014 paper: A = NULL, B = AND(A, C), C = OR(A, B)."""
    substrate = examples.fig5b_substrate()
    for idx in range(8):
        state = (idx & 1, (idx >> 1) & 1, (idx >> 2) & 1)
        a, b, c = state
        assert substrate.factored_tpm.factor(0)[(*state, 1)] == 1.0  # NULL: always ON
        assert substrate.factored_tpm.factor(1)[(*state, 1)] == float(a and c)
        assert substrate.factored_tpm.factor(2)[(*state, 1)] == float(a or b)


def test_differentiation_macro_tpm_epsilon_zero_limit():
    """At epsilon = 0 the three micro states in macro state 0 have identical
    dynamics, so the coarse-grained probability must equal p^2 exactly."""
    p = 0.9
    macro = examples.differentiation_macro_tpm(p, 0.0)
    assert macro[0][0] == pytest.approx(p * p)
    assert macro[1][0] == pytest.approx((1 - p) * (1 - p))
    # General form: p^2 + (2/3) p epsilon.
    eps = 0.01
    macro = examples.differentiation_macro_tpm(p, eps)
    assert macro[0][0] == pytest.approx(p * p + 2 * p * eps / 3)
