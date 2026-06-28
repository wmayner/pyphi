"""Value-based tests for pyphi.substrate_generator."""

import numpy as np
import pytest

from pyphi import Substrate
from pyphi.substrate_generator import UNIT_FUNCTIONS
from pyphi.substrate_generator import build_substrate
from pyphi.substrate_generator import build_tpm
from pyphi.substrate_generator import ising
from pyphi.substrate_generator import unit_functions

# 3-node all-to-all weights (no self-loops); element 0 has two inputs (nodes 1, 2),
# each weight 1, so total_weighted_input(0, W, state) == state[1] + state[2].
W3 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.parametrize(
    "s1,s2,expected_or,expected_and,expected_parity",
    [
        (0, 0, False, False, False),  # twi = 0
        (1, 0, True, False, True),  # twi = 1
        (0, 1, True, False, True),  # twi = 1
        (1, 1, True, True, False),  # twi = 2 (>=2 inputs; even parity)
    ],
)
def test_logical_unit_functions(s1, s2, expected_or, expected_and, expected_parity):
    state = (0, s1, s2)  # state[0] is irrelevant to element 0 (W[0,0] == 0)
    assert unit_functions.logical_or_function(0, W3, state) == expected_or
    assert unit_functions.logical_and_function(0, W3, state) == expected_and
    assert unit_functions.logical_parity_function(0, W3, state) == expected_parity
    # Negations are the logical complements.
    assert unit_functions.logical_nor_function(0, W3, state) == (not expected_or)
    assert unit_functions.logical_nand_function(0, W3, state) == (not expected_and)
    assert unit_functions.logical_nparity_function(0, W3, state) == (not expected_parity)


def test_naka_rushton():
    # x = twi**exponent; return x / (x + threshold). exponent=2, threshold=1.
    # state (0,1,1): twi=2 -> x=4 -> 4/5 = 0.8
    assert unit_functions.naka_rushton(
        0, W3, (0, 1, 1), exponent=2.0, threshold=1.0
    ) == pytest.approx(0.8)
    # state (0,1,0): twi=1 -> x=1 -> 1/2 = 0.5
    assert unit_functions.naka_rushton(
        0, W3, (0, 1, 0), exponent=2.0, threshold=1.0
    ) == pytest.approx(0.5)


def test_gaussian():
    # gaussian binary2spin's the state first (0 -> -1), then gauss(twi, mu, sigma).
    # state (0,1,1) -> spin (-1,1,1); twi(0) = -1*0 + 1*1 + 1*1 = 2.
    # gauss(2, mu=0, sigma=0.5) = exp(-0.5 * (2/0.5)**2) = exp(-8).
    assert unit_functions.gaussian(0, W3, (0, 1, 1), mu=0.0, sigma=0.5) == pytest.approx(
        np.exp(-8.0)
    )


def test_ising_energy_and_probability():
    # energy == total_weighted_input on the (already-spin) state.
    assert ising.energy(0, W3, (-1, 1, 1)) == pytest.approx(2.0)
    # probability binary2spin's first: state (0,1,1) -> spin (-1,1,1); E=2;
    # sigmoid(E, T=1, field=0) = 1 / (1 + exp(-2)).
    assert ising.probability(
        0, W3, (0, 1, 1), temperature=1.0, field=0.0
    ) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_unit_functions_registry_keys():
    # The original weighted-threshold logical gates and core unit functions.
    base = {
        "ising",
        "boolean",
        "gaussian",
        "naka_rushton",
        "or",
        "and",
        "parity",
        "nor",
        "nand",
        "nparity",
    }
    assert base <= set(UNIT_FUNCTIONS)
    # The ported substrate_modeler mechanisms are also registered (under names
    # not already taken, so "and"/"or" keep their weighted-threshold meaning).
    assert {"sigmoid", "resonnator", "sor", "gabor", "mismatch_corrector"} <= set(
        UNIT_FUNCTIONS
    )


# 2-node ring W[[0,1],[1,0]]: total_weighted_input(0, state)=state[1];
# total_weighted_input(1, state)=state[0].
W2 = np.array([[0, 1], [1, 0]])


def test_build_tpm_parity_exact():
    tpm = build_tpm("parity", W2)
    # tpm[s0, s1, element]; parity(elem) is True iff its single input == 1.
    expected = np.array(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ],  # (0,0)->[0,0]  (0,1)->[parity(s1=1)=1, parity(s0=0)=0]
            [[0.0, 1.0], [1.0, 1.0]],  # (1,0)->[0,1]  (1,1)->[1,1]
        ]
    )
    assert tpm.shape == (2, 2, 2)
    assert np.array_equal(tpm, expected)


def test_build_tpm_rejects_non_square_weights():
    with pytest.raises(ValueError, match="square"):
        build_tpm("or", np.array([[0, 1, 1], [1, 0, 1]]))


def test_build_tpm_rejects_mismatched_unit_function_count():
    with pytest.raises(ValueError, match="match"):
        build_tpm(["or", "and", "or"], W2)  # 3 funcs for a 2-node weight matrix


def test_build_substrate_returns_substrate_with_correct_cm():
    sub = build_substrate("parity", W2)
    assert isinstance(sub, Substrate)
    assert np.array_equal(np.asarray(sub.cm), np.array([[0, 1], [1, 0]]))
    assert [str(label) for label in sub.node_labels] == ["A", "B"]
