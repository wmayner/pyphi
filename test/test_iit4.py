"""
Golden end-to-end tests for IIT 4.0 phi_structure computation.

These tests validate complete phi_structure computations against JSON
fixtures stored in test/data/phi_structure/. They ensure that the IIT 4.0
implementation produces results matching historical validated outputs.

Test Networks:
- basic: 3-node basic network (OR, COPY, XOR gates)
- basic_noisy_selfloop: Basic network with noise and self-loops
- fig4: Example from IIT 4.0 paper (Figure 4)
- fig5a: Example from IIT 4.0 paper (Figure 5a)
- fig5b: Example from IIT 4.0 paper (Figure 5b)
- grid3: 3-node grid topology
- residue: Residue network example
- rule110: Rule 110 cellular automaton
- rule154: Rule 154 cellular automaton
- xor: XOR gate configuration

The phi_structure computation is the core of IIT 4.0, producing:
- Distinctions: Irreducible mechanisms (concepts)
- Relations: Dependencies between distinctions
- System-level integrated information structure

Test Approach:
Each test compares the complete phi_structure result against a JSON fixture.
This is a comprehensive validation but is brittle to serialization changes.
For more robust component-level tests, see test_iit4_robust.py.

Theoretical Basis:
These examples are based on the IIT 4.0 formalism described in:
Albantakis L, Barbosa L, Findlay G, Grasso M, ... Tononi G. (2023)
"Integrated information theory (IIT) 4.0: formulating the properties of
phenomenal existence in physical terms."
PLoS Computational Biology 19(10): e1011465.
https://doi.org/10.1371/journal.pcbi.1011465

The fig4, fig5a, and fig5b examples correspond to figures in this paper.
"""

from pathlib import Path

import pytest

from pyphi import jsonify
from pyphi.examples import EXAMPLES
from pyphi.formalism import iit4 as new_big_phi

EXAMPLE_NAMES = [
    "basic",
    "basic_noisy_selfloop",
    "fig4",
    "fig5a",
    "fig5b",
    "grid3",
    "residue",
    "rule110",
    pytest.param("rule154", marks=pytest.mark.slow),
    "xor",
]

DATA_PATH = Path("test/data/phi_structure")


@pytest.mark.parametrize("example_name", EXAMPLE_NAMES)
def test(example_name):
    """Test phi_structure computation against golden JSON fixture.

    This parametrized test validates phi_structure results for all
    example networks. Each example is compared against its JSON fixture,
    ensuring complete computational correctness.

    Args:
        example_name: Name of example network (see EXAMPLE_NAMES)

    What's tested:
    - Full phi_structure computation pipeline
    - Distinction finding (irreducible mechanisms)
    - Relation computation (dependencies between distinctions)
    - Complete data structure serialization/deserialization

    If this fails:
    - Check if computation algorithm changed
    - Verify example network definition unchanged
    - For fig4: Cross-reference with IIT 4.0 paper
    - Consider if JSON serialization format changed

    Note: rule154 is marked as slow due to computational expense (11 distinctions).
    """
    subsystem = EXAMPLES["subsystem"][example_name]()
    actual = new_big_phi.phi_structure(subsystem)
    expected = load_expected(example_name)
    assert actual == expected


def load_expected(example_name):
    """Load expected phi_structure result from JSON fixture.

    Args:
        example_name: Name of example network

    Returns:
        PhiStructure object deserialized from JSON
    """
    with open(DATA_PATH / f"{example_name}.json") as f:
        return jsonify.load(f)
