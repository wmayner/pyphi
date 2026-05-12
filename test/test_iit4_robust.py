"""
Robust component-level tests for IIT 4.0 phi_structure computation.

These tests complement the golden tests in test_iit4.py by providing:
1. Component-level validation (distinctions, relations)
2. Better failure diagnostics
3. Serialization-independent checks
4. Fast-running regression tests

Unlike the golden tests which compare entire phi_structure objects against
JSON fixtures, these tests check individual components. This makes them more
robust to data model refactoring.

Expected values are extracted from the same JSON fixtures used by golden
tests, ensuring consistency.
"""

import pytest

from pyphi import config
from pyphi.examples import EXAMPLES
from pyphi.formalism import iit4 as new_big_phi
from pyphi.metrics.distribution import resolve_mechanism_measure
from pyphi.metrics.distribution import resolve_system_measure


def _ces_kwargs():
    """Measure kwargs resolved from current config.

    Module-level ``new_big_phi.ces`` requires explicit measures;
    these tests use defaults so resolve from ``config`` at call time.
    """
    return {
        "system_measure": resolve_system_measure(
            config.formalism.iit.system_phi_measure
        ),
        "specification_measure": resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    }


# Expected values extracted from JSON fixtures in test/data/phi_structure/
# These serve as golden references for component counts.
EXPECTED_PHI_STRUCTURE = {
    "basic": {"num_distinctions": 2, "has_distinctions": True},
    "basic_noisy_selfloop": {"num_distinctions": 2, "has_distinctions": True},
    "fig4": {"num_distinctions": 4, "has_distinctions": True},
    "fig5a": {"num_distinctions": 2, "has_distinctions": True},
    "fig5b": {"num_distinctions": 1, "has_distinctions": True},
    "grid3": {"num_distinctions": 7, "has_distinctions": True},
    "residue": {"num_distinctions": 0, "has_distinctions": False},
    "rule110": {"num_distinctions": 0, "has_distinctions": False},
    "rule154": {"num_distinctions": 11, "has_distinctions": True},
    "xor": {"num_distinctions": 4, "has_distinctions": True},
}


# ============================================================================
# Distinction Count Tests
# ============================================================================


class TestDistinctionCounts:
    """Test that phi_structure produces expected number of distinctions.

    Distinctions (irreducible mechanisms) are fundamental to IIT 4.0.
    These tests verify that the distinction-finding algorithm produces
    the expected number of distinctions for each test substrate.
    """

    def test_phi_structure_basic_distinction_count(self):
        """Basic example produces expected number of distinctions.

        Substrate: Basic 3-node substrate (OR, COPY, XOR gates)
        Expected: 2 distinctions

        If this fails, the distinction-finding algorithm may have changed
        or the basic example substrate definition was modified.
        """
        system = EXAMPLES["system"]["basic"]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        expected_count = EXPECTED_PHI_STRUCTURE["basic"]["num_distinctions"]
        actual_count = len(result.distinctions) if hasattr(result, "distinctions") else 0

        assert actual_count == expected_count, (
            f"Basic example distinction count changed:\n"
            f"  Expected: {expected_count}\n"
            f"  Got:      {actual_count}"
        )

    def test_phi_structure_basic_noisy_selfloop_distinction_count(self):
        """Basic noisy selfloop example produces expected distinctions.

        Substrate: Basic substrate with noise and self-loops
        Expected: 2 distinctions

        Tests distinction finding with stochastic transitions.
        """
        system = EXAMPLES["system"]["basic_noisy_selfloop"]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        expected_count = EXPECTED_PHI_STRUCTURE["basic_noisy_selfloop"][
            "num_distinctions"
        ]
        actual_count = len(result.distinctions) if hasattr(result, "distinctions") else 0

        assert actual_count == expected_count, (
            f"Basic noisy selfloop distinction count changed:\n"
            f"  Expected: {expected_count}\n"
            f"  Got:      {actual_count}"
        )

    def test_phi_structure_fig4_distinction_count(self):
        """Figure 4 example produces expected number of distinctions.

        Substrate: Example from published IIT 4.0 paper (Figure 4)
        Expected: 4 distinctions

        This is a published reference example, so the expected count
        is validated against the paper.
        """
        system = EXAMPLES["system"]["fig4"]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        expected_count = EXPECTED_PHI_STRUCTURE["fig4"]["num_distinctions"]
        actual_count = len(result.distinctions) if hasattr(result, "distinctions") else 0

        assert actual_count == expected_count, (
            f"Figure 4 example distinction count changed:\n"
            f"  Expected: {expected_count}\n"
            f"  Got:      {actual_count}\n"
            f"This is a published example - check against IIT 4.0 paper"
        )

    def test_phi_structure_grid3_distinction_count(self):
        """Grid topology example produces expected distinctions.

        Substrate: 3-node grid topology
        Expected: 7 distinctions

        Tests distinction finding on grid-structured substrates.
        """
        system = EXAMPLES["system"]["grid3"]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        expected_count = EXPECTED_PHI_STRUCTURE["grid3"]["num_distinctions"]
        actual_count = len(result.distinctions) if hasattr(result, "distinctions") else 0

        assert actual_count == expected_count, (
            f"Grid3 example distinction count changed:\n"
            f"  Expected: {expected_count}\n"
            f"  Got:      {actual_count}"
        )

    def test_phi_structure_xor_distinction_count(self):
        """XOR substrate example produces expected distinctions.

        Substrate: XOR gate configuration
        Expected: 4 distinctions

        Tests distinction finding with XOR logic gates.
        """
        system = EXAMPLES["system"]["xor"]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        expected_count = EXPECTED_PHI_STRUCTURE["xor"]["num_distinctions"]
        actual_count = len(result.distinctions) if hasattr(result, "distinctions") else 0

        assert actual_count == expected_count, (
            f"XOR example distinction count changed:\n"
            f"  Expected: {expected_count}\n"
            f"  Got:      {actual_count}"
        )


# ============================================================================
# Structure Existence Tests
# ============================================================================


class TestCauseEffectStructureComponents:
    """Test that phi_structure results have expected components.

    These tests verify that phi_structure objects contain the required
    attributes and that those attributes are properly populated.
    """

    @pytest.mark.parametrize(
        "example_name",
        [
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
        ],
    )
    def test_phi_structure_has_distinctions_attribute(self, example_name):
        """Phi structure results should have distinctions attribute.

        All phi_structure results should have a 'distinctions' attribute,
        even if it's empty or None for reducible systems.

        If this fails, the CauseEffectStructure data model was changed.

        Note: rule154 is marked as slow due to computational expense (11 distinctions).
        """
        system = EXAMPLES["system"][example_name]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        assert hasattr(result, "distinctions"), (
            f"CauseEffectStructure for '{example_name}' missing 'distinctions' attribute"
        )

    @pytest.mark.parametrize(
        "example_name",
        [
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
        ],
    )
    def test_phi_structure_has_relations_attribute(self, example_name):
        """Phi structure results should have relations attribute.

        All phi_structure results should have a 'relations' attribute
        to store dependencies between distinctions.

        If this fails, the CauseEffectStructure data model was changed.

        Note: rule154 is marked as slow due to computational expense (11 distinctions).
        """
        system = EXAMPLES["system"][example_name]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        assert hasattr(result, "relations"), (
            f"CauseEffectStructure for '{example_name}' missing 'relations' attribute"
        )

    @pytest.mark.parametrize(
        "example_name",
        [
            "basic",
            "basic_noisy_selfloop",
            "fig4",
            "fig5a",
            "fig5b",
            "grid3",
            pytest.param("rule154", marks=pytest.mark.slow),
            "xor",
        ],
    )
    def test_phi_structure_distinctions_are_non_empty(self, example_name):
        """Non-trivial systems should have non-empty distinctions.

        These examples are known to have irreducible distinctions.
        If the distinctions list is empty, distinction finding failed.

        Note: rule154 is marked as slow due to computational expense (11 distinctions).
        """
        system = EXAMPLES["system"][example_name]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        assert hasattr(result, "distinctions"), (
            "CauseEffectStructure missing distinctions attribute"
        )

        distinctions = result.distinctions
        assert distinctions is not None, (
            f"CauseEffectStructure for '{example_name}' has None distinctions "
            f"(expected non-empty)"
        )

        # Get count depending on structure type
        if hasattr(distinctions, "__len__"):
            count = len(distinctions)
        elif hasattr(distinctions, "concepts"):
            count = len(distinctions.concepts)
        else:
            count = 0

        assert count > 0, (
            f"CauseEffectStructure for '{example_name}' has zero distinctions "
            f"(expected at least one)"
        )


# ============================================================================
# Distinction Properties Tests
# ============================================================================


class TestDistinctionProperties:
    """Test properties of individual distinctions.

    These tests verify that distinctions themselves are well-formed
    and have the expected attributes.
    """

    def test_distinctions_have_mechanism_attribute(self):
        """Distinctions should have mechanism attribute.

        Each distinction represents an irreducible mechanism, so it
        must have a 'mechanism' attribute identifying the nodes.
        """
        system = EXAMPLES["system"]["basic"]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        assert hasattr(result, "distinctions"), "No distinctions attribute"

        # Get the concepts (distinctions)
        if hasattr(result.distinctions, "concepts"):
            concepts = result.distinctions.concepts
        else:
            concepts = result.distinctions

        # Check each distinction has a mechanism
        for i, distinction in enumerate(concepts):
            assert hasattr(distinction, "mechanism"), (
                f"Distinction {i} missing 'mechanism' attribute"
            )

    def test_distinctions_mechanisms_are_within_system(self):
        """Distinction mechanisms should be subsets of system nodes.

        Each distinction's mechanism must be composed of nodes that
        exist in the system. This is a fundamental requirement.

        If this fails, distinction finding is assigning mechanisms
        outside the system boundaries.
        """
        system = EXAMPLES["system"]["basic"]()
        result = new_big_phi.ces(system, **_ces_kwargs())

        system_nodes = set(system.node_indices)

        # Get the concepts (distinctions)
        if hasattr(result.distinctions, "concepts"):
            concepts = result.distinctions.concepts
        else:
            concepts = result.distinctions

        for distinction in concepts:
            mechanism = distinction.mechanism
            mechanism_nodes = set(mechanism)

            assert mechanism_nodes.issubset(system_nodes), (
                f"Distinction mechanism {mechanism} not subset of "
                f"system nodes {system_nodes}"
            )


# ============================================================================
# Cross-Example Consistency Tests
# ============================================================================


class TestCrossExampleConsistency:
    """Test consistency properties across multiple examples.

    These tests verify that phi_structure computation is consistent
    across different substrates and doesn't have example-specific bugs.
    """

    @pytest.mark.parametrize(
        "example_name",
        [
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
        ],
    )
    def test_phi_structure_is_deterministic(self, example_name):
        """Phi structure computation should be deterministic.

        Running phi_structure twice on the same system should
        produce identical results (unless there's intentional randomness).

        This test catches non-deterministic bugs in computation.

        Note: rule154 is marked as slow due to computational expense (11 distinctions).
        """
        system = EXAMPLES["system"][example_name]()

        # Compute twice
        result1 = new_big_phi.ces(system, **_ces_kwargs())
        result2 = new_big_phi.ces(system, **_ces_kwargs())

        # Should get same number of distinctions
        count1 = len(result1.distinctions) if hasattr(result1, "distinctions") else 0
        count2 = len(result2.distinctions) if hasattr(result2, "distinctions") else 0

        assert count1 == count2, (
            f"Non-deterministic distinction count for '{example_name}':\n"
            f"  First run:  {count1}\n"
            f"  Second run: {count2}\n"
            f"This indicates non-deterministic behavior in phi_structure"
        )

        # Results should be equal
        assert result1 == result2, (
            f"Non-deterministic phi_structure for '{example_name}'\n"
            f"Two runs produced different results"
        )
