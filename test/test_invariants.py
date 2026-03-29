"""
Invariant tests for PyPhi golden test suite.

These tests validate mathematical and theoretical properties that must
hold regardless of implementation details. They catch logical errors
that might slip past value-based regression tests.

These invariants are based on fundamental IIT properties and should always
hold true, making them excellent regression tests for refactoring.
"""

import pytest

from pyphi import config
from pyphi import new_big_phi
from pyphi.new_big_phi import NullSystemIrreducibilityAnalysis
from pyphi.subsystem import Subsystem

from . import example_networks
from .conftest import skip_if_no_pyemd


class TestPhiInvariants:
    """Test invariants related to phi values.

    These tests verify that phi values satisfy fundamental IIT properties:
    - Phi is always non-negative
    - Empty subsystems have zero phi
    - Reducible/disconnected systems have zero phi
    """

    def test_phi_non_negative(self, s, micro_s, macro_s):
        """Phi must always be non-negative (theoretical requirement).

        IIT Property: Integrated information is a non-negative quantity.
        Any negative phi value indicates a fundamental computation error.
        """
        assert s.sia().phi >= 0
        assert micro_s.sia().phi >= 0
        assert macro_s.sia().phi >= 0

    def test_empty_subsystem_has_zero_phi(self, s_empty):
        """Empty subsystems have no integration (phi=0).

        IIT Property: A system with no elements cannot have integrated
        information. Empty subsystems must return NullSIA with phi=0.
        """
        result = s_empty.sia()
        assert result.phi == 0.0
        assert isinstance(result, NullSystemIrreducibilityAnalysis)

    def test_reducible_systems_have_zero_phi(self, reducible):
        """Disconnected/reducible systems have phi=0.

        IIT Property: Systems that can be partitioned without loss of
        information are reducible and have zero integrated information.
        """
        result = reducible.sia()
        assert result.phi == 0.0
        assert isinstance(result, NullSystemIrreducibilityAnalysis)

    def test_single_node_without_selfloop_has_zero_phi(self, s_single):
        """Single nodes without self-loops have phi=0.

        IIT Property: A single node without causal self-interaction
        cannot specify states beyond itself, thus has no integration.
        """
        result = s_single.sia()
        assert result.phi == 0.0


class TestParallelConsistency:
    """Test that parallel and sequential computation agree.

    Parallelization is purely a performance optimization and must not
    change computational results. These tests verify that the parallel
    execution path produces identical results to sequential execution.
    """

    @pytest.mark.parametrize(
        "subsystem_fixture",
        ["s", "micro_s", "macro_s", "s_noised"],
    )
    def test_sequential_equals_parallel(self, subsystem_fixture, request):
        """Sequential and parallel must produce identical results.

        This tests the fundamental requirement that parallelization
        cannot change results. If this fails, there's a bug in the
        parallel implementation (likely a race condition or shared
        state issue).

        Args:
            subsystem_fixture: Name of subsystem fixture to test
            request: Pytest request object for getting fixture values
        """
        subsystem = request.getfixturevalue(subsystem_fixture)

        # Compute with sequential mode
        with config.override(PARALLEL=False):
            seq_result = subsystem.sia()

        # Compute with parallel mode
        with config.override(PARALLEL=True):
            par_result = subsystem.sia()

        # Results must be exactly equal
        assert seq_result == par_result, (
            f"Parallel and sequential results differ for {subsystem_fixture}:\n"
            f"  Sequential phi: {seq_result.phi}\n"
            f"  Parallel phi:   {par_result.phi}"
        )

        # Also check phi values explicitly for better error messages
        assert seq_result.phi == par_result.phi, (
            f"Phi values differ for {subsystem_fixture}:\n"
            f"  Sequential: {seq_result.phi}\n"
            f"  Parallel:   {par_result.phi}\n"
            f"  Diff:       {abs(seq_result.phi - par_result.phi)}"
        )


class TestStructuralInvariants:
    """Test structural properties of results.

    These tests verify that SIA objects have the expected structure
    and that internal consistency requirements are met.
    """

    def test_sia_has_required_attributes(self, s):
        """SIA must have all required attributes.

        The SystemIrreducibilityAnalysis class defines _sia_attributes
        as the canonical list of required attributes. All SIA objects
        must have these attributes present (even if some are None).

        If this fails, the SIA data structure has been modified in a
        way that breaks compatibility with the defined schema.
        """
        result = s.sia()

        # Check all _sia_attributes are present
        for attr in result._sia_attributes:
            assert hasattr(result, attr), (
                f"SIA is missing required attribute: '{attr}'\n"
                f"This indicates a structural change to SystemIrreducibilityAnalysis"
            )

    def test_sia_partition_attribute_consistency(self, s, micro_s, reducible):
        """Partition attribute must be consistent with phi value.

        Invariant: If phi > 0, partition should not be NullCut.
        If phi == 0, partition should be NullCut (for reducible systems).

        This tests internal consistency of the SIA result.
        """
        from pyphi.models.cuts import NullCut

        # Irreducible systems with phi > 0 should have non-null partitions
        s_result = s.sia()
        if s_result.phi > 0:
            assert not isinstance(
                s_result.partition, NullCut
            ), "System with phi > 0 has NullCut partition (should have real partition)"

        micro_result = micro_s.sia()
        if micro_result.phi > 0:
            assert not isinstance(
                micro_result.partition, NullCut
            ), "System with phi > 0 has NullCut partition (should have real partition)"

        # Reducible system should have null partition
        reducible_result = reducible.sia()
        assert isinstance(
            reducible_result.partition, NullCut
        ), "Reducible system should have NullCut partition"

    def test_partition_reduces_or_maintains_phi(self, s, micro_s):
        """Partitioned system cannot have more phi than unpartitioned.

        This is a fundamental IIT property: partitioning can only reduce
        or maintain integrated information. The partitioned CES should
        have phi <= unpartitioned CES phi.

        This validates that the partition finding algorithm correctly
        identified a minimizing partition.
        """
        # Only test systems with phi > 0
        for subsystem in [s, micro_s]:
            result = subsystem.sia()

            if result.phi > 0:
                # If partitioned_ces exists, its phi should be <= unpartitioned
                # (This validates the minimization found the right partition)
                if hasattr(result, "partitioned_ces") and result.partitioned_ces:
                    assert result.partitioned_ces.phi <= result.ces.phi, (
                        f"Partitioned CES has more phi than unpartitioned CES!\n"
                        f"  Unpartitioned phi: {result.ces.phi}\n"
                        f"  Partitioned phi:   {result.partitioned_ces.phi}\n"
                        f"This violates the IIT minimization property."
                    )


class TestConfigurationInvariants:
    """Test configuration-dependent behavior is consistent.

    Configuration options should have predictable effects on results.
    These tests verify that changing configuration produces expected
    behavior changes.
    """

    @pytest.mark.emd
    @skip_if_no_pyemd
    def test_selfloop_phi_depends_on_config(self, noisy_selfloop_single):
        """Single nodes with self-loops: phi depends on config.

        Configuration: SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI
        - When False: single micro-nodes with self-loops should have phi=0
        - When True (with EMD): should have phi > 0

        This tests that the configuration option is correctly respected.
        """
        # With config disabled, phi should be 0
        with config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=False):
            result_disabled = noisy_selfloop_single.sia()
            assert (
                result_disabled.phi == 0.0
            ), "Expected phi=0 when SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=False"

        # With config enabled and EMD, phi should be > 0
        with config.override(
            SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True,
            REPERTOIRE_DISTANCE="EMD",
        ):
            result_enabled = noisy_selfloop_single.sia()
            assert result_enabled.phi > 0.0, (
                "Expected phi > 0 when SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True "
                "and using EMD distance measure"
            )

    def test_cache_clearing_option(self, s):
        """Cache clearing configuration should be respected.

        Configuration: CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA
        - When True: caches should be empty after SIA computation
        - When False: caches should contain data after SIA computation

        This tests configuration-dependent side effects.
        """
        # Test with cache clearing disabled
        with config.override(
            CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=False,
            PARALLEL=False,
            CACHE_REPERTOIRES=True,
        ):
            _ = s.sia()
            assert (
                s._repertoire_cache.cache
            ), "Cache should have entries when clearing is disabled"

        # Test with cache clearing enabled
        with config.override(
            CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=True,
            PARALLEL=False,
            CACHE_REPERTOIRES=True,
        ):
            _ = s.sia()
            assert (
                not s._repertoire_cache.cache
            ), "Cache should be empty when clearing is enabled"


class TestPhiStructureInvariants:
    """Test invariants for IIT 4.0 phi_structure results.

    These tests verify properties specific to IIT 4.0's phi_structure
    computation, including distinctions and relations.
    """

    @pytest.mark.parametrize("example_name", ["basic", "fig4", "xor"])
    def test_phi_structure_has_distinctions(self, example_name):
        """Non-trivial systems should have at least one distinction.

        IIT 4.0 Property: Systems with integrated information should have
        at least one irreducible distinction (concept with cause-effect power).

        If a system has phi > 0 but no distinctions, something is wrong
        with the distinction-finding algorithm.
        """
        from pyphi.examples import EXAMPLES

        subsystem = EXAMPLES["subsystem"][example_name]()
        result = new_big_phi.phi_structure(subsystem)

        # Systems that have phi should have distinctions
        if hasattr(result, "phi") and result.phi > 0:
            assert hasattr(
                result, "distinctions"
            ), f"System '{example_name}' has phi > 0 but no distinctions attribute"
            assert (
                len(result.distinctions) > 0
            ), f"System '{example_name}' has phi > 0 but zero distinctions"

    @pytest.mark.parametrize("example_name", ["basic", "fig4"])
    def test_phi_structure_has_relations(self, example_name):
        """Systems with multiple distinctions should have relations.

        IIT 4.0 Property: Relations capture dependencies between distinctions.
        Systems with 2+ distinctions typically have relations between them.

        Note: This is a soft requirement - some systems might have independent
        distinctions with no relations.
        """
        from pyphi.examples import EXAMPLES

        subsystem = EXAMPLES["subsystem"][example_name]()
        result = new_big_phi.phi_structure(subsystem)

        # If system has multiple distinctions, check for relations
        if hasattr(result, "distinctions") and len(result.distinctions) >= 2:
            assert hasattr(result, "relations"), (
                f"System '{example_name}' has {len(result.distinctions)} distinctions "
                f"but no relations attribute"
            )
            # Relations might be None for some systems, so don't assert it's non-empty


class TestPermutationSymmetry:
    """Systems related by node permutation must have identical phi values.

    AND-XOR and XOR-AND have the same all-ones connectivity matrix but swap
    which node gets the AND vs XOR gate. They are related by the node
    permutation π: 0↔1. Under this permutation, state (a,b) in AND-XOR
    maps to (b,a) in XOR-AND.

    All measures must be invariant under this permutation.
    """

    def test_system_intrinsic_information_symmetric(self):
        """Cause/effect intrinsic information must be equal for permuted systems."""
        sub_ax = Subsystem(example_networks.and_xor_network(), (0, 1))
        sub_xa = Subsystem(example_networks.xor_and_network(), (1, 0))
        ss_ax = new_big_phi.system_intrinsic_information(sub_ax)
        ss_xa = new_big_phi.system_intrinsic_information(sub_xa)
        assert float(ss_ax.cause.intrinsic_information) == pytest.approx(
            float(ss_xa.cause.intrinsic_information)
        )
        assert float(ss_ax.effect.intrinsic_information) == pytest.approx(
            float(ss_xa.effect.intrinsic_information)
        )

    def test_sia_phi_symmetric(self):
        """Overall phi must be equal for permuted systems."""
        sub_ax = Subsystem(example_networks.and_xor_network(), (0, 1))
        sub_xa = Subsystem(example_networks.xor_and_network(), (1, 0))
        sia_ax = new_big_phi.sia(sub_ax)
        sia_xa = new_big_phi.sia(sub_xa)
        assert float(sia_ax.phi) == pytest.approx(float(sia_xa.phi))

    def test_sia_phi_c_symmetric(self):
        """phi_c must be equal for permuted systems.

        This is the specific invariant that was violated by the tied-state
        bug: AND-XOR(0,1) reported phi_c=0.5 while XOR-AND(1,0) reported
        phi_c=0.0, due to arbitrary tie-breaking in the specified cause state.
        """
        sub_ax = Subsystem(example_networks.and_xor_network(), (0, 1))
        sub_xa = Subsystem(example_networks.xor_and_network(), (1, 0))
        sia_ax = new_big_phi.sia(sub_ax)
        sia_xa = new_big_phi.sia(sub_xa)
        phi_c_ax = float(sia_ax.cause.phi) if sia_ax.cause else 0.0
        phi_c_xa = float(sia_xa.cause.phi) if sia_xa.cause else 0.0
        assert phi_c_ax == pytest.approx(phi_c_xa), (
            f"phi_c differs for permuted systems: "
            f"AND-XOR(0,1)={phi_c_ax}, XOR-AND(1,0)={phi_c_xa}"
        )

    def test_sia_phi_e_symmetric(self):
        """phi_e must be equal for permuted systems."""
        sub_ax = Subsystem(example_networks.and_xor_network(), (0, 1))
        sub_xa = Subsystem(example_networks.xor_and_network(), (1, 0))
        sia_ax = new_big_phi.sia(sub_ax)
        sia_xa = new_big_phi.sia(sub_xa)
        phi_e_ax = float(sia_ax.effect.phi) if sia_ax.effect else 0.0
        phi_e_xa = float(sia_xa.effect.phi) if sia_xa.effect else 0.0
        assert phi_e_ax == pytest.approx(phi_e_xa), (
            f"phi_e differs for permuted systems: "
            f"AND-XOR(0,1)={phi_e_ax}, XOR-AND(1,0)={phi_e_xa}"
        )
