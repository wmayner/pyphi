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
from pyphi.conf import presets
from pyphi.formalism import iit4 as new_big_phi
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from pyphi.system import System

from . import example_substrates
from .conftest import skip_if_no_emd_backend


def _sia_kwargs():
    """Resolve measure kwargs from current config for module-level iit4 calls.

    The module-level ``sia``/``phi_structure``/``system_intrinsic_information``
    require explicit measures; tests invoke them with default-config measures.
    """
    return {
        "system_measure": resolve_system_measure(
            config.formalism.iit.system_phi_measure
        ),
        "specification_measure": resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    }


def _spec_kwargs():
    return {
        "specification_measure": resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        )
    }


class TestPhiInvariants:
    """Test invariants related to phi values.

    These tests verify that phi values satisfy fundamental IIT properties:
    - Phi is always non-negative
    - Empty systems have zero phi
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

    def test_empty_system_has_zero_phi(self, s_empty):
        """Empty systems have no integration (phi=0).

        IIT Property: A system with no elements cannot have integrated
        information. Empty systems must return NullSIA with phi=0.
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
        "system_fixture",
        ["s", "micro_s", "macro_s", "s_noised"],
    )
    def test_sequential_equals_parallel(self, system_fixture, request):
        """Sequential and parallel must produce identical results.

        This tests the fundamental requirement that parallelization
        cannot change results. If this fails, there's a bug in the
        parallel implementation (likely a race condition or shared
        state issue).

        Args:
            system_fixture: Name of system fixture to test
            request: Pytest request object for getting fixture values
        """
        system = request.getfixturevalue(system_fixture)

        # Compute with sequential mode
        with config.override(parallel=False):
            seq_result = system.sia()

        # Compute with parallel mode
        with config.override(parallel=True):
            par_result = system.sia()

        # Results must be exactly equal
        assert seq_result == par_result, (
            f"Parallel and sequential results differ for {system_fixture}:\n"
            f"  Sequential phi: {seq_result.phi}\n"
            f"  Parallel phi:   {par_result.phi}"
        )

        # Also check phi values explicitly for better error messages
        assert seq_result.phi == par_result.phi, (
            f"Phi values differ for {system_fixture}:\n"
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

        All SIA objects must have these attributes present (even if some
        are None). If this fails, the SIA data structure has been modified
        in a way that breaks compatibility with the defined schema.
        """
        result = s.sia()

        required_attributes = (
            "phi",
            "partition",
            "normalized_phi",
            "signed_phi",
            "signed_normalized_phi",
            "cause",
            "effect",
            "system_state",
            "current_state",
            "node_indices",
            "intrinsic_differentiation",
        )
        for attr in required_attributes:
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
        from pyphi.models.partitions import NullCut

        # Irreducible systems with phi > 0 should have non-null partitions
        s_result = s.sia()
        if s_result.phi > 0:
            assert not isinstance(s_result.partition, NullCut), (
                "System with phi > 0 has NullCut partition (should have real partition)"
            )

        micro_result = micro_s.sia()
        if micro_result.phi > 0:
            assert not isinstance(micro_result.partition, NullCut), (
                "System with phi > 0 has NullCut partition (should have real partition)"
            )

        # Reducible system should have null partition
        reducible_result = reducible.sia()
        assert isinstance(reducible_result.partition, NullCut), (
            "Reducible system should have NullCut partition"
        )

    def test_partition_reduces_or_maintains_phi(self, s, micro_s):
        """Partitioned system cannot have more phi than unpartitioned.

        This is a fundamental IIT property: partitioning can only reduce
        or maintain integrated information. The partitioned CES should
        have phi <= unpartitioned CES phi.

        This validates that the partition finding algorithm correctly
        identified a minimizing partition.
        """
        # Only test systems with phi > 0
        for system in [s, micro_s]:
            result = system.sia()

            if (
                result.phi > 0
                and hasattr(result, "partitioned_ces")
                and result.partitioned_ces
            ):
                # Partitioned CES should have phi <= unpartitioned CES
                # (validates the minimization found the right partition).
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
    @skip_if_no_emd_backend
    def test_selfloop_phi_depends_on_config(self, noisy_selfloop_single):
        """Single nodes with self-loops: phi depends on config.

        Configuration: ``single_micro_nodes_with_selfloops_have_phi``
        - When False: single micro-nodes with self-loops should have phi=0
        - When True (with EMD): should have phi > 0

        This tests that the configuration option is correctly respected.
        """
        # With config disabled, phi should be 0
        with config.override(single_micro_nodes_with_selfloops_have_phi=False):
            result_disabled = noisy_selfloop_single.sia()
            assert result_disabled.phi == 0.0, (
                "Expected phi=0 when single_micro_nodes_with_selfloops_have_phi=False"
            )

        # With config enabled under the canonical IIT 3.0 preset, phi > 0
        with config.override(
            **presets.iit3,
            single_micro_nodes_with_selfloops_have_phi=True,
        ):
            result_enabled = noisy_selfloop_single.sia()
            assert result_enabled.phi > 0.0, (
                "Expected phi > 0 when "
                "single_micro_nodes_with_selfloops_have_phi=True "
                "and using EMD distance measure"
            )

    def test_cache_clearing_option(self, s):
        """Cache clearing configuration should be respected.

        Configuration: ``clear_system_caches_after_computing_sia``
        - When True: caches should be empty after SIA computation
        - When False: caches should contain data after SIA computation

        This tests configuration-dependent side effects.
        """
        # Test with cache clearing disabled
        with config.override(
            clear_system_caches_after_computing_sia=False,
            parallel=False,
            cache_repertoires=True,
        ):
            _ = s.sia()
            assert any(stats["size"] > 0 for stats in s.cache_info().values()), (
                "Cache should have entries when clearing is disabled"
            )

        # Test with cache clearing enabled
        with config.override(
            clear_system_caches_after_computing_sia=True,
            parallel=False,
            cache_repertoires=True,
        ):
            _ = s.sia()
            assert all(stats["size"] == 0 for stats in s.cache_info().values()), (
                "Cache should be empty when clearing is enabled"
            )


class TestCauseEffectStructureInvariants:
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

        system = EXAMPLES["system"][example_name]()
        result = new_big_phi.ces(system, **_sia_kwargs())

        # Systems that have phi should have distinctions
        if hasattr(result, "phi") and result.phi > 0:
            assert hasattr(result, "distinctions"), (
                f"System '{example_name}' has phi > 0 but no distinctions attribute"
            )
            assert len(result.distinctions) > 0, (
                f"System '{example_name}' has phi > 0 but zero distinctions"
            )

    @pytest.mark.parametrize("example_name", ["basic", "fig4"])
    def test_phi_structure_has_relations(self, example_name):
        """Systems with multiple distinctions should have relations.

        IIT 4.0 Property: Relations capture dependencies between distinctions.
        Systems with 2+ distinctions typically have relations between them.

        Note: This is a soft requirement - some systems might have independent
        distinctions with no relations.
        """
        from pyphi.examples import EXAMPLES

        system = EXAMPLES["system"][example_name]()
        result = new_big_phi.ces(system, **_sia_kwargs())

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
        sub_ax = System(example_substrates.and_xor_substrate(), (0, 1))
        sub_xa = System(example_substrates.xor_and_substrate(), (1, 0))
        ss_ax = new_big_phi.system_intrinsic_information(sub_ax, **_spec_kwargs())
        ss_xa = new_big_phi.system_intrinsic_information(sub_xa, **_spec_kwargs())
        assert float(ss_ax.cause.intrinsic_information) == pytest.approx(
            float(ss_xa.cause.intrinsic_information)
        )
        assert float(ss_ax.effect.intrinsic_information) == pytest.approx(
            float(ss_xa.effect.intrinsic_information)
        )

    def test_sia_phi_symmetric(self):
        """Overall phi must be equal for permuted systems."""
        sub_ax = System(example_substrates.and_xor_substrate(), (0, 1))
        sub_xa = System(example_substrates.xor_and_substrate(), (1, 0))
        sia_ax = new_big_phi.sia(sub_ax, **_sia_kwargs())
        sia_xa = new_big_phi.sia(sub_xa, **_sia_kwargs())
        assert float(sia_ax.phi) == pytest.approx(float(sia_xa.phi))

    @pytest.mark.xfail(
        reason=(
            "Per-direction phi for permuted substrates is sensitive to the "
            "cascade's tie-break when sia.phi ties at zero. Strict "
            "permutation-invariant per-direction reporting requires "
            "substrate canonicalization — see ROADMAP P11.95c."
        ),
        strict=True,
    )
    def test_sia_per_direction_phi_multiset_symmetric(self):
        """The multiset ``{phi_c, phi_e}`` is equal for permuted systems."""
        sub_ax = System(example_substrates.and_xor_substrate(), (0, 1))
        sub_xa = System(example_substrates.xor_and_substrate(), (1, 0))
        sia_ax = new_big_phi.sia(sub_ax, **_sia_kwargs())
        sia_xa = new_big_phi.sia(sub_xa, **_sia_kwargs())
        phi_c_ax = float(sia_ax.cause.phi) if sia_ax.cause else 0.0
        phi_e_ax = float(sia_ax.effect.phi) if sia_ax.effect else 0.0
        phi_c_xa = float(sia_xa.cause.phi) if sia_xa.cause else 0.0
        phi_e_xa = float(sia_xa.effect.phi) if sia_xa.effect else 0.0
        ax_pair = sorted((phi_c_ax, phi_e_ax))
        xa_pair = sorted((phi_c_xa, phi_e_xa))
        assert ax_pair[0] == pytest.approx(xa_pair[0]) and ax_pair[1] == pytest.approx(
            xa_pair[1]
        )

    def test_system_state_reflects_mip_resolution(self):
        """system_state should reflect the specified state chosen by the MIP.

        When tied specified states are resolved by the MIP, the winning state
        (most vulnerable to the partition) should be back-propagated to
        system_state, so downstream consumers see the correct state.
        """
        sub = System(example_substrates.and_xor_substrate(), (0, 1))
        sia = new_big_phi.sia(sub, **_sia_kwargs())
        if sia.cause and sia.cause.specified_state:
            assert sia.system_state.cause.state == sia.cause.specified_state.state
        if sia.effect and sia.effect.specified_state:
            assert sia.system_state.effect.state == sia.effect.specified_state.state

    def test_system_state_preserves_ties_after_resolution(self):
        """system_state should still record all tied states after resolution."""
        sub = System(example_substrates.and_xor_substrate(), (0, 1))
        sia = new_big_phi.sia(sub, **_sia_kwargs())
        # The cause direction had 2 tied states
        assert len(sia.system_state.cause.ties) == 2
        tied_states = {t.state for t in sia.system_state.cause.ties}
        assert tied_states == {(0, 1), (1, 0)}

    def test_system_state_lies_in_same_permutation_orbit(self):
        """Cause states chosen for permuted substrates lie in the same orbit.

        Under paper-faithful state-tie resolution (P11.95b), the chosen
        cause state for a substrate is one of the tied states at maximum
        intrinsic information. For two substrates related by a node-
        permutation, the chosen state on each must be permutation-
        equivalent to the chosen state on the other — but not necessarily
        the direct image under the permutation. Strict per-direction
        equality requires substrate canonicalization (ROADMAP P11.95c).
        """
        sub_ax = System(example_substrates.and_xor_substrate(), (0, 1))
        sub_xa = System(example_substrates.xor_and_substrate(), (1, 0))
        sia_ax = new_big_phi.sia(sub_ax, **_sia_kwargs())
        sia_xa = new_big_phi.sia(sub_xa, **_sia_kwargs())
        ax_state = sia_ax.system_state.cause.state
        xa_state = sia_xa.system_state.cause.state
        ax_orbit = {ax_state, tuple(reversed(ax_state))}
        xa_orbit = {xa_state, tuple(reversed(xa_state))}
        assert ax_orbit == xa_orbit, (
            f"system_state.cause.state orbits differ for permuted systems: "
            f"AND-XOR orbit={ax_orbit}, XOR-AND orbit={xa_orbit}"
        )


@config.override(parallel=False)
def test_sia_is_deterministic_across_runs_sequential(big_subsys_all_complete):
    """Two consecutive ``.sia()`` calls return equal SIAs in sequential mode.

    The fully-connected 5-node substrate has multiple partitions tied at
    the MIP key ``(normalized_phi, -phi)``; ``PARTITION_LEX`` selects a
    canonical winner.
    """
    s1 = big_subsys_all_complete.sia()
    s2 = big_subsys_all_complete.sia()
    assert s1 == s2


@config.override(parallel=True)
def test_sia_is_deterministic_across_runs_parallel(big_subsys_all_complete):
    """Two consecutive ``.sia()`` calls return equal SIAs in parallel mode.

    The parallel dispatch path delivers MapReduce results in worker-completion
    order; deterministic SIA selection requires the strategy chain to fully
    canonicalize across runs.
    """
    s1 = big_subsys_all_complete.sia()
    s2 = big_subsys_all_complete.sia()
    assert s1 == s2
