"""
Robust component-level tests for System Irreducibility Analysis (SIA).

These tests complement the golden tests in test_big_phi.py by providing:
1. Serialization-independent phi value checks
2. Component-level validation (partitions, repertoires)
3. Better failure diagnostics
4. Fast-running regression tests

Unlike the golden tests which compare entire SIA objects against JSON
fixtures, these tests check individual components. This makes them more
robust to refactoring, especially data model changes.

Expected values are extracted from the same JSON fixtures used by golden
tests, ensuring consistency.
"""

import pytest

from pyphi import config
from pyphi.models.cuts import NullCut
from pyphi.new_big_phi import NullSystemIrreducibilityAnalysis

from .conftest import skip_if_no_pyemd

# Expected phi values extracted from JSON fixtures in test/data/sia/
# These values serve as golden references, extracted once to avoid
# repeated JSON parsing in tests.
EXPECTED_PHI_VALUES = {
    "s": 0.41503749927884376,
    "s_noised": 1.5232604640011718,
    "micro_s": 1.3317633845170522,
    "macro_s": 1.6562000000000006,
    "big_subsys_0_thru_3": 0.0,
    "big_subsys_all_complete": 0.0,
    "rule152_s": 0.8300749985576875,
}


# ============================================================================
# Phi Value Tests (Serialization-Independent)
# ============================================================================


class TestPhiValues:
    """Test phi values independently of object serialization.

    These tests verify that core phi computations produce expected values
    without depending on how SIA objects are serialized. They're fast and
    provide clear regression detection.
    """

    def test_sia_standard_example_phi_value(self, s):
        """Standard example computes correct phi value.

        Network: 3-node standard network (OR, COPY, XOR gates)
        State: (1, 0, 0)
        Expected: phi = 0.415...

        If this fails, the core integration computation has changed.
        Check partition finding and repertoire distance calculation.
        """
        result = s.sia()
        expected_phi = EXPECTED_PHI_VALUES["s"]
        # Convert to float for comparison (result.phi is PyPhiFloat)
        actual_phi = float(result.phi)
        assert actual_phi == pytest.approx(expected_phi, rel=1e-13), (
            f"Standard example phi changed:\n"
            f"  Expected: {expected_phi}\n"
            f"  Got:      {actual_phi}\n"
            f"  Diff:     {abs(actual_phi - expected_phi)}"
        )

    def test_sia_noised_example_phi_value(self, s_noised):
        """Noised example computes correct phi value.

        Network: Standard network with noise added to TPM
        Expected: phi = 1.523...

        Noise affects repertoire distributions and thus phi values.
        """
        result = s_noised.sia()
        expected_phi = EXPECTED_PHI_VALUES["s_noised"]
        assert float(result.phi) == pytest.approx(expected_phi, rel=1e-13)

    def test_sia_micro_phi_value(self, micro_s):
        """Micro network computes correct phi value.

        Network: 4-node highly connected network
        Expected: phi = 1.331...

        Tests computation on denser network topology.
        """
        result = micro_s.sia()
        expected_phi = EXPECTED_PHI_VALUES["micro_s"]
        assert float(result.phi) == pytest.approx(expected_phi, rel=1e-13)

    def test_sia_macro_phi_value(self, macro_s):
        """Macro network computes correct phi value.

        Network: 2-node stochastic/macro-level network
        Expected: phi = 1.656...

        Tests computation with probabilistic transitions.
        """
        result = macro_s.sia()
        expected_phi = EXPECTED_PHI_VALUES["macro_s"]
        assert float(result.phi) == pytest.approx(expected_phi, rel=1e-13)

    def test_sia_big_network_0_thru_3_phi_value(self, big_subsys_0_thru_3):
        """Big network subsystem (nodes 0-3) computes correct phi value.

        Network: 4-node subset of 5-node network
        Expected: phi = 0.0 (this particular subsystem is reducible)
        """
        result = big_subsys_0_thru_3.sia()
        expected_phi = EXPECTED_PHI_VALUES["big_subsys_0_thru_3"]
        assert float(result.phi) == pytest.approx(expected_phi, abs=1e-13)

    @pytest.mark.slow
    def test_sia_big_network_complete_phi_value(self, big_subsys_all_complete):
        """Big network all nodes (complete graph) phi value.

        Network: 5-node complete graph
        Expected: phi = 0.0

        This is marked @slow because it's computationally expensive.
        """
        result = big_subsys_all_complete.sia()
        expected_phi = EXPECTED_PHI_VALUES["big_subsys_all_complete"]
        assert float(result.phi) == pytest.approx(expected_phi, abs=1e-13)

    @pytest.mark.veryslow
    def test_sia_rule152_phi_value(self, rule152_s):
        """Rule 152 cellular automaton computes correct phi value.

        Network: 5-node cellular automaton (rule 152)
        Expected: phi = 0.830...

        Note: Full SIA comparison has ties in partitions, so golden test
        only checks phi value. This test does the same.

        This is marked @veryslow because cellular automaton networks
        are computationally very expensive.
        """
        result = rule152_s.sia()
        expected_phi = EXPECTED_PHI_VALUES["rule152_s"]
        assert float(result.phi) == pytest.approx(expected_phi, rel=1e-13)


# ============================================================================
# Component Structure Tests
# ============================================================================


class TestSIAComponentStructure:
    """Test that SIA components have expected structure.

    These tests verify that SIA objects contain the expected components
    (repertoires, partitions, etc.) without checking full object equality.
    This catches missing or incorrectly constructed components.
    """

    def test_sia_standard_example_has_repertoires(self, s):
        """Standard example SIA includes cause/effect repertoires.

        Validates that repertoire computation occurred and results
        were stored in the SIA object.

        If this fails, repertoire computation may have been skipped
        or RIA storage was modified.
        """
        result = s.sia()

        # Check cause repertoire exists and has required attributes
        assert result.cause is not None, "SIA missing cause repertoire"
        assert hasattr(result.cause, "phi"), "Cause RIA missing phi attribute"
        assert hasattr(
            result.cause, "mechanism"
        ), "Cause RIA missing mechanism attribute"
        assert hasattr(result.cause, "purview"), "Cause RIA missing purview attribute"

        # Check effect repertoire exists and has required attributes
        assert result.effect is not None, "SIA missing effect repertoire"
        assert hasattr(result.effect, "phi"), "Effect RIA missing phi attribute"
        assert hasattr(
            result.effect, "mechanism"
        ), "Effect RIA missing mechanism attribute"
        assert hasattr(result.effect, "purview"), "Effect RIA missing purview attribute"

    def test_sia_standard_example_has_system_state(self, s):
        """Standard example SIA includes system state specification.

        The system state specification captures the cause/effect states
        specified by the full system.
        """
        result = s.sia()

        assert result.system_state is not None, "SIA missing system_state attribute"
        assert hasattr(result.system_state, "cause"), "SystemState missing cause"
        assert hasattr(result.system_state, "effect"), "SystemState missing effect"

    def test_sia_has_partition_info(self, s, micro_s):
        """SIA includes partition information.

        For irreducible systems (phi > 0), partition information describes
        the minimum information partition (MIP).
        """
        for subsystem in [s, micro_s]:
            result = subsystem.sia()

            # All SIAs should have partition attribute
            assert hasattr(result, "partition"), "SIA missing partition attribute"

            # Non-null SIAs should have non-null partitions
            if not isinstance(result, NullSystemIrreducibilityAnalysis):
                assert result.partition is not None, "Non-null SIA has None partition"


# ============================================================================
# Partition Type Tests
# ============================================================================


class TestPartitionTypes:
    """Test partition types are appropriate for each system.

    Different systems should have different kinds of partitions depending
    on their reducibility and structure.
    """

    def test_sia_standard_example_partition_type(self, s):
        """Standard example uses expected partition type.

        The standard example is irreducible (phi > 0), so it should
        have a real partition (not NullCut).

        If this fails, partition scheme or reducibility detection
        may have changed.
        """
        result = s.sia()

        # System has phi > 0, so should have non-null partition
        assert result.phi > 0, "Standard example should have phi > 0"
        assert not isinstance(
            result.partition, NullCut
        ), "Irreducible system has NullCut partition"

    def test_reducible_system_has_null_partition(self, reducible):
        """Reducible system should have null partition.

        Reducible/disconnected systems have phi=0 and should return
        NullSystemIrreducibilityAnalysis with NullCut partition.
        """
        result = reducible.sia()

        assert isinstance(
            result, NullSystemIrreducibilityAnalysis
        ), "Reducible system should return NullSIA"
        assert isinstance(
            result.partition, NullCut
        ), "Reducible system should have NullCut partition"
        assert result.phi == 0.0, "Reducible system should have phi=0"

    def test_empty_subsystem_has_null_partition(self, s_empty):
        """Empty subsystem should have null partition.

        Empty subsystems (no nodes) cannot have integration and should
        return NullSystemIrreducibilityAnalysis.
        """
        result = s_empty.sia()

        assert isinstance(
            result, NullSystemIrreducibilityAnalysis
        ), "Empty subsystem should return NullSIA"
        assert isinstance(
            result.partition, NullCut
        ), "Empty subsystem should have NullCut partition"
        assert result.phi == 0.0, "Empty subsystem should have phi=0"


# ============================================================================
# Configuration-Dependent Tests
# ============================================================================


class TestConfigurationDependentValues:
    """Test configuration-dependent phi values.

    Some phi values depend on specific configuration settings.
    These tests document those dependencies and verify the values
    are correct under each configuration.
    """

    @pytest.mark.emd
    @skip_if_no_pyemd
    @config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True)
    @config.override(REPERTOIRE_DISTANCE="EMD")
    def test_sia_selfloop_node_phi_with_emd(self, noisy_selfloop_single):
        """Single node with self-loop has phi under EMD distance.

        Expected: phi = 0.36

        Configuration:
        - SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True
        - REPERTOIRE_DISTANCE="EMD"

        Network: Single node with noisy self-loop

        Theoretical basis: Self-loops create cause-effect structure even
        in single-node systems under micro-level analysis. The specific
        value (0.36) is derived from EMD computation on the self-loop
        repertoire.

        Precision sensitivity: Value is stable to 2 decimal places across
        different EMD implementations. We use slightly looser tolerance
        (abs=0.01) to account for EMD numerical variations.
        """
        result = noisy_selfloop_single.sia()

        # Use absolute tolerance for EMD-based values (less precise)
        assert result.phi == pytest.approx(0.36, abs=0.01), (
            f"Single node with self-loop phi changed under EMD:\n"
            f"  Expected: 0.36\n"
            f"  Got:      {result.phi}\n"
            f"  Diff:     {abs(result.phi - 0.36)}"
        )

    @config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=False)
    def test_sia_selfloop_node_no_phi_when_disabled(self, noisy_selfloop_single):
        """Single node with self-loop has phi=0 when config disabled.

        Configuration: SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=False

        When this configuration is disabled, single micro-nodes should
        not have phi even if they have self-loops.
        """
        result = noisy_selfloop_single.sia()

        assert result.phi == 0.0, (
            "Single node should have phi=0 when "
            "SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=False"
        )


# ============================================================================
# Sequential vs Parallel Consistency
# ============================================================================


class TestSequentialParallelConsistency:
    """Test that sequential and parallel execution produce identical SIAs.

    These tests duplicate some coverage from test_invariants.py but are
    specific to SIA computation. They verify that parallelization doesn't
    change results for specific test networks.
    """

    @pytest.mark.parametrize(
        "subsystem_fixture",
        ["s", "micro_s", "macro_s", "s_noised"],
    )
    def test_sia_sequential_equals_parallel_phi(self, subsystem_fixture, request):
        """Verify sequential and parallel SIA have same phi value.

        This is a quick check that parallelization doesn't change phi.
        For full equality checking, see test_invariants.py.

        Args:
            subsystem_fixture: Name of subsystem fixture
            request: Pytest request object
        """
        subsystem = request.getfixturevalue(subsystem_fixture)

        # Sequential computation
        with config.override(PARALLEL=False):
            seq_result = subsystem.sia()

        # Parallel computation
        with config.override(PARALLEL=True):
            par_result = subsystem.sia()

        # Phi values must match exactly
        assert seq_result.phi == par_result.phi, (
            f"Sequential and parallel phi differ for "
            f"{subsystem_fixture}:\n"
            f"  Sequential: {seq_result.phi}\n"
            f"  Parallel:   {par_result.phi}\n"
            f"  Diff:       {abs(seq_result.phi - par_result.phi)}"
        )


# ============================================================================
# Eq. 23: ii(s) cap and GID-only partition evaluation
# ============================================================================


class TestEq23IntrinsicInformationCap:
    """Test that sia() implements Eq. 23 from Mayner, Marshall, & Tononi 2025.

    φ_s(s) = min{φ_c(s), φ_e(s), ii(s)}

    where ii(s) = min{ii_c(s), ii_e(s)} and ii_d = min{i_diff_d, i_spec_d}.

    Partition evaluation uses GID only (Eqs. 19-20); i_diff and i_spec are
    applied as the ii(s) cap separately.
    """

    II_CONFIG = dict(
        REPERTOIRE_DISTANCE="INTRINSIC_INFORMATION",
        REPERTOIRE_DISTANCE_SPECIFICATION="INTRINSIC_SPECIFICATION",
        REPERTOIRE_DISTANCE_DIFFERENTIATION="INTRINSIC_DIFFERENTIATION",
    )

    @staticmethod
    def _noisy_copy_subsystem(p, state):
        """Create a 2-node noisy COPY system.

        Each node copies the other with probability p (LOLI state ordering).
        """
        import numpy as np
        from pyphi import Network, Subsystem

        tpm = np.array([
            [1 - p, 1 - p],  # (0,0)
            [1 - p, p],      # (1,0)
            [p, 1 - p],      # (0,1)
            [p, p],          # (1,1)
        ])
        cm = np.array([[0, 1], [1, 0]])
        network = Network(tpm, cm=cm, node_labels=["A", "B"])
        return Subsystem(network, state)

    def test_phi_capped_by_ii(self):
        """phi is capped by ii(s) = min(i_diff, i_spec) per direction.

        With p=0.8, state (1,1): GID(MIP) ≈ 0.868 but i_diff ≈ 0.644,
        so ii(s) ≈ 0.644 caps phi below GID(MIP).
        """
        from pyphi.direction import Direction
        from pyphi.new_big_phi import sia, system_intrinsic_information

        subsystem = self._noisy_copy_subsystem(0.8, (1, 1))
        with config.override(**self.II_CONFIG):
            sys_state = system_intrinsic_information(subsystem)
            result = sia(subsystem)

            # Compute ii(s) from components
            ii_cause = min(
                float(sys_state.cause.intrinsic_information),
                float(result.intrinsic_differentiation[Direction.CAUSE]),
            )
            ii_effect = min(
                float(sys_state.effect.intrinsic_information),
                float(result.intrinsic_differentiation[Direction.EFFECT]),
            )
            ii_s = min(ii_cause, ii_effect)

            # phi must equal ii(s), not GID(MIP)
            assert float(result.phi) == pytest.approx(ii_s, abs=1e-9)
            # GID(MIP) is larger than ii(s) — cap is binding
            gid_mip = min(float(result.cause.phi), float(result.effect.phi))
            assert gid_mip > ii_s + 1e-6

    def test_partition_evaluation_uses_gid_only(self):
        """Per-direction phi values at MIP are GID, not min(GID, i_diff).

        With p=0.8: GID ≈ 0.868, i_diff ≈ 0.644. The cause/effect phi
        values on the SIA should be the GID values (un-folded), not the
        old min(GID, i_diff).
        """
        from pyphi.direction import Direction
        from pyphi.new_big_phi import sia

        subsystem = self._noisy_copy_subsystem(0.8, (1, 1))
        with config.override(**self.II_CONFIG):
            result = sia(subsystem)

            i_diff = float(result.intrinsic_differentiation[Direction.CAUSE])
            cause_phi = float(result.cause.phi)
            effect_phi = float(result.effect.phi)

            # cause/effect phi should be GID, which is LARGER than i_diff
            assert cause_phi > i_diff + 1e-6
            assert effect_phi > i_diff + 1e-6

    def test_gid_distance_unaffected(self, s):
        """GID-based computation is unchanged by the Eq. 23 logic.

        The ii(s) cap and GID-only partition override only activate when
        REPERTOIRE_DISTANCE=INTRINSIC_INFORMATION.
        """
        from pyphi.new_big_phi import sia

        # Default config uses GENERALIZED_INTRINSIC_DIFFERENCE
        result = sia(s)
        assert float(result.phi) == pytest.approx(
            EXPECTED_PHI_VALUES["s"], abs=1e-9
        )


# ============================================================================
# Paper examples: Mayner, Marshall, & Tononi 2025 (arXiv:2510.03881)
# ============================================================================


class TestPaperExamples:
    """Regression tests for the paper's worked examples.

    These verify that PyPhi reproduces the analytical results from
    Mayner, Marshall, & Tononi 2025, "Intrinsic cause-effect power:
    the tradeoff between differentiation and specification."
    """

    II_CONFIG = dict(
        REPERTOIRE_DISTANCE="INTRINSIC_INFORMATION",
        REPERTOIRE_DISTANCE_SPECIFICATION="INTRINSIC_SPECIFICATION",
        REPERTOIRE_DISTANCE_DIFFERENTIATION="INTRINSIC_DIFFERENTIATION",
    )

    @staticmethod
    def _monad_subsystem(p):
        """Single-node system that stays in current state with probability p."""
        import numpy as np
        from pyphi import Network, Subsystem

        tpm = np.array([[1 - p], [p]])
        cm = np.array([[1]])
        network = Network(tpm, cm=cm)
        return Subsystem(network, state=(1,))

    def test_monad_intrinsic_information(self):
        """Example 1, Eq. 27: ii(s) = min{p*log(2p), -log(p)}.

        At p=0.744 (near the optimal): i_diff ≈ i_spec ≈ 0.427.
        The paper reports φ_s = 0.427 (Figure 2C).
        """
        import numpy as np
        from pyphi.new_big_phi import system_intrinsic_information

        p = 0.744
        subsystem = self._monad_subsystem(p)
        i_diff_expected = -np.log2(p)
        i_spec_expected = p * np.log2(2 * p)
        ii_expected = min(i_diff_expected, i_spec_expected)

        with config.override(**self.II_CONFIG):
            sys_state = system_intrinsic_information(subsystem)
            # system_intrinsic_information uses INTRINSIC_SPECIFICATION,
            # so it returns i_spec
            i_spec_pyphi = float(sys_state.effect.intrinsic_information)
            assert i_spec_pyphi == pytest.approx(i_spec_expected, abs=1e-6)
            # Verify the analytical ii value matches the paper
            assert ii_expected == pytest.approx(0.427, abs=0.001)

    @pytest.mark.parametrize(
        "p,i_diff_expected,i_spec_expected",
        [
            (0.744, 0.426625, 0.426591),  # crossover point (Figure 2C)
            (0.9, 0.152003, 0.763197),    # high determinism
            (0.6, 0.736966, 0.157821),    # high noise
        ],
    )
    def test_monad_i_diff_i_spec_tradeoff(self, p, i_diff_expected, i_spec_expected):
        """Verify i_diff and i_spec values across the tradeoff curve (Figure 2C).

        i_diff = -log2(p), i_spec = p*log2(2p) for a monad in its ON state.
        """
        import numpy as np
        from pyphi import direction, metrics
        from pyphi.new_big_phi import system_intrinsic_information

        subsystem = self._monad_subsystem(p)

        with config.override(**self.II_CONFIG):
            sys_state = system_intrinsic_information(subsystem)
            i_spec = float(sys_state.effect.intrinsic_information)
            assert i_spec == pytest.approx(i_spec_expected, abs=1e-5)

            # Compute i_diff from forward repertoire
            fr = subsystem.forward_repertoire(
                direction.Direction.EFFECT,
                subsystem.node_indices,
                subsystem.node_indices,
                None,
            )
            i_diff = float(
                metrics.distribution.intrinsic_differentiation(
                    fr, fr, state=subsystem.proper_state
                )
            )
            assert i_diff == pytest.approx(i_diff_expected, abs=1e-5)
