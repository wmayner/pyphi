"""
Golden end-to-end tests for System Irreducibility Analysis (SIA).

These are "golden reference tests" that validate complete SIA computations
against JSON fixtures stored in test/data/sia/. They ensure that the entire
computation pipeline produces results matching historical validated outputs.

Test Networks:
- s: Standard 3-node network (OR, COPY, XOR gates)
- s_noised: Standard network with noise added to TPM
- micro_s: 4-node highly connected network
- macro_s: 2-node stochastic/macro-level network
- big_subsys_0_thru_3: 4-node subset of 5-node network
- big_subsys_all_complete: 5-node complete graph
- rule152_s: 5-node cellular automaton (rule 152)

Each test compares the full SIA object against a JSON fixture. These tests
are comprehensive but brittle to serialization format changes. For more
robust component-level tests, see test_big_phi_robust.py.

Test Organization:
- Configuration tests: Verify config-dependent behavior
- Golden tests: Compare full SIA against JSON fixtures (both sequential and parallel)
- Edge case tests: Empty subsystems, single nodes, disconnected networks

Historical Note:
These tests represent validated computational results from the PyPhi
implementation of IIT 4.0. When refactoring, these tests ensure that
algorithmic behavior remains unchanged.
"""

import pytest

from pyphi import compute
from pyphi import config
from pyphi import new_big_phi

from .conftest import skip_if_no_pyemd

# pylint: disable=unused-argument

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_clear_subsystem_caches_after_computing_sia_config_option(s):
    with config.override(
        CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=False,
        PARALLEL=False,
        CACHE_REPERTOIRES=True,
    ):
        s.sia()
        assert s._repertoire_cache.cache

    with config.override(
        CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=True,
        PARALLEL=False,
        CACHE_REPERTOIRES=True,
    ):
        s.sia()
        assert not s._repertoire_cache.cache


def test_conceptual_info(s):
    assert compute.subsystem.conceptual_info(s) == 1.0


def test_sia_empty_subsystem(s_empty):
    assert s_empty.sia() == new_big_phi.NullSystemIrreducibilityAnalysis(
        node_indices=s_empty.node_indices,
    )


def test_sia_disconnected_network(reducible):
    assert reducible.sia() == new_big_phi.NullSystemIrreducibilityAnalysis(
        node_indices=reducible.node_indices,
    )


@pytest.mark.emd
@skip_if_no_pyemd
@config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True)
@config.override(REPERTOIRE_DISTANCE="EMD")
def test_sia_single_micro_node_selfloops_have_phi(noisy_selfloop_single):
    """Test that single micro-nodes with self-loops have phi under EMD.

    Expected phi value: 0.36

    Configuration:
    - SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True
    - REPERTOIRE_DISTANCE="EMD"

    Network: Single node with noisy self-loop

    Theoretical basis: Self-loops create cause-effect structure even in
    single-node systems under micro-level analysis. The specific value
    (0.36) is derived from Earth Mover's Distance computation on the
    self-loop repertoire distribution.

    Precision sensitivity: Value is stable to 2 decimal places across
    different EMD implementations.

    Requires: pyemd package (install with: pip install pyphi[emd])
    """
    assert noisy_selfloop_single.sia().phi == 0.36


@config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=False)
def test_sia_single_micro_node_selfloops_dont_have_phi(noisy_selfloop_single):
    assert noisy_selfloop_single.sia().phi == 0.0


def test_sia_single_micro_nodes_without_selfloops_dont_have_phi(s_single):
    assert s_single.sia().phi == 0.0


# s ======================================================


@config.override(PARALLEL=False)
def test_sia_standard_example_sequential(s, s_expected_sia):
    assert s.sia() == s_expected_sia


def test_sia_standard_example_parallel(s, s_expected_sia):
    assert s.sia() == s_expected_sia


def test_sia_standard_example_complete_parallel(s_complete, s_expected_sia):
    assert s_complete.sia() == s_expected_sia


# s_noised ======================================================


@config.override(PARALLEL=False)
def test_sia_noised_example_sequential(s_noised, s_noised_expected_sia):
    assert s_noised.sia() == s_noised_expected_sia


def test_sia_noised_example_parallel(s_noised, s_noised_expected_sia):
    actual = s_noised.sia()
    assert actual == s_noised_expected_sia


# micro_s ======================================================


@config.override(PARALLEL=False)
def test_sia_micro_sequential(micro_s, micro_s_expected_sia):
    assert micro_s.sia() == micro_s_expected_sia


def test_sia_micro_parallel(micro_s, micro_s_expected_sia):
    assert micro_s.sia() == micro_s_expected_sia


# big_subsys_all_complete ======================================================


@pytest.mark.slow
@config.override(PARALLEL=False)
def test_sia_big_subsys_all_complete_sequential(
    big_subsys_all_complete, big_subsys_all_complete_expected_sia
):
    assert big_subsys_all_complete.sia() == big_subsys_all_complete_expected_sia


@pytest.mark.slow
def test_sia_big_subsys_all_complete_parallel(
    big_subsys_all_complete, big_subsys_all_complete_expected_sia
):
    assert big_subsys_all_complete.sia() == big_subsys_all_complete_expected_sia


# big_subsys_0_thru_3 ======================================================


@config.override(PARALLEL=False)
def test_sia_big_network_0_thru_3_sequential(
    big_subsys_0_thru_3, big_subsys_0_thru_3_expected_sia
):
    assert big_subsys_0_thru_3.sia() == big_subsys_0_thru_3_expected_sia


def test_sia_big_network_0_thru_3_parallel(
    big_subsys_0_thru_3, big_subsys_0_thru_3_expected_sia
):
    assert big_subsys_0_thru_3.sia() == big_subsys_0_thru_3_expected_sia


# rule152_s ======================================================
# Has ties, so just checking big phi for now
#
# Note: The rule152 cellular automaton network has tied partitions (multiple
# partitions with the same phi value). When ties exist, the partition selection
# may vary between runs while phi remains constant. Therefore, we only check
# the phi value rather than full SIA equality.
#
# Network: 5-node cellular automaton following rule 152
# Expected phi: 0.83...
# Marked @veryslow: Cellular automaton networks are computationally very expensive


@pytest.mark.veryslow
@config.override(PARALLEL=False)
def test_sia_rule152_s_sequential(rule152_s, rule152_s_expected_sia):
    """Rule 152 cellular automaton sequential computation.

    Only checks phi value due to tied partitions - see note above.
    """
    assert rule152_s.sia().phi == rule152_s_expected_sia.phi


@pytest.mark.veryslow
def test_sia_rule152_s_parallel(rule152_s, rule152_s_expected_sia):
    """Rule 152 cellular automaton parallel computation.

    Only checks phi value due to tied partitions - see note above.
    """
    assert rule152_s.sia().phi == rule152_s_expected_sia.phi


# macro_s ======================================================


@config.override(PARALLEL=False)
def test_sia_macro_sequential(macro_s, macro_s_expected_sia):
    assert macro_s.sia() == macro_s_expected_sia


def test_sia_macro_parallel(macro_s, macro_s_expected_sia):
    assert macro_s.sia() == macro_s_expected_sia


# ======================================================

# sia_bipartitions no longer exists
"""def test_sia_bipartitions():
    with config.override(CUT_ONE_APPROXIMATION=False):
        answer = [
            models.Cut((1,), (2, 3, 4)),
            models.Cut((2,), (1, 3, 4)),
            models.Cut((1, 2), (3, 4)),
            models.Cut((3,), (1, 2, 4)),
            models.Cut((1, 3), (2, 4)),
            models.Cut((2, 3), (1, 4)),
            models.Cut((1, 2, 3), (4,)),
            models.Cut((4,), (1, 2, 3)),
            models.Cut((1, 4), (2, 3)),
            models.Cut((2, 4), (1, 3)),
            models.Cut((1, 2, 4), (3,)),
            models.Cut((3, 4), (1, 2)),
            models.Cut((1, 3, 4), (2,)),
            models.Cut((2, 3, 4), (1,)),
        ]
        assert compute.subsystem.sia_bipartitions((1, 2, 3, 4)) == answer

    with config.override(CUT_ONE_APPROXIMATION=True):
        answer = [
            models.Cut((1,), (2, 3, 4)),
            models.Cut((2,), (1, 3, 4)),
            models.Cut((3,), (1, 2, 4)),
            models.Cut((4,), (1, 2, 3)),
            models.Cut((2, 3, 4), (1,)),
            models.Cut((1, 3, 4), (2,)),
            models.Cut((1, 2, 4), (3,)),
            models.Cut((1, 2, 3), (4,)),
        ]
        assert compute.subsystem.sia_bipartitions((1, 2, 3, 4)) == answer
"""


@pytest.mark.outdated
@pytest.mark.slow
@config.override(SYSTEM_PARTITION_TYPE="DIRECTED_BI")
def test_system_cut_styles(s):
    with config.override(SYSTEM_CUTS="3.0_STYLE"):
        assert compute.subsystem.phi(s) == 0.5  # 2.3125

    with config.override(SYSTEM_CUTS="CONCEPT_STYLE"):
        assert compute.subsystem.phi(s) == 0.6875


# Not relevant anymore because ces concepts do not store subsystem
"""@pytest.mark.parametrize("parallel", [False, True])
def test_ces_concepts_share_the_same_subsystem(parallel, s):
    with config.override(PARALLEL=parallel):
        ces = compute.subsystem.ces(s)
        for concept in ces:
            assert concept.subsystem is ces.subsystem
"""


@pytest.mark.slow
def test_parallel_and_sequential_ces_are_equal(s, micro_s, macro_s):
    with config.override(PARALLEL=False):
        c = compute.subsystem.ces(s)
        c_micro = compute.subsystem.ces(micro_s)
        c_macro = compute.subsystem.ces(macro_s)

    with config.override(PARALLEL=True):
        assert set(c) == set(compute.subsystem.ces(s))
        assert set(c_micro) == set(compute.subsystem.ces(micro_s))
        assert set(c_macro) == set(compute.subsystem.ces(macro_s))
