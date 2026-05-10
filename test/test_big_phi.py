"""
Golden end-to-end tests for System Irreducibility Analysis (SIA).

These are "golden reference tests" that validate complete SIA computations
against JSON fixtures stored in test/data/sia/. They ensure that the entire
computation pipeline produces results matching historical validated outputs.

Test Substrates:
- s: Standard 3-node substrate (OR, COPY, XOR gates)
- s_noised: Standard substrate with noise added to TPM
- micro_s: 4-node highly connected substrate
- macro_s: 2-node stochastic/macro-level substrate
- big_subsys_0_thru_3: 4-node subset of 5-node substrate
- big_subsys_all_complete: 5-node complete graph
- rule152_s: 5-node cellular automaton (rule 152)

Each test compares the full SIA object against a JSON fixture. These tests
are comprehensive but brittle to serialization format changes. For more
robust component-level tests, see test_big_phi_robust.py.

Test Organization:
- Configuration tests: Verify config-dependent behavior
- Golden tests: Compare full SIA against JSON fixtures (both sequential and parallel)
- Edge case tests: Empty systems, single nodes, disconnected substrates

Historical Note:
These tests represent validated computational results from the PyPhi
implementation of IIT 4.0. When refactoring, these tests ensure that
algorithmic behavior remains unchanged.
"""

import pytest

from pyphi import cache as pyphi_cache
from pyphi import config
from pyphi.formalism import iit3
from pyphi.formalism import iit4 as new_big_phi

from .conftest import skip_if_no_pyemd

# pylint: disable=unused-argument

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_clear_system_caches_after_computing_sia_config_option(s):
    pyphi_cache.clear_all()
    with config.override(
        clear_system_caches_after_computing_sia=False,
        parallel=False,
        cache_repertoires=True,
    ):
        s.sia()
        assert any(stats["size"] > 0 for stats in s.cache_info().values())

    with config.override(
        clear_system_caches_after_computing_sia=True,
        parallel=False,
        cache_repertoires=True,
    ):
        s.sia()
        assert all(stats["size"] == 0 for stats in s.cache_info().values())


def test_conceptual_info(s):
    assert iit3.conceptual_info(s) == 1.0


def test_sia_empty_system(s_empty):
    assert s_empty.sia() == new_big_phi.NullSystemIrreducibilityAnalysis(
        node_indices=s_empty.node_indices,
    )


def test_sia_disconnected_substrate(reducible):
    assert reducible.sia() == new_big_phi.NullSystemIrreducibilityAnalysis(
        node_indices=reducible.node_indices,
    )


@pytest.mark.emd
@skip_if_no_pyemd
@config.override(single_micro_nodes_with_selfloops_have_phi=True)
@config.override(version="IIT_3_0", repertoire_measure="EMD")
def test_sia_single_micro_node_selfloops_have_phi(noisy_selfloop_single):
    """Test that single micro-nodes with self-loops have phi under IIT 3.0 + EMD.

    Configuration:
    - FORMALISM="IIT_3_0"
    - SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True
    - REPERTOIRE_DISTANCE="EMD"

    Substrate: Single node with noisy self-loop

    Theoretical basis: Self-loops create cause-effect structure even in
    single-node systems under micro-level analysis. The specific value
    is derived from Earth Mover's Distance computation on the self-loop
    repertoire distribution under the IIT 3.0 SIA path.

    Requires: pyemd package (install with: pip install pyphi[emd])
    """
    assert noisy_selfloop_single.sia().phi == pytest.approx(0.6868774943095, rel=1e-10)


@config.override(single_micro_nodes_with_selfloops_have_phi=False)
def test_sia_single_micro_node_selfloops_dont_have_phi(noisy_selfloop_single):
    assert noisy_selfloop_single.sia().phi == 0.0


def test_sia_single_micro_nodes_without_selfloops_dont_have_phi(s_single):
    assert s_single.sia().phi == 0.0


# s ======================================================


@config.override(parallel=False)
def test_sia_standard_example_sequential(s, s_expected_sia):
    assert s.sia() == s_expected_sia


def test_sia_standard_example_parallel(s, s_expected_sia):
    assert s.sia() == s_expected_sia


def test_sia_standard_example_complete_parallel(s_complete, s_expected_sia):
    assert s_complete.sia() == s_expected_sia


# s_noised ======================================================


@config.override(parallel=False)
def test_sia_noised_example_sequential(s_noised, s_noised_expected_sia):
    assert s_noised.sia() == s_noised_expected_sia


def test_sia_noised_example_parallel(s_noised, s_noised_expected_sia):
    actual = s_noised.sia()
    assert actual == s_noised_expected_sia


# micro_s ======================================================


@config.override(parallel=False)
def test_sia_micro_sequential(micro_s, micro_s_expected_sia):
    assert micro_s.sia() == micro_s_expected_sia


def test_sia_micro_parallel(micro_s, micro_s_expected_sia):
    assert micro_s.sia() == micro_s_expected_sia


# big_subsys_all_complete ======================================================


@pytest.mark.slow
@config.override(parallel=False)
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


@config.override(parallel=False)
def test_sia_big_substrate_0_thru_3_sequential(
    big_subsys_0_thru_3, big_subsys_0_thru_3_expected_sia
):
    assert big_subsys_0_thru_3.sia() == big_subsys_0_thru_3_expected_sia


def test_sia_big_substrate_0_thru_3_parallel(
    big_subsys_0_thru_3, big_subsys_0_thru_3_expected_sia
):
    assert big_subsys_0_thru_3.sia() == big_subsys_0_thru_3_expected_sia


# rule152_s ======================================================
# Has ties, so just checking big phi for now
#
# Note: The rule152 cellular automaton substrate has tied partitions (multiple
# partitions with the same phi value). When ties exist, the partition selection
# may vary between runs while phi remains constant. Therefore, we only check
# the phi value rather than full SIA equality.
#
# Substrate: 5-node cellular automaton following rule 152
# Expected phi: 0.83...
# Marked @veryslow: Cellular automaton substrates are computationally very expensive


@pytest.mark.veryslow
@config.override(parallel=False)
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


@config.override(parallel=False)
def test_sia_macro_sequential(macro_s, macro_s_expected_sia):
    assert macro_s.sia() == macro_s_expected_sia


def test_sia_macro_parallel(macro_s, macro_s_expected_sia):
    assert macro_s.sia() == macro_s_expected_sia


# ======================================================

# sia_bipartitions no longer exists
"""def test_sia_bipartitions():
    with config.override(CUT_ONE_APPROXIMATION=False):
        answer = [
            SystemPartition(Direction.EFFECT, (1,), (2, 3, 4)),
            SystemPartition(Direction.EFFECT, (2,), (1, 3, 4)),
            SystemPartition(Direction.EFFECT, (1, 2), (3, 4)),
            SystemPartition(Direction.EFFECT, (3,), (1, 2, 4)),
            SystemPartition(Direction.EFFECT, (1, 3), (2, 4)),
            SystemPartition(Direction.EFFECT, (2, 3), (1, 4)),
            SystemPartition(Direction.EFFECT, (1, 2, 3), (4,)),
            SystemPartition(Direction.EFFECT, (4,), (1, 2, 3)),
            SystemPartition(Direction.EFFECT, (1, 4), (2, 3)),
            SystemPartition(Direction.EFFECT, (2, 4), (1, 3)),
            SystemPartition(Direction.EFFECT, (1, 2, 4), (3,)),
            SystemPartition(Direction.EFFECT, (3, 4), (1, 2)),
            SystemPartition(Direction.EFFECT, (1, 3, 4), (2,)),
            SystemPartition(Direction.EFFECT, (2, 3, 4), (1,)),
        ]
        assert iit3.sia_bipartitions((1, 2, 3, 4)) == answer

    with config.override(CUT_ONE_APPROXIMATION=True):
        answer = [
            SystemPartition(Direction.EFFECT, (1,), (2, 3, 4)),
            SystemPartition(Direction.EFFECT, (2,), (1, 3, 4)),
            SystemPartition(Direction.EFFECT, (3,), (1, 2, 4)),
            SystemPartition(Direction.EFFECT, (4,), (1, 2, 3)),
            SystemPartition(Direction.EFFECT, (2, 3, 4), (1,)),
            SystemPartition(Direction.EFFECT, (1, 3, 4), (2,)),
            SystemPartition(Direction.EFFECT, (1, 2, 4), (3,)),
            SystemPartition(Direction.EFFECT, (1, 2, 3), (4,)),
        ]
        assert iit3.sia_bipartitions((1, 2, 3, 4)) == answer
"""


# Not relevant anymore because ces concepts do not store system
"""@pytest.mark.parametrize("parallel", [False, True])
def test_ces_concepts_share_the_same_system(parallel, s):
    with config.override(parallel=parallel):
        ces = iit3.ces(s)
        for concept in ces:
            assert concept.system is ces.system
"""


@pytest.mark.slow
def test_parallel_and_sequential_ces_are_equal(s, micro_s, macro_s):
    with config.override(parallel=False):
        c = iit3.ces(s)
        c_micro = iit3.ces(micro_s)
        c_macro = iit3.ces(macro_s)

    with config.override(parallel=True):
        assert set(c) == set(iit3.ces(s))
        assert set(c_micro) == set(iit3.ces(micro_s))
        assert set(c_macro) == set(iit3.ces(macro_s))
