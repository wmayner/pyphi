import pytest

from pyphi import config, compute, models, new_big_phi, jsonify

# pylint: disable=unused-argument

# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_clear_subsystem_caches_after_computing_sia_config_option(s):
    with config.override(
        CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=False,
        PARALLEL=False,
        CACHE_REPERTOIRES=True,
    ):
        sia = s.sia()
        assert s._repertoire_cache.cache

    with config.override(
        CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=True,
        PARALLEL=False,
        CACHE_REPERTOIRES=True,
    ):
        sia = s.sia()
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


@config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True)
@config.override(REPERTOIRE_DISTANCE="EMD")
def test_sia_single_micro_node_selfloops_have_phi(noisy_selfloop_single):
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


@pytest.mark.veryslow
@config.override(PARALLEL=False)
def test_sia_rule152_s_sequential(rule152_s, rule152_s_expected_sia):
    assert rule152_s.sia().phi == rule152_s_expected_sia.phi


@pytest.mark.veryslow
def test_sia_rule152_s_parallel(rule152_s, rule152_s_expected_sia):
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
