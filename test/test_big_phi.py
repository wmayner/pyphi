import pytest

from pyphi import config, compute, models

# pylint: disable=unused-argument

# Answers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

standard_answer = {
    "phi": 2.3125,
    "unpartitioned_small_phis": {
        (1,): 0.25,
        (2,): 0.5,
        (0, 1): 0.333333,
        (0, 1, 2): 0.5,
    },
    "len_partitioned_ces": 1,
    "sum_partitioned_small_phis": 0.5,
    "cut": models.Cut(from_nodes=(1, 2), to_nodes=(0,)),
}


noised_answer = {
    "phi": 1.928592,
    "unpartitioned_small_phis": {
        (0,): 0.0625,
        (1,): 0.2,
        (2,): 0.316326,
        (0, 1): 0.319047,
        (0, 2): 0.0125,
        (1, 2): 0.263847,
        (0, 1, 2): 0.35,
    },
    "len_partitioned_ces": 7,
    "sum_partitioned_small_phis": 0.504906,
    "cut": models.Cut(from_nodes=(1, 2), to_nodes=(0,)),
}


big_answer = {
    "phi": 10.729491,
    "unpartitioned_small_phis": {
        (0,): 0.25,
        (1,): 0.25,
        (2,): 0.25,
        (3,): 0.25,
        (4,): 0.25,
        (0, 1): 0.2,
        (0, 2): 0.2,
        (0, 3): 0.2,
        (0, 4): 0.2,
        (1, 2): 0.2,
        (1, 3): 0.2,
        (1, 4): 0.2,
        (2, 3): 0.2,
        (2, 4): 0.2,
        (3, 4): 0.2,
        (0, 1, 2): 0.2,
        (0, 1, 3): 0.257143,
        (0, 1, 4): 0.2,
        (0, 2, 3): 0.257143,
        (0, 2, 4): 0.257143,
        (0, 3, 4): 0.2,
        (1, 2, 3): 0.2,
        (1, 2, 4): 0.257143,
        (1, 3, 4): 0.257143,
        (2, 3, 4): 0.2,
        (0, 1, 2, 3): 0.185709,
        (0, 1, 2, 4): 0.185709,
        (0, 1, 3, 4): 0.185709,
        (0, 2, 3, 4): 0.185709,
        (1, 2, 3, 4): 0.185709,
    },
    "len_partitioned_ces": 17,
    "sum_partitioned_small_phis": 3.564909,
    "cut": models.Cut(from_nodes=(2, 4), to_nodes=(0, 1, 3)),
}


big_subsys_0_thru_3_answer = {
    "phi": 0.366389,
    "unpartitioned_small_phis": {
        (0,): 0.166667,
        (1,): 0.166667,
        (2,): 0.166667,
        (3,): 0.25,
        (0, 1): 0.133333,
        (1, 2): 0.133333,
    },
    "len_partitioned_ces": 5,
    "sum_partitioned_small_phis": 0.883334,
    "cut": models.Cut(from_nodes=(1, 3), to_nodes=(0, 2)),
}


rule152_answer = {
    "phi": 6.952286,
    "unpartitioned_small_phis": {
        (0,): 0.125,
        (1,): 0.125,
        (2,): 0.125,
        (3,): 0.125,
        (4,): 0.125,
        (0, 1): 0.25,
        (0, 2): 0.184614,
        (0, 3): 0.184614,
        (0, 4): 0.25,
        (1, 2): 0.25,
        (1, 3): 0.184614,
        (1, 4): 0.184614,
        (2, 3): 0.25,
        (2, 4): 0.184614,
        (3, 4): 0.25,
        (0, 1, 2): 0.25,
        (0, 1, 3): 0.316666,
        (0, 1, 4): 0.25,
        (0, 2, 3): 0.316666,
        (0, 2, 4): 0.316666,
        (0, 3, 4): 0.25,
        (1, 2, 3): 0.25,
        (1, 2, 4): 0.316666,
        (1, 3, 4): 0.316666,
        (2, 3, 4): 0.25,
        (0, 1, 2, 3): 0.25,
        (0, 1, 2, 4): 0.25,
        (0, 1, 3, 4): 0.25,
        (0, 2, 3, 4): 0.25,
        (1, 2, 3, 4): 0.25,
        (0, 1, 2, 3, 4): 0.25,
    },
    "len_partitioned_ces": 24,
    "sum_partitioned_small_phis": 4.185363,
    "cuts": [
        models.Cut(from_nodes=(0, 1, 2, 3), to_nodes=(4,)),
        models.Cut(from_nodes=(0, 1, 2, 4), to_nodes=(3,)),
        models.Cut(from_nodes=(0, 1, 3, 4), to_nodes=(2,)),
        models.Cut(from_nodes=(0, 2, 3, 4), to_nodes=(1,)),
        models.Cut(from_nodes=(1, 2, 3, 4), to_nodes=(0,)),
        # TODO: are there other possible cuts?
    ],
}


micro_answer = {
    "phi": 0.974411,
    "unpartitioned_small_phis": {
        (0,): 0.175,
        (1,): 0.175,
        (2,): 0.175,
        (3,): 0.175,
        (0, 1): 0.348114,
        (2, 3): 0.348114,
    },
    "cuts": [
        models.Cut(from_nodes=(0, 2), to_nodes=(1, 3)),
        models.Cut(from_nodes=(1, 2), to_nodes=(0, 3)),
        models.Cut(from_nodes=(0, 3), to_nodes=(1, 2)),
        models.Cut(from_nodes=(1, 3), to_nodes=(0, 2)),
    ],
}

macro_answer = {
    "phi": 0.86905,
    "unpartitioned_small_phis": {
        (0,): 0.455,
        (1,): 0.455,
    },
    "cuts": [
        models.Cut(from_nodes=(0,), to_nodes=(1,)),
        models.Cut(from_nodes=(1,), to_nodes=(0,)),
    ],
}


# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def check_unpartitioned_small_phis(small_phis, ces):
    assert len(small_phis) == len(ces)
    for c in ces:
        assert c.phi == small_phis[c.mechanism]


def check_partitioned_small_phis(answer, partitioned_ces):
    if "len_partitioned_ces" in answer:
        assert answer["len_partitioned_ces"] == len(partitioned_ces)
    if "sum_partitioned_small_phis" in answer:
        assert (
            round(sum(c.phi for c in partitioned_ces), config.PRECISION)
            == answer["sum_partitioned_small_phis"]
        )


def check_sia(sia, answer):
    # Check big phi value.
    assert sia.phi == answer["phi"]
    # Check small phis of unpartitioned CES.
    check_unpartitioned_small_phis(answer["unpartitioned_small_phis"], sia.ces)
    # Check sum of small phis of partitioned CES if answer is
    # available.
    check_partitioned_small_phis(answer, sia.partitioned_ces)
    # Check cut.
    if "cut" in answer:
        assert sia.cut == answer["cut"]
    elif "cuts" in answer:
        assert sia.cut in answer["cuts"]


# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_clear_subsystem_caches_after_computing_sia_config_option(use_iit_3_config, s):
    with config.override(
        CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=False,
        PARALLEL_CONCEPT_EVALUATION=False,
        PARALLEL_CUT_EVALUATION=False,
        CACHE_REPERTOIRES=True,
    ):
        sia = compute.subsystem.sia(s)
        assert s._repertoire_cache.cache

    with config.override(
        CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA=True,
        PARALLEL_CONCEPT_EVALUATION=False,
        PARALLEL_CUT_EVALUATION=False,
        CACHE_REPERTOIRES=True,
    ):
        sia = compute.subsystem.sia(s)
        assert not s._repertoire_cache.cache


def test_conceptual_info(s):
    assert compute.subsystem.conceptual_info(s) == 1.0


def test_sia_empty_subsystem(s_empty):
    assert compute.subsystem.sia(s_empty) == models.SystemIrreducibilityAnalysis(
        phi=0.0, ces=(), partitioned_ces=(), subsystem=s_empty, cut_subsystem=s_empty
    )


def test_sia_disconnected_network(reducible):
    assert compute.subsystem.sia(reducible) == models.SystemIrreducibilityAnalysis(
        subsystem=reducible,
        cut_subsystem=reducible,
        phi=0.0,
        ces=[],
        partitioned_ces=[],
    )


def test_sia_wrappers(reducible):
    assert compute.subsystem.sia(reducible) == models.SystemIrreducibilityAnalysis(
        subsystem=reducible,
        cut_subsystem=reducible,
        phi=0.0,
        ces=[],
        partitioned_ces=[],
    )
    assert compute.subsystem.phi(reducible) == 0.0


@config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=True)
@config.override(REPERTOIRE_DISTANCE="EMD")
def test_sia_single_micro_node_selfloops_have_phi(noisy_selfloop_single):
    assert compute.subsystem.sia(noisy_selfloop_single).phi == 0.2736


@config.override(SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI=False)
def test_sia_single_micro_node_selfloops_dont_have_phi(noisy_selfloop_single):
    assert compute.subsystem.sia(noisy_selfloop_single).phi == 0.0


def test_sia_single_micro_nodes_without_selfloops_dont_have_phi(s_single):
    assert compute.subsystem.sia(s_single).phi == 0.0


@pytest.fixture
def standard_ComputeSystemIrreducibility(s):
    ces = compute.subsystem.ces(s)
    cuts = compute.subsystem.sia_bipartitions(s.node_indices)
    return compute.subsystem.ComputeSystemIrreducibility(cuts, s, ces)


@config.override(PARALLEL_CUT_EVALUATION=False)
def test_find_sia_sequential_standard_example(standard_ComputeSystemIrreducibility):
    sia = standard_ComputeSystemIrreducibility.run_sequential()
    check_sia(sia, standard_answer)


@config.override(PARALLEL_CUT_EVALUATION=True, NUMBER_OF_CORES=-2)
def test_find_sia_parallel_standard_example(standard_ComputeSystemIrreducibility):
    sia = standard_ComputeSystemIrreducibility.run_parallel()
    check_sia(sia, standard_answer)


@pytest.fixture
def s_noised_ComputeSystemIrreducibility(s_noised):
    ces = compute.subsystem.ces(s_noised)
    cuts = compute.subsystem.sia_bipartitions(s_noised.node_indices)
    return compute.subsystem.ComputeSystemIrreducibility(cuts, s_noised, ces)


@config.override(PARALLEL_CUT_EVALUATION=False)
def test_find_sia_sequential_noised_example(s_noised_ComputeSystemIrreducibility):
    sia = s_noised_ComputeSystemIrreducibility.run_sequential()
    check_sia(sia, noised_answer)


@config.override(PARALLEL_CUT_EVALUATION=True, NUMBER_OF_CORES=-2)
def test_find_sia_parallel_noised_example(s_noised_ComputeSystemIrreducibility):
    sia = s_noised_ComputeSystemIrreducibility.run_parallel()
    check_sia(sia, noised_answer)


@pytest.fixture
def micro_s_ComputeSystemIrreducibility(micro_s):
    ces = compute.subsystem.ces(micro_s)
    cuts = compute.subsystem.sia_bipartitions(micro_s.node_indices)
    return compute.subsystem.ComputeSystemIrreducibility(cuts, micro_s, ces)


@config.override(PARALLEL_CUT_EVALUATION=True)
def test_find_sia_parallel_micro(micro_s_ComputeSystemIrreducibility):
    sia = micro_s_ComputeSystemIrreducibility.run_parallel()
    check_sia(sia, micro_answer)


@config.override(PARALLEL_CUT_EVALUATION=False)
def test_find_sia_sequential_micro(micro_s_ComputeSystemIrreducibility):
    sia = micro_s_ComputeSystemIrreducibility.run_sequential()
    check_sia(sia, micro_answer)


def test_sia_complete_graph_standard_example(use_iit_3_config, s_complete):
    sia = compute.subsystem.sia(s_complete)
    check_sia(sia, standard_answer)


def test_sia_complete_graph_s_noised(use_iit_3_config, s_noised):
    sia = compute.subsystem.sia(s_noised)
    check_sia(sia, noised_answer)


@pytest.mark.slow
def test_sia_complete_graph_big_subsys_all(big_subsys_all_complete):
    sia = compute.subsystem.sia(big_subsys_all_complete)
    check_sia(sia, big_answer)


@pytest.mark.slow
def test_sia_complete_graph_rule152_s(rule152_s_complete):
    sia = compute.subsystem.sia(rule152_s_complete)
    check_sia(sia, rule152_answer)


@pytest.mark.slow
def test_sia_big_network(big_subsys_all):
    sia = compute.subsystem.sia(big_subsys_all)
    check_sia(sia, big_answer)


def test_sia_big_network_0_thru_3(big_subsys_0_thru_3):
    sia = compute.subsystem.sia(big_subsys_0_thru_3)
    check_sia(sia, big_subsys_0_thru_3_answer)


@pytest.mark.slow
def test_sia_rule152(rule152_s):
    sia = compute.subsystem.sia(rule152_s)
    check_sia(sia, rule152_answer)


def test_sia_macro(macro_s):
    sia = compute.subsystem.sia(macro_s)
    check_sia(sia, macro_answer)


def test_sia_bipartitions():
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


def test_system_cut_styles(use_iit_3_config, s):
    with config.override(SYSTEM_CUTS="3.0_STYLE"):
        assert compute.subsystem.phi(s) == 2.3125

    with config.override(SYSTEM_CUTS="CONCEPT_STYLE"):
        assert compute.subsystem.phi(s) == 0.6875


@pytest.mark.parametrize("parallel", [False, True])
def test_ces_concepts_share_the_same_subsystem(parallel, s):
    with config.override(PARALLEL_CONCEPT_EVALUATION=parallel):
        ces = compute.subsystem.ces(s)
        for concept in ces:
            assert concept.subsystem is ces.subsystem


def test_parallel_and_sequential_ces_are_equal(s, micro_s, macro_s):
    with config.override(PARALLEL_CONCEPT_EVALUATION=False):
        c = compute.subsystem.ces(s)
        c_micro = compute.subsystem.ces(micro_s)
        c_macro = compute.subsystem.ces(macro_s)

    with config.override(PARALLEL_CONCEPT_EVALUATION=True):
        assert set(c) == set(compute.subsystem.ces(s))
        assert set(c_micro) == set(compute.subsystem.ces(micro_s))
        assert set(c_macro) == set(compute.subsystem.ces(macro_s))
