#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import pytest
import numpy as np

from cyphi import constants, compute, models, utils, db, Network
from cyphi.constants import DIRECTIONS, PAST, FUTURE


# Precision for testing
PRECISION = 5

# Use a test database
db.collection = db.database.test


# Answers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

standard_answer = {
    'phi': 2.3125,
    'unpartitioned_small_phis': {
        (1,): 0.25,
        (2,): 0.5,
        (0, 1): 1/3,
        (0, 1, 2): 0.5
    },
    'len_partitioned_constellation': 1,
    'sum_partitioned_small_phis': 0.5,
    'cut': models.Cut(severed=(1, 2), intact=(0,))
}


noised_answer = {
    'phi': 1.928588,
    'unpartitioned_small_phis': {
        (0,): 0.0625,
        (1,): 0.2,
        (2,): 0.316326,
        (0, 1): 0.31904700000000025,
        (0, 2): 0.0125,
        (1, 2): 0.26384703498300066,
        (0, 1, 2): 0.34999965000000005
    },
    'len_partitioned_constellation': 7,
    'sum_partitioned_small_phis': 0.50491,
    'cut': models.Cut(severed=(1, 2), intact=(0,))
}


big_answer = {
    'phi': 10.729482,
    'unpartitioned_small_phis': {
        (0,): 0.24999975000000002,
        (1,): 0.24999975000000002,
        (2,): 0.24999975000000002,
        (3,): 0.24999975000000002,
        (4,): 0.24999975000000002,
        (0, 1): 0.19999980000000003,
        (0, 2): 0.2000000000000017,
        (0, 3): 0.2000000000000017,
        (0, 4): 0.19999980000000003,
        (1, 2): 0.19999980000000003,
        (1, 3): 0.20000000000000057,
        (1, 4): 0.2000000000000017,
        (2, 3): 0.19999980000000003,
        (2, 4): 0.2000000000000017,
        (3, 4): 0.19999980000000003,
        (0, 1, 2): 0.2,
        (0, 1, 3): 0.257142871428,
        (0, 1, 4): 0.2,
        (0, 2, 3): 0.257142871428,
        (0, 2, 4): 0.257142871428,
        (0, 3, 4): 0.2,
        (1, 2, 3): 0.2,
        (1, 2, 4): 0.257142871428,
        (1, 3, 4): 0.257142871428,
        (2, 3, 4): 0.2,
        (0, 1, 2, 3): 0.18570900000000226,
        (0, 1, 2, 4): 0.18570900000000112,
        (0, 1, 3, 4): 0.18570900000000112,
        (0, 2, 3, 4): 0.18570900000000226,
        (1, 2, 3, 4): 0.18570900000000112
    },
    'len_partitioned_constellation': 17,
    'sum_partitioned_small_phis': 3.564907,
    'cut': models.Cut(severed=(0, 3), intact=(1, 2, 4))
}


big_subsys_0_thru_3_answer = {
    'phi': 0.3663872111473395,
    'unpartitioned_small_phis': {
        (0,): 0.166667,
        (1,): 0.166667,
        (2,): 0.166667,
        (3,): 0.24999975000000002,
        (0, 1): 0.133333,
        (1, 2): 0.133333
    },
    'len_partitioned_constellation': 5,
    'sum_partitioned_small_phis': 0.883334,
    'cut': models.Cut(severed=(1, 3), intact=(0, 2))
}


rule152_answer = {
    'phi': 6.9749327784596415,
    'unpartitioned_small_phis': {
        (0,): 0.125001874998,
        (1,): 0.125001874998,
        (2,): 0.125001874998,
        (3,): 0.125001874998,
        (4,): 0.125001874998,
        (0, 1): 0.25,
        (0, 2): 0.18461400000000092,
        (0, 3): 0.18461400000000092,
        (0, 4): 0.25,
        (1, 2): 0.25,
        (1, 3): 0.18461400000000092,
        (1, 4): 0.18461400000000092,
        (2, 3): 0.25,
        (2, 4): 0.18461400000000092,
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
        (0, 1, 2, 3, 4): 0.25
    },
    'len_partitioned_constellation': 24,
    'sum_partitioned_small_phis': 4.185364469227005,
    'cut': models.Cut(severed=(0, 1, 3, 4), intact=(2,))
}


# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _flushdb():
    return db.collection.remove({})


@pytest.fixture
def flushdb():
    return _flushdb()


def check_unpartitioned_small_phis(small_phis, unpartitioned_constellation):
    assert len(small_phis) == len(unpartitioned_constellation)
    for c in unpartitioned_constellation:
        np.testing.assert_almost_equal(
            c.phi,
            small_phis[utils.nodes2indices(c.mechanism)],
            PRECISION)


def check_partitioned_small_phis(answer, partitioned_constellation):
    if 'len_partitioned_constellation' in answer:
        assert (answer['len_partitioned_constellation'] ==
                len(partitioned_constellation))
    if 'sum_partitioned_small_phis' in answer:
        np.testing.assert_almost_equal(
            sum(c.phi for c in partitioned_constellation),
            answer['sum_partitioned_small_phis'],
            PRECISION)


def check_mip(mip, answer):
    # Check big phi value.
    np.testing.assert_almost_equal(mip.phi, answer['phi'], PRECISION)
    # Check small phis of unpartitioned constellation.
    check_unpartitioned_small_phis(answer['unpartitioned_small_phis'],
                                   mip.unpartitioned_constellation)
    # Check sum of small phis of partitioned constellation if answer is
    # available.
    check_partitioned_small_phis(answer, mip.partitioned_constellation)
    # Check cut.
    if 'cut' in answer:
        assert mip.cut == answer['cut']


# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_null_concept(s, flushdb):
    cause = models.Mice(models.Mip(
        unpartitioned_repertoire=s.unconstrained_cause_repertoire(s.nodes),
        phi=0, direction=DIRECTIONS[PAST], mechanism=(), purview=s.nodes,
        partition=None, partitioned_repertoire=None))
    effect = models.Mice(models.Mip(
        unpartitioned_repertoire=s.unconstrained_effect_repertoire(s.nodes),
        phi=0, direction=DIRECTIONS[FUTURE], mechanism=(), purview=s.nodes,
        partition=None, partitioned_repertoire=None))
    assert (s.null_concept ==
            models.Concept(mechanism=(), phi=0, cause=cause, effect=effect,
                           subsystem=s))


def test_concept_nonexistent(s, flushdb):
    assert not compute.concept(s, (s.nodes[0], s.nodes[2]))


def test_conceptual_information(s, flushdb):
    np.testing.assert_almost_equal(compute.conceptual_information(s), 2.812497,
                                   PRECISION)


def test_big_mip_empty_subsystem(s_empty, flushdb):
    assert (compute.big_mip.func(hash(s_empty), s_empty) ==
            models.BigMip(phi=0.0,
                          unpartitioned_constellation=[],
                          partitioned_constellation=[],
                          subsystem=s_empty,
                          cut_subsystem=s_empty))


def test_big_mip_disconnected_network(reducible, flushdb):
    assert (compute.big_mip.func(hash(reducible), reducible) ==
            models.BigMip(subsystem=reducible, cut_subsystem=reducible,
                          phi=0.0, unpartitioned_constellation=[],
                          partitioned_constellation=[]))


def test_big_mip_wrappers(reducible, flushdb):
    assert (compute.big_mip(reducible) ==
            models.BigMip(subsystem=reducible, cut_subsystem=reducible,
                          phi=0.0, unpartitioned_constellation=[],
                          partitioned_constellation=[]))
    assert compute.big_phi(reducible) == 0.0


def test_big_mip_single_node(s_single, flushdb):
    initial_option = constants.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI
    constants.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI = True
    assert compute.big_mip.func(hash(s_single), s_single).phi == 0.5
    constants.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI = False
    assert compute.big_mip.func(hash(s_single), s_single).phi == 0.0
    constants.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI = initial_option


def test_big_mip_standard_example_sequential(s, flushdb):
    initial = constants.NUMBER_OF_CORES
    constants.NUMBER_OF_CORES = 1

    mip = compute.big_mip.func(hash(s), s)
    check_mip(mip, standard_answer)

    constants.NUMBER_OF_CORES = initial


def test_big_mip_standard_example_parallel(s, flushdb):
    initial = constants.NUMBER_OF_CORES
    constants.NUMBER_OF_CORES = -2

    mip = compute.big_mip.func(hash(s), s)
    check_mip(mip, standard_answer)

    constants.NUMBER_OF_CORES = initial


def test_big_mip_noised_example_sequential(s_noised, flushdb):
    initial = constants.NUMBER_OF_CORES
    constants.NUMBER_OF_CORES = 1

    mip = compute.big_mip.func(hash(s_noised), s_noised)
    check_mip(mip, noised_answer)

    constants.NUMBER_OF_CORES = initial


def test_big_mip_noised_example_parallel(s_noised, flushdb):
    initial = constants.NUMBER_OF_CORES
    constants.NUMBER_OF_CORES = -2

    mip = compute.big_mip.func(hash(s_noised), s_noised)
    check_mip(mip, noised_answer)

    constants.NUMBER_OF_CORES = initial


# TODO!! add more assertions for the smaller subsystems
def test_complexes_standard(standard, flushdb):
    complexes = list(compute.complexes(standard))
    check_mip(complexes[7], standard_answer)


@pytest.mark.slow
def test_big_mip_big_network(big_subsys_all, flushdb):
    mip = compute.big_mip.func(hash(big_subsys_all), big_subsys_all)
    check_mip(mip, big_answer)


@pytest.mark.slow
def test_big_mip_big_network_0_thru_3(big_subsys_0_thru_3, flushdb):
    mip = compute.big_mip.func(hash(big_subsys_0_thru_3), big_subsys_0_thru_3)
    check_mip(mip, big_subsys_0_thru_3_answer)


@pytest.mark.slow
def test_big_mip_rule152(rule152_s, flushdb):
    mip = compute.big_mip.func(hash(rule152_s), rule152_s)
    check_mip(mip, rule152_answer)


def test_concept_normalization(standard, flushdb):
    # Compute each subsystem using accumulated precomputed results.
    db_complexes = list(compute.complexes(standard))
    # Compute each subsystem with an empty db.
    no_db_complexes = []
    for subsystem in compute.subsystems(standard):
        _flushdb()
        no_db_complexes.append(compute.big_mip(subsystem))
    # The empty-db and full-db results should be the same if concept caching is
    # working properly.
    assert db_complexes == no_db_complexes
