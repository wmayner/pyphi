#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from cyphi import constants, compute, models, utils, db
from cyphi.constants import DIRECTIONS, PAST, FUTURE


# Precision for testing
PRECISION = 4
# Expected standard example phi value
STANDARD_EXAMPLE_PHI = 2.3125

# Use a test database
db.collection = db.database.test


# Helpers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# TODO: use different db for tests
@pytest.fixture
def flushdb():
    return db.collection.remove({})


def standard_example_is_correct(mip):
    # Check that the given MIP is the correct output for the standard example
    # (full subsystem)
    np.testing.assert_almost_equal(mip.phi, STANDARD_EXAMPLE_PHI, PRECISION)
    np.testing.assert_almost_equal(
        sum(C.phi for C in mip.unpartitioned_constellation),
        1.5833,
        PRECISION)
    np.testing.assert_almost_equal(
        sum(c.phi for c in mip.partitioned_constellation),
        0.5)
    assert len(mip.unpartitioned_constellation) == 4
    assert len(mip.partitioned_constellation) == 1
    return True


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
                          subsystem=s_empty))


def test_big_mip_disconnected_network(reducible, flushdb):
    assert (compute.big_mip.func(hash(reducible), reducible) ==
            models.BigMip(subsystem=reducible, phi=0.0,
                          unpartitioned_constellation=[],
                          partitioned_constellation=[]))


def test_big_mip_wrappers(reducible, flushdb):
    assert (compute.big_mip(reducible) ==
            models.BigMip(subsystem=reducible, phi=0.0,
                          unpartitioned_constellation=[],
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
    assert standard_example_is_correct(mip)
    constants.NUMBER_OF_CORES = initial


def test_big_mip_standard_example_parallel(s, flushdb):
    initial = constants.NUMBER_OF_CORES
    constants.NUMBER_OF_CORES = -2
    mip = compute.big_mip.func(hash(s), s)
    assert standard_example_is_correct(mip)
    constants.NUMBER_OF_CORES = initial


@pytest.mark.slow
def test_big_mip_big_network(big_subsys_all, flushdb):
    initial_precision = constants.PRECISION
    constants.PRECISION = 4
    mip = compute.big_mip(big_subsys_all)
    assert utils.phi_eq(mip.phi, 10.744578)
    assert utils.phi_eq(
        sum(C.phi for C in mip.unpartitioned_constellation),
        6.464257107140015)
    assert utils.phi_eq(
        sum(c.phi for c in mip.partitioned_constellation),
        3.564907)
    assert len(mip.unpartitioned_constellation) == 30
    assert len(mip.partitioned_constellation) == 17
    constants.PRECISION = initial_precision


def test_big_mip_big_network_0_thru_3(big_subsys_0_thru_3, flushdb):
    initial_precision = constants.PRECISION
    constants.PRECISION = 4
    mip = compute.big_mip(big_subsys_0_thru_3)
    assert utils.phi_eq(mip.phi, 0.3663872111473395)
    assert utils.phi_eq(
        sum(C.phi for C in mip.unpartitioned_constellation),
        1.0166667500000002)
    assert utils.phi_eq(
        sum(c.phi for c in mip.partitioned_constellation),
        0.883334)
    assert len(mip.unpartitioned_constellation) == 6
    assert len(mip.partitioned_constellation) == 5
    constants.PRECISION = initial_precision


# TODO!! add more assertions for the smaller subsystems
def test_complexes(standard, flushdb):
    complexes = list(compute.complexes(standard))
    assert standard_example_is_correct(complexes[7])


def test_concept_normalization(standard, flushdb):
    # Compute each subsystem using accumulated precomputed results.
    db_complexes = list(compute.complexes(standard))
    # Compute each subsystem with an empty db.
    no_db_complexes = []
    for subsystem in compute.subsystems(standard):
        flushdb()
        no_db_complexes.append(compute.big_mip(subsystem))
    # The empty-db and full-db results should be the same if concept caching is
    # working properly.
    assert db_complexes == no_db_complexes
