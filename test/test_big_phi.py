#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from cyphi import options, compute, models, utils

import example_networks as examples


# Precision for testing
PRECISION = 4
# Expected standard example phi value
STANDARD_EXAMPLE_PHI = 2.3125
# Compute the BigMip of the standard example full subsystem, once
# Don't interact with cache
big_mip_standard_example = compute.big_mip.func(examples.s(),
                                                hash(examples.s()))


def test_null_concept(s):
    assert (s.null_concept == models.Concept(
        mechanism=(),
        location=np.array([s.unconstrained_cause_repertoire(s.nodes),
                           s.unconstrained_effect_repertoire(s.nodes)]),
        phi=0, cause=None, effect=None))


def test_concept_nonexistent(s):
    assert not compute.concept(s, (s.nodes[0], s.nodes[2]))


def test_big_phi_standard_example_sequential(s):
    initial = options.PARALLEL_CUT_EVALUATION
    options.PARALLEL_CUT_EVALUATION = False
    phi = compute.big_mip.func(s, hash(s)).phi
    np.testing.assert_almost_equal(phi, STANDARD_EXAMPLE_PHI, PRECISION)
    options.PARALLEL_CUT_EVALUATION = initial


def test_big_phi_standard_example_parallel(s):
    initial = options.PARALLEL_CUT_EVALUATION
    options.PARALLEL_CUT_EVALUATION = True
    phi = compute.big_mip.func(s, hash(s)).phi
    np.testing.assert_almost_equal(phi, STANDARD_EXAMPLE_PHI, PRECISION)
    options.PARALLEL_CUT_EVALUATION = initial


def test_standard_example(s):
    mip = big_mip_standard_example
    np.testing.assert_almost_equal(
        sum(C.phi for C in mip.unpartitioned_constellation),
        1.5833,
        PRECISION)
    np.testing.assert_almost_equal(
        sum(c.phi for c in mip.partitioned_constellation),
        0.5)
    assert len(mip.unpartitioned_constellation) == 4
    assert len(mip.partitioned_constellation) == 1


@pytest.mark.slow
def test_big_mip_big_network(big_subsys_all):
    initial_precision = options.PRECISION
    options.PRECISION = 4
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
    options.PRECISION = initial_precision


def test_big_mip_empty_subsystem(s_empty):
    assert (compute.big_mip.func(s_empty, hash(s_empty)) ==
            models.BigMip(phi=0.0,
                          cut=models.Cut(severed=(), intact=()),
                          unpartitioned_constellation=[],
                          partitioned_constellation=[],
                          subsystem=s_empty))
