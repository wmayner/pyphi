#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

import example_networks
from pyphi import Direction, constants
from pyphi.models import Part, Bipartition, RepertoireIrreducibilityAnalysis

s = example_networks.s()


# `find_mip` tests {{{
# ====================

# Test scenario structure:
#
# (
#     direction of RIA (Direction.CAUSE or Direction.EFFECT),
#     subsystem, cut,
#     mechanism,
#     purview,
#     expected result
# )
#
# where 'expected result' is a dictionary with the following structure:
#
# {'partitions':
#     {
#         (first part, second part): expected partitioned repertoire
#     },
#  'phi': expected phi value
# }

scenarios = [
# Previous {{{
# ~~~~~~~~
    # No cut {{{
    # ----------
    (
        Direction.CAUSE,
        s, None,
        (0,),
        (0,),
        {'partitions': {
            Bipartition(Part(mechanism=(), purview=(0,)),
                        Part(mechanism=(0,), purview=())):
                np.array([0.5, 0.5]).reshape(2, 1, 1, order="F")
         },
         'repertoire':
            np.array([0.5, 0.5]).reshape(2, 1, 1, order="F"),
         'phi': 0.0}
    ),
    # }}}
    # With cut {{{
    # ------------
    (
        Direction.CAUSE,
        s, (0, (1, 2)),
        (1,),
        (2,),
        {'partitions': {
            Bipartition(Part(mechanism=(), purview=(2,)),
                        Part(mechanism=(1,), purview=())):
                np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
         },
         'repertoire':
            np.array([1., 0.]).reshape(1, 1, 2),
         'phi': 0.5}
    ),
    # }}}
# }}}
# Next {{{
# ~~~~~~~~~~
    # No cut {{{
    # ----------
    (
        Direction.EFFECT,
        s, None,
        (0, 1, 2),
        (0, 1, 2),
        {'partitions': {
            # Any of these partitions is valid; there is no well-defined way of
            # breaking ties
            Bipartition(Part(mechanism=(2,), purview=()),
                        Part(mechanism=(0, 1), purview=(0, 1, 2))):
                np.array([0., 0., 0.5, 0.5, 0., 0., 0., 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(), purview=(0,)),
                        Part(mechanism=(0, 1, 2), purview=(1, 2))):
                np.array([0., 0., 0.5, 0.5, 0., 0., 0., 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(2,), purview=(0,)),
                        Part(mechanism=(0, 1), purview=(1, 2))):
                np.array([0., 0., 0.5, 0.5, 0., 0., 0., 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(0,), purview=()),
                        Part(mechanism=(1, 2), purview=(0, 1, 2))):
                np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(), purview=(1,)),
                        Part(mechanism=(0, 1, 2), purview=(0, 2))):
                np.array([0., 0., 0., 0., 0.5, 0., 0.5, 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(2,), purview=(1,)),
                        Part(mechanism=(0, 1), purview=(0, 2))):
                np.array([0., 0., 0., 0., 0.5, 0.5, 0., 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(), purview=(2,)),
                        Part(mechanism=(0, 1, 2), purview=(0, 1))):
                np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(0,), purview=(2,)),
                        Part(mechanism=(1, 2), purview=(0, 1))):
                np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0.]).reshape(
                    2, 2, 2, order="F"),
            Bipartition(Part(mechanism=(2,), purview=(0, 1)),
                        Part(mechanism=(0, 1), purview=(2,))):
                np.array([0., 0., 0., 0., 0.5, 0.5, 0., 0.]).reshape(
                    2, 2, 2, order="F"),
         },
         'repertoire':
            np.array([0., 1., 0., 0., 0., 0., 0., 0.]).reshape(2, 2, 2),
         'phi': 0.5}
    ),
    # }}}

    # With cut {{{
    # ------------
    (
        Direction.EFFECT,
        s, ((1, 2), 0),
        (2,),
        (1,),
        {'partitions': {
            Bipartition(Part(mechanism=(), purview=(1,)),
                        Part(mechanism=(2,), purview=())):
                np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
         },
         'repertoire':
            np.array([1., 0.]).reshape(1, 2, 1),
         'phi': 0.5}
    ), (
        Direction.EFFECT,
        s, ((0, 2), 1),
        (2,),
        (0,),
        {'partitions': {
            Bipartition(Part(mechanism=(), purview=(0,)),
                        Part(mechanism=(2,), purview=())):
                np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
         },
         'repertoire':
            np.array([0.5, 0.5]).reshape(2, 1, 1),
         'phi': 0.25}
    ), (
        Direction.EFFECT,
        s, ((0, 2), 1),
        (0, 1, 2),
        (0, 2),
        {'partitions': {
            # Any of these partitions is valid; there is no well-defined way of
            # breaking ties
            Bipartition(Part(mechanism=(0,), purview=()),
                        Part(mechanism=(1, 2), purview=(0, 2))):
                np.array([0.5, 0., 0.5, 0.]).reshape(2, 1, 2, order="F"),
            Bipartition(Part(mechanism=(2,), purview=()),
                        Part(mechanism=(0, 1), purview=(0, 2))):
                np.array([0., 0., 0.5, 0.5]).reshape(2, 1, 2, order="F"),
            Bipartition(Part(mechanism=(2,), purview=(0,)),
                        Part(mechanism=(0, 1), purview=(2,))):
                np.array([0., 0., 0.5, 0.5]).reshape(2, 1, 2, order="F"),
            Bipartition(Part(mechanism=(), purview=(2,)),
                        Part(mechanism=(0, 1, 2), purview=(0,))):
                np.array([0.5, 0., 0.5, 0.]).reshape(2, 1, 2, order="F"),
            Bipartition(Part(mechanism=(0,), purview=(2,)),
                        Part(mechanism=(1, 2), purview=(0,))):
                np.array([0.5, 0., 0.5, 0.]).reshape(2, 1, 2, order="F")
        },
         'repertoire':
             np.array([0., 1., 0., 0.]).reshape(2, 1, 2),
        'phi': 0.5}
    ), (
        Direction.EFFECT,
        s, ((0, 1), 2),
        (1,),
        (0,),
        {'partitions': {
            Bipartition(Part(mechanism=(), purview=(0,)),
                        Part(mechanism=(1,), purview=())):
                np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
         },
         'repertoire':
            np.array([0.5, 0.5]).reshape(2, 1, 1),
         'phi': 0.25}
    )
    # }}}
# }}}
]
parameter_string = "direction,subsystem,cut,mechanism,purview,expected"


@pytest.mark.parametrize(parameter_string, scenarios)
def test_find_mip(direction, subsystem, cut, mechanism, purview, expected):
    # Set up testing parameters from scenario
    result = subsystem.find_mip(direction, mechanism, purview)

    # IMPORTANT: Since several different ways of partitioning the system can
    # yield the same phi value, the partition used in finding the MIP is not
    # unique. Thus, ``expected['partitions']`` is a dictionary that maps all
    # the ways of partitioning the system that yeild the minimal phi value to
    # their expected partitioned repertoires.

    if expected:
        # Construct expected list of possible MIPs
        expected = [
            RepertoireIrreducibilityAnalysis(
                direction=direction,
                partition=expected_partition,
                mechanism=mechanism,
                purview=purview,
                repertoire=expected['repertoire'],
                partitioned_repertoire=expected_partitioned_repertoire,
                phi=expected['phi']
            )
            for expected_partition, expected_partitioned_repertoire
            in expected['partitions'].items()
        ]

    print('Result:', '---------', '', result, '', sep='\n')
    print('Expected:',  '---------', '', sep='\n')
    if expected:
        print(*[mip for mip in expected], sep='\n')
    else:
        print(expected)
    print('\n')

    if expected:
        assert result in expected
    else:
        assert result == expected

# }}}


# Wrapper method tests {{{
# ========================


def test_cause_mip(s):
    mechanism = s.node_indices
    purview = s.node_indices
    mip_cause = s.find_mip(Direction.CAUSE, mechanism, purview)
    assert mip_cause == s.cause_mip(mechanism, purview)


def test_effect_mip(s):
    mechanism = s.node_indices
    purview = s.node_indices
    mip_effect = s.find_mip(Direction.EFFECT, mechanism, purview)
    assert mip_effect == s.effect_mip(mechanism, purview)


def test_phi_cause_mip(s):
    mechanism = s.node_indices
    purview = s.node_indices
    assert (s.phi_cause_mip(mechanism, purview) ==
            s.cause_mip(mechanism, purview).phi)


def test_phi_cause_mip_reducible(s):
    mechanism = (1,)
    purview = (0,)
    assert s.phi_cause_mip(mechanism, purview) == 0


def test_phi_effect_mip(s):
    mechanism = s.node_indices
    purview = s.node_indices
    assert (s.phi_effect_mip(mechanism, purview) ==
            s.effect_mip(mechanism, purview).phi)


def test_phi_effect_mip_reducible(s):
    mechanism = (0, 1)
    purview = (1, )
    assert s.phi_effect_mip(mechanism, purview) == 0


def test_phi(s):
    mechanism = s.node_indices
    purview = s.node_indices
    assert abs(0.5 - s.phi(mechanism, purview)) < constants.EPSILON

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
