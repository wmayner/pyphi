#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pprint import pprint
import numpy as np
from cyphi import constants
from cyphi.utils import tuple_eq, mip_eq
from cyphi.models import Mip, Part


# `find_mip` tests {{{
# ====================

# Test scenario structure:
#
# (
#     direction of MIP ('past' or 'future'),
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
# Past {{{
# ~~~~~~~~
    # No cut {{{
    # ----------
    (
        'past',
        'subsys_all', None,
        [0],
        [0],
        None,
    ),
    # }}}
    # With cut {{{
    # ------------
    (
        'past',
        'subsys_all', (0, (1, 2)),
        [1],
        [2],
        {'partitions': {
            (Part(mechanism=(), purview=(2,)),
             Part(mechanism=(1,), purview=())):
                np.array([0.5, 0.5]).reshape(1, 1, 2, order="F")
         },
         'unpartitioned_repertoire':
            np.array([1., 0.]).reshape(1, 1, 2),
         'phi': 0.5}
    ),
    # }}}
# }}}
# Future {{{
# ~~~~~~~~~~
    # No cut {{{
    # ----------
    (
        'future',
        'subsys_all', None,
        [0, 1, 2],
        [0, 1, 2],
        {'partitions': {
            # Any of these partitions is valid; there is no well-defined way of
            # breaking ties
            (Part(mechanism=(2,), purview=()),
             Part(mechanism=(0, 1), purview=(0, 1, 2))):
                np.array([0., 0., 0.5, 0.5, 0., 0., 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(), purview=(0,)),
             Part(mechanism=(0, 1, 2), purview=(1, 2))):
                np.array([0., 0., 0.5, 0.5, 0., 0., 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(2,), purview=(0,)),
             Part(mechanism=(0, 1), purview=(1, 2))):
                np.array([0., 0., 0.5, 0.5, 0., 0., 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(0,), purview=()),
             Part(mechanism=(1, 2), purview=(0, 1, 2))):
                np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(), purview=(1,)),
             Part(mechanism=(0, 1, 2), purview=(0, 2))):
                np.array([0., 0., 0., 0., 0.5, 0., 0.5, 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(2,), purview=(1,)),
             Part(mechanism=(0, 1), purview=(0, 2))):
                np.array([0., 0., 0., 0., 0.5, 0.5, 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(), purview=(2,)),
             Part(mechanism=(0, 1, 2), purview=(0, 1))):
                np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(0,), purview=(2,)),
             Part(mechanism=(1, 2), purview=(0, 1))):
                np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
            (Part(mechanism=(2,), purview=(0, 1)),
             Part(mechanism=(0, 1), purview=(2,))):
                np.array([0., 0., 0., 0., 0.5, 0.5, 0., 0.]).reshape(2,2,2,
                                                                     order="F"),
         },
         'unpartitioned_repertoire':
            np.array([0., 1., 0., 0., 0., 0., 0., 0.]).reshape(2, 2, 2),
         'phi': 0.5}
    ),
    # }}}

    # With cut {{{
    # ------------
    (
        'future',
        'subsys_all', ((1, 2), 0),
        [2],
        [1],
        {'partitions': {
            (Part(mechanism=(), purview=(1,)),
             Part(mechanism=(2,), purview=())):
                np.array([0.5, 0.5]).reshape(1, 2, 1, order="F")
         },
         'unpartitioned_repertoire':
            np.array([1., 0.]).reshape(1, 2, 1),
         'phi': 0.5}
    ), (
        'future',
        'subsys_all', ((0, 2), 1),
        [2],
        [0],
        {'partitions': {
            (Part(mechanism=(), purview=(0,)),
             Part(mechanism=(2,), purview=())):
                np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
         },
         'unpartitioned_repertoire':
            np.array([0.5, 0.5]).reshape(2, 1, 1),
         'phi': 0.25}
    ), (
        'future',
        'subsys_all', ((0, 2), 1),
        [0, 1, 2],
        [0, 2],
        {'partitions': {
            # Any of these partitions is valid; there is no well-defined way of
            # breaking ties
            (Part(mechanism=(0,), purview=()),
             Part(mechanism=(1, 2), purview=(0, 2))):
                np.array([0.5, 0., 0.5, 0.]).reshape(2, 1, 2, order="F"),
            (Part(mechanism=(2,), purview=()),
             Part(mechanism=(0, 1), purview=(0, 2))):
                np.array([0., 0., 0.5, 0.5]).reshape(2, 1, 2, order="F"),
            (Part(mechanism=(2,), purview=(0,)),
             Part(mechanism=(0, 1), purview=(2,))):
                np.array([0., 0., 0.5, 0.5]).reshape(2, 1, 2, order="F"),
            (Part(mechanism=(), purview=(2,)),
             Part(mechanism=(0, 1, 2), purview=(0,))):
                np.array([0.5, 0., 0.5, 0.]).reshape(2, 1, 2, order="F"),
            (Part(mechanism=(0,), purview=(2,)),
             Part(mechanism=(1, 2), purview=(0,))):
                np.array([0.5, 0., 0.5, 0.]).reshape(2, 1, 2, order="F")
        },
         'unpartitioned_repertoire':
            np.array([0., 1., 0., 0.]).reshape(2, 1, 2),
        'phi': 0.5}
    ), (
        'future',
        'subsys_all', ((0, 1), 2),
        [1],
        [0],
        {'partitions': {
            (Part(mechanism=(), purview=(0,)),
             Part(mechanism=(1,), purview=())):
                np.array([0.25, 0.75]).reshape(2, 1, 1, order="F")
         },
         'unpartitioned_repertoire':
            np.array([0.5, 0.5]).reshape(2, 1, 1),
         'phi': 0.25}
    )
    # }}}
# }}}
]
parameter_string = "direction,subsystem,cut,mechanism,purview,expected"


@pytest.mark.parametrize(parameter_string, scenarios)
def test_find_mip(m, direction, subsystem, cut, mechanism, purview, expected):
    # Set up testing parameters from scenario
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mechanism = [m.nodes[index] for index in mechanism]
    purview = [m.nodes[index] for index in purview]
    subsystem = getattr(m, subsystem)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    result = subsystem.find_mip(direction, mechanism, purview)

    # IMPORTANT: Since several different ways of partitioning the system can
    # yield the same phi value, the partition used in finding the MIP is not
    # unique. Thus, expected['partitions'] is a dictionary that maps all the
    # ways of partitioning the system that yeild the minimal phi value to their
    # expected partitioned repertoires.

    if expected:
        # Construct expected list of possible MIPs
        expected = [
            Mip(direction=direction,
                partition=(
                    Part(mechanism=tuple(m.nodes[i] for i in
                                         expected_partition[0].mechanism),
                         purview=tuple(m.nodes[i] for i in
                                       expected_partition[0].purview)),
                    Part(mechanism=tuple(m.nodes[i] for i in
                                         expected_partition[1].mechanism),
                         purview=tuple(m.nodes[i] for i in
                                       expected_partition[1].purview))
                ),
                unpartitioned_repertoire=expected['unpartitioned_repertoire'],
                partitioned_repertoire=expected_partitioned_repertoire,
                difference=expected['phi'])
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
        assert any(mip_eq(result, mip) for mip in expected)
    else:
        assert mip_eq(result, expected)


# Test input validation {{{
def test_find_mip_bad_direction(m):
    mechanism = [m.nodes[0]]
    purview = [m.nodes[0]]
    with pytest.raises(ValueError):
        m.subsys_all.find_mip('doge', mechanism, purview)
# }}}

# }}}


# Wrapper method tests {{{
# ========================


def test_mip_past(m):
    s = m.subsys_all
    mechanism = m.nodes
    purview = m.nodes
    mip_past = s.find_mip('past', mechanism, purview)
    assert tuple_eq(mip_past, s.mip_past(mechanism, purview))


def test_mip_future(m):
    s = m.subsys_all
    mechanism = m.nodes
    purview = m.nodes
    mip_future = s.find_mip('future', mechanism, purview)
    assert tuple_eq(mip_future, s.mip_future(mechanism, purview))


def test_phi_mip_past(m):
    s = m.subsys_all
    mechanism = m.nodes
    purview = m.nodes
    assert (s.phi_mip_past(mechanism, purview) ==
            s.mip_past(mechanism, purview).difference)


def test_phi_mip_past_reducible(m):
    s = m.subsys_all
    mechanism = [m.nodes[1]]
    purview = [m.nodes[0]]
    assert (0 == s.phi_mip_past(mechanism, purview))


def test_phi_mip_future(m):
    s = m.subsys_all
    mechanism = m.nodes
    purview = m.nodes
    assert (s.phi_mip_future(mechanism, purview) ==
            s.mip_future(mechanism, purview).difference)

def test_phi_mip_future_reducible(m):
    s = m.subsys_all
    mechanism = m.nodes[0:2]
    purview = [m.nodes[1]]
    assert (0 == s.phi_mip_future(mechanism, purview))


def test_phi(m):
    s = m.subsys_all
    mechanism = m.nodes
    purview = m.nodes
    assert abs(0.5 - s.phi(mechanism, purview)) < constants.EPSILON

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
