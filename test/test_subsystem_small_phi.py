#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from cyphi.utils import tuple_eq
from cyphi.subsystem import Mip, Part


# Helper for checking MIP equality {{{
# ====================================
def mip_eq(a, b):
    """Return whether two MIPs are equal.

    Phi is compared up to 6 digits.
    """
    if not a or not b:
        return a == b
    return ((a.partition == b.partition or a.partition == (b.partition[1],
                                                           b.partition[0])) and
            (round(a.difference, 6) == round(b.difference, 6)) and
            (np.array_equal(a.repertoire, b.repertoire)))
# }}}


# `find_mip` tests {{{
# ====================

# Test scenario structure
# (
#     direction of MIP ('past' or 'future'),
#     subsystem, cut,
#     mechanism,
#     purview,
#     expected result,
# )
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
        {'part0': Part(mechanism=(), purview=(2,)),
         'part1': Part(mechanism=(1,), purview=()),
         'partitioned_repertoire': np.array([0.5, 0.5]).reshape(1, 1, 2),
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
        {'part0': Part(mechanism=(2,), purview=(0, 1)),
         'part1': Part(mechanism=(0, 1), purview=(2,)),
         'partitioned_repertoire': np.array([[[0.0, 0.5],
                                              [0.0, 0.0]],
                                             [[0.0, 0.5],
                                              [0.0, 0.0]]]),
         'phi': 0.5}
    ), (
        'future',
        'subsys_all', None,
        [0, 1, 2],
        [0, 1, 2],
        {'part0': Part(mechanism=(2,), purview=(0, 1)),
         'part1': Part(mechanism=(0, 1), purview=(2,)),
         'partitioned_repertoire': np.array([[[0.0, 0.5],
                                              [0.0, 0.0]],
                                             [[0.0, 0.5],
                                              [0.0, 0.0]]]),
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
        {'part0': Part(mechanism=(), purview=(1,)),
         'part1': Part(mechanism=(2,), purview=()),
         'partitioned_repertoire': np.array([0.5, 0.5]).reshape(1, 2, 1),
         'phi': 0.5}
    ), (
        'future',
        'subsys_all', ((0, 2), 1),
        [2],
        [0],
        {'part0': Part(mechanism=(), purview=(0,)),
         'part1': Part(mechanism=(2,), purview=()),
         'partitioned_repertoire': np.array([0.25, 0.75]).reshape(2, 1, 1),
         'phi': 0.25}
    ), (
        'future',
        'subsys_all', ((0, 2), 1),
        [0, 1, 2],
        [0, 2],
        {'part0': Part(mechanism=(0,), purview=(2,)),
         'part1': Part(mechanism=(1, 2), purview=(0,)),
         'partitioned_repertoire':
            np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2),
         'phi': 0.5}
    ), (
        'future',
        'subsys_all', ((0, 1), 2),
        [1],
        [0],
        {'part0': Part(mechanism=(), purview=(0,)),
         'part1': Part(mechanism=(1,), purview=()),
         'partitioned_repertoire': np.array([0.25, 0.75]).reshape(2, 1, 1),
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

    if expected:
        # Construct expected MIP
        part0 = Part(mechanism=tuple(m.nodes[i] for i in
                                     expected['part0'][0]),
                     purview=tuple(m.nodes[i] for i in expected['part0'][1]))
        part1 = Part(mechanism=tuple(m.nodes[i] for i in
                                     expected['part1'][0]),
                     purview=tuple(m.nodes[i] for i in expected['part1'][1]))
        expected = Mip(direction=direction,
                       partition=(part0, part1),
                       repertoire=expected['partitioned_repertoire'],
                       difference=expected['phi'])

    assert mip_eq(result, expected)


# Validation {{{
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
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    purview = [m.nodes[0], m.nodes[1], m.nodes[2]]
    mip_past = s.find_mip('past', mechanism, purview)
    assert tuple_eq(mip_past, s.mip_past(mechanism, purview))


def test_mip_future(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    purview = [m.nodes[0], m.nodes[1], m.nodes[2]]
    mip_future = s.find_mip('future', mechanism, purview)
    assert tuple_eq(mip_future, s.mip_future(mechanism, purview))


def test_phi_mip_past(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    purview = [m.nodes[0], m.nodes[1], m.nodes[2]]
    assert (s.phi_mip_past(mechanism, purview) ==
            s.mip_past(mechanism, purview).difference)


def test_phi_mip_future(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    purview = [m.nodes[0], m.nodes[2]]
    assert (s.phi_mip_future(mechanism, purview) ==
            s.mip_future(mechanism, purview).difference)


def test_phi(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    purview = [m.nodes[0], m.nodes[1], m.nodes[2]]
    assert 0.5 == round(s.phi(mechanism, purview), 6)

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
