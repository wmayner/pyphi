#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from itertools import chain

from cyphi.models import Mice
from cyphi.utils import phi_eq
from .example_networks import m


# Expected results {{{
# ====================

m = m()
subsystem = m.subsys_all
directions = ['past', 'future']


def indices2nodes(indices):
    return tuple(m.nodes[index] for index in indices)


expected_purview_indices = {
    'past': {
        (1,): (2,),
        (2,): (0, 1),
        (0, 1): (1, 2),
        (0, 1, 2): (0, 1, 2)},
    'future': {
        (1,): (0,),
        (2,): (1,),
        (0, 1): (2,),
        (0, 1, 2): (0, 1, 2)}
}
expected_purviews = {
    direction: {
        indices2nodes(mechanism): indices2nodes(purview) for mechanism, purview
        in expected_purview_indices[direction].items()
    } for direction in directions
}
expected_mips = {
    direction: {
        mechanism: subsystem.find_mip(direction, mechanism, purview) for
        mechanism, purview in expected_purviews[direction].items()
    } for direction in directions
}
expected_mice = {
    direction: [
        Mice(direction=direction,
             mechanism=mechanism,
             purview=expected_purviews[direction][mechanism],
             repertoire=mip.unpartitioned_repertoire,
             mip=mip,
             phi=mip.phi)
        for mechanism, mip in expected_mips[direction].items()
    ] for direction in directions
}

# }}}
# `_find_mice` tests {{{
# =====================

mice_scenarios = [
    [(direction, mice) for mice in expected_mice[direction]]
    for direction in directions
]
mice_scenarios = list(chain(*mice_scenarios))
mice_parameter_string = "direction,expected"


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_find_mice(m, direction, expected):
    assert (subsystem._find_mice(direction, expected.mechanism,
                                 subsystem.null_cut)
            == expected)


def test_find_mice_empty(m):
    s = m.subsys_all
    expected = [
        Mice(direction=direction,
             mechanism=(),
             purview=s.nodes,
             repertoire=None,
             mip=s._null_mip(direction, (), s.nodes),
             phi=0)
        for direction in directions]
    assert all(s._find_mice(mice.direction, mice.mechanism, s.null_cut) == mice
               for mice in expected)


# Test input validation
def test_find_mice_validation_bad_direction(m):
    mechanism = (m.nodes[0])
    s = m.subsys_all
    with pytest.raises(ValueError):
        s._find_mice('doge', mechanism, s.null_cut)


def test_find_mice_validation_nonnode(m):
    s = m.subsys_all
    with pytest.raises(ValueError):
        s._find_mice('past', [0, 1], s.null_cut)


def test_find_mice_validation_noniterable(m):
    s = m.subsys_all
    with pytest.raises(ValueError):
        s._find_mice('past', 0, s.null_cut)

# }}}
# `phi_max` tests {{{
# ===================

@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_core_cause_or_effect(m, direction, expected):
    if direction == 'past':
        core_ce = subsystem.core_cause
    elif direction == 'future':
        core_ce = subsystem.core_effect
    else:
        raise ValueError("Direction must be 'past' or 'future'")
    assert core_ce(expected.mechanism) == expected


phi_max_scenarios = [
    (past.mechanism, min(past.phi, future.phi))
    for past, future in zip(expected_mice['past'], expected_mice['future'])
]


@pytest.mark.parametrize('mechanism, expected_phi_max', phi_max_scenarios)
def test_phi_max(m, expected_phi_max, mechanism):
    assert phi_eq(m.subsys_all.phi_max(mechanism), expected_phi_max)

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
