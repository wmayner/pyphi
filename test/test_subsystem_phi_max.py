#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from itertools import chain
from cyphi.utils import tuple_eq
from cyphi.subsystem import Mice, EPSILON
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
             repertoire=expected_mips[direction][mechanism].unpartitioned_repertoire,
             mip=expected_mips[direction][mechanism],
             phi=expected_mips[direction][mechanism].difference)
        for mechanism in expected_mips[direction].keys()
    ] for direction in directions
}

# }}}


# `find_mice` tests {{{
# =====================

mice_scenarios = [
    [(direction, mice) for mice in expected_mice[direction]]
    for direction in directions
]
mice_scenarios = chain(*mice_scenarios)
mice_parameter_string = "direction,expected"

@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_find_mice(m, direction, expected):
    assert tuple_eq(subsystem.find_mice(direction, expected.mechanism),
                    expected)


# Test input validation
def test_find_mice_validation_bad_direction(m):
    mechanism = (m.nodes[0])
    with pytest.raises(ValueError):
        m.subsys_all.find_mice('doge', mechanism)

def test_find_mice_validation_nonnode(m):
    with pytest.raises(ValueError):
        m.subsys_all.find_mice('past', [0,1])

def test_find_mice_validation_noniterable(m):
    with pytest.raises(ValueError):
        m.subsys_all.find_mice('past', 0)
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
    assert tuple_eq(core_ce(expected.mechanism), expected)


phi_max_scenarios = [
    (past.mechanism, min(past.phi, future.phi))
    for past, future in zip(expected_mice['past'], expected_mice['future'])
]


@pytest.mark.parametrize('mechanism, expected_phi_max', phi_max_scenarios)
def test_phi_max(m, expected_phi_max, mechanism):
    assert abs(m.subsys_all.phi_max(mechanism) - expected_phi_max) < EPSILON

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
