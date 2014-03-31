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
    direction: {
        mechanism: Mice(
            direction=direction,
            purview=expected_purviews[direction][mechanism],
            mip=expected_mips[direction][mechanism],
            phi=expected_mips[direction][mechanism].difference)
        for mechanism in expected_mips[direction].keys()
    } for direction in directions
}

# }}}


# `find_mice` tests {{{
# =====================

mice_scenarios = [
    [(direction, mechanism) for mechanism in expected_mice[direction].keys()]
    for direction in directions
]
mice_scenarios = chain(*mice_scenarios)
mice_parameter_string = "direction,mechanism"

@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_find_mice(m, direction, mechanism):
    assert tuple_eq(subsystem.find_mice(direction, mechanism),
                    expected_mice[direction][mechanism])


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
def test_core_cause_or_effect(m, direction, mechanism):
    if direction == 'past':
        core_ce = subsystem.core_cause
    elif direction == 'future':
        core_ce = subsystem.core_effect
    else:
        raise ValueError("Direction must be 'past' or 'future'")
    assert tuple_eq(core_ce(mechanism), expected_mice[direction][mechanism])


phi_max_scenarios = [(mechanism, min(expected_mice['past'][mechanism].phi,
                                     expected_mice['future'][mechanism].phi))
                     for mechanism in expected_mice['past'].keys()]


@pytest.mark.parametrize('mechanism,expected', phi_max_scenarios)
def test_phi_max(m, expected, mechanism):
    assert abs(m.subsys_all.phi_max(mechanism) - expected) < EPSILON

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
