#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from itertools import chain

from pyphi import Subsystem
from pyphi.models import Mice, Cut
from pyphi.utils import phi_eq

import example_networks


# Expected results {{{
# ====================

s = example_networks.s()
directions = ('past', 'future')
cuts = (None, Cut((1, 2), (0,)))
subsystem = {
    cut: Subsystem(s.node_indices, s.network, cut=cut)
    for cut in cuts
}

expected_purview_indices = {
    cuts[0]: {
        'past': {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (1, 2),
            (0, 1, 2): (0, 1, 2)
        },
        'future': {
            (1,): (0,),
            (2,): (1,),
            (0, 1): (2,),
            (0, 1, 2): (0, 1, 2)
        }
    },
    cuts[1]: {
        'past': {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (2,),
            (0, 1, 2): (0, 1, 2)
        },
        'future': {
            (1,): (0, 1, 2),
            (2,): (1,),
            (0, 1): (2,),
            (0, 1, 2): (0, 1, 2)
        }
    }
}
expected_purviews = {
    cut: {
        direction: {
            subsystem[cut].indices2nodes(mechanism):
                subsystem[cut].indices2nodes(purview)
            for mechanism, purview in
            expected_purview_indices[cut][direction].items()
        } for direction in directions
    } for cut in cuts
}
expected_mips = {
    cut: {
        direction: {
            mechanism: subsystem[cut].find_mip(direction, mechanism, purview)
            for mechanism, purview in
            expected_purviews[cut][direction].items()
        } for direction in directions
    } for cut in cuts
}
expected_mice = {
    cut: {
        direction: [
            Mice(mip) for mechanism, mip in
            expected_mips[cut][direction].items()
        ] for direction in directions
    } for cut in cuts
}

# }}}
# `find_mice` tests {{{
# =====================

mice_scenarios = [
    [[(cut, direction, mice) for mice in expected_mice[cut][direction]]
     for direction in directions] for cut in cuts
]
# Flatten doubly-nested list of scenarios.
mice_scenarios = list(chain(*list(chain(*mice_scenarios))))


mice_parameter_string = "cut,direction,expected"


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_find_mice(cut, direction, expected):
    result = subsystem[cut].find_mice(direction, expected.mechanism)
    print("Expected:\n", expected)
    print("Result:\n", result)
    assert result == expected


def test_find_mice_empty(s):
    expected = [Mice(s._null_mip(direction, (), s.nodes)) for direction in
                directions]
    assert all(s.find_mice(mice.direction, mice.mechanism) == mice
               for mice in expected)


# Test input validation
def test_find_mice_validation_bad_direction(s):
    mechanism = (s.nodes[0])
    with pytest.raises(ValueError):
        s.find_mice('doge', mechanism)

# }}}
# `phi_max` tests {{{
# ===================


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_core_cause_or_effect(cut, direction, expected):
    if direction == 'past':
        core_ce = subsystem[cut].core_cause
    elif direction == 'future':
        core_ce = subsystem[cut].core_effect
    else:
        raise ValueError("Direction must be 'past' or 'future'")
    assert core_ce(expected.mechanism) == expected


phi_max_scenarios = [
    [
        (cut, past.mechanism, min(past.phi, future.phi))
        for past, future in zip(expected_mice[cut]['past'],
                                expected_mice[cut]['future'])]
    for cut in cuts
]
# Flatten singly-nested list of scenarios.
phi_max_scenarios = list(chain(*phi_max_scenarios))


@pytest.mark.parametrize('cut,mechanism, expected_phi_max', phi_max_scenarios)
def test_phi_max(cut, expected_phi_max, mechanism):
    assert phi_eq(subsystem[cut].phi_max(mechanism), expected_phi_max)

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
