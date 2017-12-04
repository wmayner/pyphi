#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import chain

import pytest

import example_networks
from pyphi import Direction, Subsystem
from pyphi.models import Cut, Mice, _null_mip
from pyphi.utils import eq

# Expected results {{{
# ====================

s = example_networks.s()
directions = (Direction.PAST, Direction.FUTURE)
cuts = (None, Cut((1, 2), (0,)))
subsystem = {
    cut: Subsystem(s.network, s.state, s.node_indices, cut=cut)
    for cut in cuts
}

expected_purview_indices = {
    cuts[0]: {
        Direction.PAST: {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (1, 2),
            (0, 1, 2): (0, 1, 2)
        },
        Direction.FUTURE: {
            (1,): (0,),
            (2,): (1,),
            (0, 1): (2,),
            (0, 1, 2): (0, 1, 2)
        }
    },
    cuts[1]: {
        Direction.PAST: {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (),
            (0, 1, 2): (),
        },
        Direction.FUTURE: {
            (1,): (2,),
            (2,): (1,),
            (0, 1): (2,),
            (0, 1, 2): (),
        }
    }
}
expected_purviews = {
    cut: {
        direction: {
            mechanism: purview
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
    expected = [Mice(_null_mip(direction, (), ())) for direction in
                directions]
    assert all(s.find_mice(mice.direction, mice.mechanism) == mice
               for mice in expected)


# }}}
# `phi_max` tests {{{
# ===================


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_core_cause_or_effect(cut, direction, expected):
    if direction == Direction.PAST:
        core_ce = subsystem[cut].core_cause
    elif direction == Direction.FUTURE:
        core_ce = subsystem[cut].core_effect
    assert core_ce(expected.mechanism) == expected


phi_max_scenarios = [
    [
        (cut, past.mechanism, min(past.phi, future.phi))
        for past, future in zip(expected_mice[cut][Direction.PAST],
                                expected_mice[cut][Direction.FUTURE])]
    for cut in cuts
]
# Flatten singly-nested list of scenarios.
phi_max_scenarios = list(chain(*phi_max_scenarios))


@pytest.mark.parametrize('cut,mechanism,expected_phi_max', phi_max_scenarios)
def test_phi_max(cut, expected_phi_max, mechanism):
    assert eq(subsystem[cut].phi_max(mechanism), expected_phi_max)

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
