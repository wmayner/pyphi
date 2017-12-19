#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import chain

import pytest

import example_networks
from pyphi import Direction, Subsystem
from pyphi.models import Cut, MaximallyIrreducibleCauseOrEffect, _null_ria
from pyphi.utils import eq

# Expected results {{{
# ====================

s = example_networks.s()
directions = (Direction.CAUSE, Direction.EFFECT)
cuts = (None, Cut((1, 2), (0,)))
subsystem = {
    cut: Subsystem(s.network, s.state, s.node_indices, cut=cut)
    for cut in cuts
}

expected_purview_indices = {
    cuts[0]: {
        Direction.CAUSE: {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (1, 2),
            (0, 1, 2): (0, 1, 2)
        },
        Direction.EFFECT: {
            (1,): (0,),
            (2,): (1,),
            (0, 1): (2,),
            (0, 1, 2): (0, 1, 2)
        }
    },
    cuts[1]: {
        Direction.CAUSE: {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (),
            (0, 1, 2): (),
        },
        Direction.EFFECT: {
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
            MaximallyIrreducibleCauseOrEffect(mip) for mechanism, mip in
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
    expected = [
        MaximallyIrreducibleCauseOrEffect(_null_ria(direction, (), ()))
        for direction in directions
    ]
    assert all(s.find_mice(mice.direction, mice.mechanism) == mice
               for mice in expected)


# }}}
# `phi_max` tests {{{
# ===================


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_mic_or_mie(cut, direction, expected):
    if direction == Direction.CAUSE:
        mice = subsystem[cut].mic
    elif direction == Direction.EFFECT:
        mice = subsystem[cut].mie
    assert mice(expected.mechanism) == expected


phi_max_scenarios = [
    [
        (cut, cause.mechanism, min(cause.phi, effect.phi))
        for cause, effect in zip(expected_mice[cut][Direction.CAUSE],
                                 expected_mice[cut][Direction.EFFECT])
    ] for cut in cuts
]
# Flatten singly-nested list of scenarios.
phi_max_scenarios = list(chain(*phi_max_scenarios))


@pytest.mark.parametrize('cut,mechanism,expected_phi_max', phi_max_scenarios)
def test_phi_max(cut, expected_phi_max, mechanism):
    assert eq(subsystem[cut].phi_max(mechanism), expected_phi_max)

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
