#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from cyphi.utils import tuple_eq
from cyphi.subsystem import Mice, EPSILON


from pprint import pprint


@pytest.fixture()
def expected_mice(m):
    s = m.subsys_all
    # MICE for {n0, n1, n2}
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    expected_maximal_purview = (m.nodes[0], m.nodes[1], m.nodes[2])
    expected_past_mip = s.find_mip('past', mechanism, expected_maximal_purview)
    expected_future_mip = s.find_mip('future', mechanism,
                                     expected_maximal_purview)
    return {
        'past': Mice(direction='past',
                       purview=expected_maximal_purview,
                       mip=expected_past_mip,
                       phi=expected_past_mip.difference),
        'future': Mice(direction='future',
                         purview=expected_maximal_purview,
                         mip=expected_future_mip,
                         phi=expected_future_mip.difference)}


# TODO finish
# XXX
def test_find_mice_bad_direction(m):
    mechanism = (m.nodes[0])
    with pytest.raises(ValueError):
        m.subsys_all.find_mice('doge', mechanism)


def test_find_past_mice(m, expected_mice):
    s = m.subsys_all
    mechanism = (m.nodes[0], m.nodes[1], m.nodes[2])
    past_mice = s.find_mice('past', mechanism)
    assert tuple_eq(past_mice, expected_mice['past'])


def test_find_future_mice(m, expected_mice):
    s = m.subsys_all
    mechanism = (m.nodes[0], m.nodes[1], m.nodes[2])
    future_mice = s.find_mice('future', mechanism)
    assert tuple_eq(future_mice, expected_mice['future'])


def test_core_cause(m, expected_mice):
    s = m.subsys_all
    mechanism = (m.nodes[0], m.nodes[1], m.nodes[2])
    assert tuple_eq(s.core_cause(mechanism), expected_mice['past'])


def test_core_effect(m, expected_mice):
    s = m.subsys_all
    mechanism = (m.nodes[0], m.nodes[1], m.nodes[2])
    for item in ('result mice:', s.core_effect(mechanism), 'expected mice:', expected_mice['future']):
        pprint(item)
    assert tuple_eq(s.core_effect(mechanism), expected_mice['future'])


def test_phi_max(m):
    s = m.subsys_all
    mechanism = (m.nodes[0], m.nodes[1], m.nodes[2])
    assert abs(0.5 - s.phi_max(mechanism)) < EPSILON
