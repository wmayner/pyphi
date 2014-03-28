#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


# TODO finish
# XXX
def test_find_mice_bad_direction(m):
    mechanism = [m.nodes[0]]
    with pytest.raises(ValueError):
        m.subsys_all.find_mice('doge', mechanism)


def test_find_mice(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    past_mice = s.find_mice('past', mechanism)
    future_mice = s.find_mice('future', mechanism)
    assert 1


def test_core_cause(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    assert s.core_cause(mechanism) == s.find_mice('past', mechanism)


def test_core_effect(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    assert s.core_effect(mechanism) == s.find_mice('future', mechanism)


def test_phi_max(m):
    s = m.subsys_all
    mechanism = [m.nodes[0], m.nodes[1], m.nodes[2]]
    # assert 0.5 == round(s.phi_max(mechanism), 4)
