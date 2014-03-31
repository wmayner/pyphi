#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from cyphi.subsystem import Subsystem, Cut


# TODO test against other matlab examples
def test_empty_init(m):
    # Empty mechanism
    s = Subsystem([], m.current_state, m.past_state, m)
    assert s.nodes == ()


def test_eq(m):
    assert m.subsys_n0n2 == m.subsys_n0n2
    assert m.subsys_n0n2 != m.subsys_n1n2


def test_hash(m):
    print(hash(m.subsys_all))


def test_cut_bad_input(m):
    s = m.subsys_all
    with pytest.raises(ValueError):
        s.cut((), ())
    with pytest.raises(ValueError):
        s.cut(m.nodes[0], m.nodes[1])
    with pytest.raises(ValueError):
        s.cut(m.nodes[0], (m.nodes[1], m.nodes[1]))


def test_cut_single_node(m):
    s = m.subsys_all
    s.cut(m.nodes[0], (m.nodes[1], m.nodes[2]))
    assert s._cut == Cut((m.nodes[0],), (m.nodes[1], m.nodes[2]))


def test_cut_list_input(m):
    s = m.subsys_all
    s.cut([m.nodes[0]], [m.nodes[1], m.nodes[2]])
    assert s._cut == Cut((m.nodes[0],), (m.nodes[1], m.nodes[2]))


def test_cut(m):
    s = m.subsys_all
    s.cut((m.nodes[0],), (m.nodes[1], m.nodes[2]))
    assert s._cut == Cut((m.nodes[0],), (m.nodes[1], m.nodes[2]))
