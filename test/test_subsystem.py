#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cyphi.subsystem import Subsystem


def test_empty_init(standard):
    # Empty mechanism
    s = Subsystem([], standard.current_state, standard.past_state, standard)
    assert s.nodes == ()


def test_eq(subsys_n0n2, subsys_n1n2):
    assert subsys_n0n2 == subsys_n0n2
    assert subsys_n0n2 != subsys_n1n2


def test_cmp(subsys_n0n2, subsys_n1n2, s):
    assert s > subsys_n0n2
    assert s > subsys_n1n2
    assert subsys_n0n2 >= subsys_n1n2
    assert s >= subsys_n0n2
    assert subsys_n0n2 < s
    assert subsys_n1n2 < s
    assert subsys_n0n2 <= s
    assert subsys_n0n2 <= subsys_n1n2


def test_hash(s):
    print(hash(s))
