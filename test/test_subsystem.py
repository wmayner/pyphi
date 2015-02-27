#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyphi.subsystem import Subsystem
from pyphi.models import Cut
import numpy as np

def test_empty_init(standard):
    # Empty mechanism
    s = Subsystem((), standard)
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


def test_len(s, big_subsys_0_thru_3, big_subsys_all):
    assert len(s) == 3
    assert len(big_subsys_0_thru_3) == 4
    assert len(big_subsys_all) == 5


def test_hash(s):
    print(hash(s))

def test_find_cut_matrix(s, big_subsys_0_thru_3):
    cut = Cut((0,), (1,2))
    cut_s = Subsystem(s.node_indices,
                      s.network,
                      cut=cut,
                      mice_cache=s._mice_cache)
    answer_s = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    cut = Cut((0,1), (2,3))
    cut_big = Subsystem(big_subsys_0_thru_3.node_indices,
                        big_subsys_0_thru_3.network,
                        cut=cut,
                        mice_cache=big_subsys_0_thru_3._mice_cache)
    answer_big = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert np.array_equal(cut_s.cut_matrix, answer_s)
    assert np.array_equal(cut_big.cut_matrix, answer_big)

