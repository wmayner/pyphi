#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_partition.py

from pyphi.partition import directed_bipartition, directed_tripartition_indices


def test_directed_bipartition():
    answer = [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,)),
              ((3,), (1, 2)), ((1, 3), (2,)), ((2, 3), (1,)), ((1, 2, 3), ())]
    assert answer == directed_bipartition((1, 2, 3))
    # Test with empty input
    assert [] == directed_bipartition(())


def test_directed_tripartition_indices():
    assert directed_tripartition_indices(0) == []
    assert directed_tripartition_indices(2) == [
        ((0, 1), (), ()),
        ((0,), (1,), ()),
        ((0,), (), (1,)),
        ((1,), (0,), ()),
        ((), (0, 1), ()),
        ((), (0,), (1,)),
        ((1,), (), (0,)),
        ((), (1,), (0,)),
        ((), (), (0, 1))]
