#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from cyphi.models import Cut
from cyphi.subsystem import Subsystem


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
