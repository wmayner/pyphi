#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
from cyphi.models import tuple_eq


nt = namedtuple('nt', ['this', 'that'])
a = nt(this=nt('consciousness', 'is phi'), that=np.arange(3))


def test_tuple_eq_noniterable():
    b = 1
    assert not tuple_eq(a, b)


def test_tuple_eq_nparray():
    b = np.arange(3)
    assert not tuple_eq(a, b)


def test_tuple_eq_tuple_nparrays():
    b = (np.arange(3), np.arange(3))
    assert not tuple_eq(a, b)


def test_tuple_eq_identical():
    b = a
    assert tuple_eq(a, b)
