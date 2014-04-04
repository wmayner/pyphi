#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from copy import deepcopy
import numpy as np

import cyphi.models as models

nt_attributes = ['this', 'that', 'phi']
nt = namedtuple('nt', nt_attributes)
a = nt(this=nt('consciousness', 'is phi'), that=np.arange(3), phi=0.5)


def test_numpy_aware_eq_noniterable():
    b = 1
    assert not models._numpy_aware_eq(a, b)


def test_numpy_aware_eq_nparray():
    b = np.arange(3)
    assert not models._numpy_aware_eq(a, b)


def test_numpy_aware_eq_tuple_nparrays():
    b = (np.arange(3), np.arange(3))
    assert not models._numpy_aware_eq(a, b)


def test_numpy_aware_eq_identical():
    b = a
    assert models._numpy_aware_eq(a, b)


def test_general_eq_different_attributes():
    similar_nt = namedtuple('nt', nt_attributes + ['supbro'])
    b = similar_nt(this=nt('consciousness', 'is phi'),
                   that=np.arange(3),
                   supbro="nothin' much")
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_true():
    b = deepcopy(a)
    b.phi = 0.4999999999999
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_false():
    b = deepcopy(a)
    b.phi = 0.4999
    assert not models._general_eq(a, b, nt_attributes)


def test_phi_ordering():
    pass
