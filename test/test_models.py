#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np

from cyphi import models
from cyphi import options


nt_attributes = ['this', 'that', 'phi', 'mechanism', 'purview']
nt = namedtuple('nt', nt_attributes)
a = nt(this=('consciousness', 'is phi'), that=np.arange(3), phi=0.5,
       mechanism=(0, 1, 2), purview=(2, 4))


# Test equality helpers {{{
# =========================

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
    b = similar_nt(a.this, a.that, a.phi, a.mechanism, a.purview,
                   supbro="nothin' much")
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_true():
    b = nt(a.this, a.that, (a.phi - options.EPSILON/2), a.mechanism,
           a.purview)
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_false():
    b = nt(a.this, a.that, (a.phi - options.EPSILON*2), a.mechanism,
           a.purview)
    assert not models._general_eq(a, b, nt_attributes)


def test_general_eq_different_mechanism_order():
    b = nt(a.this, a.that, a.phi, a.mechanism[::-1], a.purview)
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_different_purview_order():
    b = nt(a.this, a.that, a.phi, a.mechanism, a.purview[::-1])
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_different_mechanism_and_purview_order():
    b = nt(a.this, a.that, a.phi, a.mechanism[::-1], a.purview[::-1])
    assert models._general_eq(a, b, nt_attributes)


# }}}

# Test MIP {{{
# ============

def test_mip_ordering():
    phi1 = models.Mip(
        direction=None, mechanism=(), purview=(), partition=None,
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=1.0)
    different_phi1 = models.Mip(
        direction='different', mechanism=(), purview=(), partition=0,
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=1.0)
    phi2 = models.Mip(
        direction=0, mechanism=(), purview=(), partition='stilldifferent',
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=1.0 + options.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_mip_equality():
    phi = 1.0
    mip = models.Mip(
        direction=None, mechanism=(), purview=(), partition=None,
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=phi)
    close_enough = models.Mip(
        direction=None, mechanism=(), purview=(), partition=None,
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=(phi - options.EPSILON/2))
    not_quite = models.Mip(
        direction=None, mechanism=(), purview=(), partition=None,
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=(phi - options.EPSILON*2))
    assert mip == close_enough
    assert mip != not_quite


# }}}

# Test MICE {{{
# =============

def test_mice_ordering_by_phi():
    phi1 = models.Mice(
        direction=None, mechanism=(), purview=(), repertoire=None, mip=None,
        phi=1.0)
    different_phi1 = models.Mice(
        direction='different', mechanism=(), purview=(), repertoire=None,
        mip=None, phi=1.0)
    phi2 = models.Mice(
        direction=0, mechanism=(), purview=(), repertoire=None,
        mip=None, phi=1.0 + options.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_mice_odering_by_mechanism():
    small = models.Mice(
        direction=None, mechanism=(1, 2), purview=(), repertoire=None,
        mip=None, phi=1.0)
    big = models.Mice(
        direction=None, mechanism=(1, 2, 3), purview=(), repertoire=None,
        mip=None, phi=1.0)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_mice_equality():
    phi = 1.0
    mice = models.Mice(
        direction=None, mechanism=(), purview=(), repertoire=None, mip=None,
        phi=phi)
    close_enough = models.Mice(
        direction=None, mechanism=(), purview=(), repertoire=None, mip=None,
        phi=(phi - options.EPSILON/2))
    not_quite = models.Mice(
        direction=None, mechanism=(), purview=(), repertoire=None, mip=None,
        phi=(phi - options.EPSILON*2))
    assert mice == close_enough
    assert mice != not_quite


# }}}

# Test Concept {{{
# ================

def test_concept_ordering():
    phi1 = models.Concept(
        mechanism=(0, 1), location=None, cause=None, effect=None,
        phi=1.0)
    different_phi1 = models.Concept(
        mechanism=(), location=1, cause=None, effect=None,
        phi=1.0)
    phi2 = models.Concept(
        mechanism=0, location='stilldifferent', cause=None, effect=None,
        phi=1.0 + options.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_concept_odering_by_mechanism():
    small = models.Concept(
        mechanism=(0, 1), location=None, cause=None, effect=None,
        phi=1.0)
    big = models.Concept(
        mechanism=(0, 1, 3), location=None, cause=None, effect=None,
        phi=1.0)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_concept_equality():
    phi = 1.0
    concept = models.Concept(
        mechanism=(), location=None, cause=None, effect=None,
        phi=phi)
    close_enough = models.Concept(
        mechanism=(), location=None, cause=None, effect=None,
        phi=(phi - options.EPSILON/2))
    not_quite = models.Concept(
        mechanism=(), location=None, cause=None, effect=None,
        phi=(phi - options.EPSILON*2))
    assert concept == close_enough
    assert concept != not_quite


# }}}

# Test BigMip {{{
# ===============

def test_bigmip_ordering():
    phi1 = models.BigMip(
        cut=None, unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=(),
        phi=1.0)
    different_phi1 = models.BigMip(
        cut='different', unpartitioned_constellation=0,
        partitioned_constellation=None, subsystem=(),
        phi=1.0)
    phi2 = models.BigMip(
        cut=0, unpartitioned_constellation='stilldifferent',
        partitioned_constellation=None, subsystem=(),
        phi=1.0 + options.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_bigmip_odering_by_mechanism():
    small = models.BigMip(
        cut=None, unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=1,
        phi=1.0)
    big = models.BigMip(
        cut=None, unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=2,
        phi=1.0)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_bigmip_equality(s):
    phi = 1.0
    bigmip = models.BigMip(
        cut=None, unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=s,
        phi=phi)
    close_enough = models.BigMip(
        cut=None, unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=s,
        phi=(phi - options.EPSILON/2))
    not_quite = models.BigMip(
        cut=None, unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=s,
        phi=(phi - options.EPSILON*2))
    assert bigmip == close_enough
    assert bigmip != not_quite


# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
