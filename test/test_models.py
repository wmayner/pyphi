#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np

from pyphi import models
from pyphi import constants


nt_attributes = ['this', 'that', 'phi', 'mechanism', 'purview']
nt = namedtuple('nt', nt_attributes)
a = nt(this=('consciousness', 'is phi'), that=np.arange(3), phi=0.5,
       mechanism=(0, 1, 2), purview=(2, 4))


# Test equality helpers {{{
# =========================

def test_phi_comparisons():

    class PhiThing:
        def __init__(self, phi):
            self.phi = phi
    small = PhiThing(0.0)
    large = PhiThing(2.0)

    assert models._phi_lt(small, large)
    assert not models._phi_lt(large, small)
    assert not models._phi_lt(small, small)
    assert not models._phi_lt(small, 'attr_error')

    assert models._phi_le(small, large)
    assert not models._phi_le(large, small)
    assert models._phi_le(small, small)
    assert not models._phi_le(small, 'attr_error')

    assert models._phi_gt(large, small)
    assert not models._phi_gt(small, large)
    assert not models._phi_gt(large, large)
    assert not models._phi_gt(small, 'attr_error')

    assert models._phi_ge(large, small)
    assert not models._phi_ge(small, large)
    assert models._phi_ge(large, large)
    assert not models._phi_ge(small, 'attr_error')


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
    b = nt(a.this, a.that, (a.phi - constants.EPSILON/2), a.mechanism,
           a.purview)
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_false():
    b = nt(a.this, a.that, (a.phi - constants.EPSILON*2), a.mechanism,
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
        phi=1.0 + constants.EPSILON*2)
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
        phi=(phi - constants.EPSILON/2))
    not_quite = models.Mip(
        direction=None, mechanism=(), purview=(), partition=None,
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=(phi - constants.EPSILON*2))
    assert mip == close_enough
    assert mip != not_quite


def test_mip_repr_str():
    mip = models.Mip(direction=None, mechanism=(), purview=(),
                     unpartitioned_repertoire=None,
                     partitioned_repertoire=None, phi=0.0, partition=())
    print(repr(mip))
    print(str(mip))


# }}}

# Test MICE {{{
# =============

def test_mice_ordering_by_phi():
    phi1 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=1.0, partition=()))
    different_phi1 = models.Mice(models.Mip(
        direction='different', mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=1.0, partition=()))
    phi2 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=(1.0 + constants.EPSILON*2), partition=()))
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_mice_odering_by_mechanism():
    small = models.Mice(models.Mip(
        direction=None, mechanism=(1, 2), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=1.0, partition=()))
    big = models.Mice(models.Mip(
        direction=None, mechanism=(1, 2, 3), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=1.0, partition=()))
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_mice_equality():
    phi = 1.0
    mice = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=phi, partition=()))
    close_enough = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=(phi - constants.EPSILON/2), partition=()))
    not_quite = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=(phi - constants.EPSILON*2), partition=()))
    assert mice == close_enough
    assert mice != not_quite


def test_mice_repr_str():
    mice = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=0.0, partition=()))
    print(repr(mice))
    print(str(mice))


# }}}

# Test Concept {{{
# ================

def test_concept_ordering():
    phi1 = models.Concept(
        mechanism=(0, 1), cause=1, effect=None, subsystem=None,
        phi=1.0)
    different_phi1 = models.Concept(
        mechanism=(), cause='different', effect=None, subsystem=None,
        phi=1.0)
    phi2 = models.Concept(
        mechanism=0, cause='stilldifferent', effect=None, subsystem=None,
        phi=1.0 + constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_concept_odering_by_mechanism():
    small = models.Concept(
        mechanism=(0, 1), cause=None, effect=None, subsystem=None,
        phi=1.0)
    big = models.Concept(
        mechanism=(0, 1, 3), cause=None, effect=None, subsystem=None,
        phi=1.0)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_concept_equality():
    phi = 1.0
    concept = models.Concept(
        mechanism=(), cause=None, effect=None, subsystem=None,
        phi=phi)
    close_enough = models.Concept(
        mechanism=(), cause=None, effect=None, subsystem=None,
        phi=(phi - constants.EPSILON/2))
    not_quite = models.Concept(
        mechanism=(), cause=None, effect=None, subsystem=None,
        phi=(phi - constants.EPSILON*2))
    assert concept == close_enough
    assert concept != not_quite


def test_concept_repr_str():
    r = namedtuple('object_with_repertoire', ['repertoire'])
    concept = models.Concept(
        mechanism=(), cause=r('a_repertoire'), effect=r('a_repertoire'),
        subsystem=None, phi=0.0)
    print(repr(concept))
    print(str(concept))


# }}}

# Test BigMip {{{
# ===============

def test_bigmip_ordering():
    phi1 = models.BigMip(
        unpartitioned_constellation=None, partitioned_constellation=None,
        subsystem=(), cut_subsystem=(),
        phi=1.0)
    different_phi1 = models.BigMip(
        unpartitioned_constellation=0, partitioned_constellation=None,
        subsystem=(), cut_subsystem=(),
        phi=1.0)
    phi2 = models.BigMip(
        unpartitioned_constellation='stilldifferent',
        partitioned_constellation=None, subsystem=(), cut_subsystem=(),
        phi=1.0 + constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_bigmip_odering_by_mechanism():
    small = models.BigMip(
        unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=[1], cut_subsystem=(),
        phi=1.0)
    big = models.BigMip(
        unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=[1, 2], cut_subsystem=(),
        phi=1.0)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_bigmip_equality(s):
    phi = 1.0
    bigmip = models.BigMip(
        unpartitioned_constellation=None, partitioned_constellation=None,
        subsystem=s, cut_subsystem=s,
        phi=phi)
    close_enough = models.BigMip(
        unpartitioned_constellation=None, partitioned_constellation=None,
        subsystem=s, cut_subsystem=s,
        phi=(phi - constants.EPSILON/2))
    not_quite = models.BigMip(
        unpartitioned_constellation=None, partitioned_constellation=None,
        subsystem=s, cut_subsystem=s,
        phi=(phi - constants.EPSILON*2))
    assert bigmip == close_enough
    assert bigmip != not_quite


def test_bigmip_repr_str():
    bigmip = models.BigMip(
        unpartitioned_constellation=None, partitioned_constellation=None,
        subsystem=(), cut_subsystem=(), phi=1.0)
    print(repr(bigmip))
    print(str(bigmip))


# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
