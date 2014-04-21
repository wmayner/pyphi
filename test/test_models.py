#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np

from cyphi import models, constants

nt_attributes = ['this', 'that', 'phi']
nt = namedtuple('nt', nt_attributes)
a = nt(('consciousness', 'is phi'), np.arange(3), 0.5)


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
    b = similar_nt(this=('consciousness', 'is phi'),
                   that=np.arange(3),
                   phi=0.5,
                   supbro="nothin' much")
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_true():
    b = nt(('consciousness', 'is phi'), np.arange(3), (a.phi -
                                                       constants.EPSILON/2))
    assert models._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_false():
    b = nt(('consciousness', 'is phi'), np.arange(3), (a.phi -
                                                       constants.EPSILON*2))
    assert not models._general_eq(a, b, nt_attributes)

# }}}
# Test MIP {{{
# ============

def test_mip_ordering():
    phi1 = models.Mip(direction=None, mechanism=None, purview=None,
                      partition=None, unpartitioned_repertoire=None,
                      partitioned_repertoire=None,
                      phi=1.0)
    different_phi1 = models.Mip(direction='different', mechanism=None,
                                purview=None, partition=0,
                                unpartitioned_repertoire=None,
                                partitioned_repertoire=None,
                                phi=1.0)
    phi2 = models.Mip(direction=0, mechanism=None, purview=None,
                      partition='stilldifferent',
                      unpartitioned_repertoire=None,
                      partitioned_repertoire=None,
                      phi=1.0 + constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_mip_equality():
    phi = 1.0
    mip = models.Mip(direction=None, mechanism=(), purview=(), partition=None,
                     unpartitioned_repertoire=None,
                     partitioned_repertoire=None, phi=phi)
    close_enough = models.Mip(direction=None, mechanism=(), purview=(),
                              partition=None, unpartitioned_repertoire=None,
                              partitioned_repertoire=None,
                              phi=(phi - constants.EPSILON/2))
    not_quite = models.Mip(direction=None, mechanism=(), purview=(),
                           partition=None, unpartitioned_repertoire=None,
                           partitioned_repertoire=None,
                           phi=(phi - constants.EPSILON*2))
    assert mip == close_enough
    assert mip != not_quite

# }}}
# Test MICE {{{
# =============

def test_mice_ordering():
    phi1 = models.Mice(direction=None, mechanism=None, purview=None,
                       repertoire=None, mip=None, phi=1.0)
    different_phi1 = models.Mice(direction='different', mechanism=0,
                                 purview=None, repertoire=None, mip=None,
                                 phi=1.0)
    phi2 = models.Mice(direction=0, mechanism='stilldifferent', purview=None,
                       repertoire=None, mip=None, phi=1.0 +
                       constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_mice_equality():
    phi = 1.0
    mice = models.Mice(direction=None, mechanism=None, purview=None,
                       repertoire=None, mip=None, phi=phi)
    close_enough = models.Mice(direction=None, mechanism=None, purview=None,
                       repertoire=None, mip=None, phi=(phi - constants.EPSILON/2))
    not_quite = models.Mice(direction=None, mechanism=None, purview=None,
                            repertoire=None, mip=None,
                            phi=(phi - constants.EPSILON*2))
    assert mice == close_enough
    assert mice != not_quite

# }}}
# Test Concept {{{
# ================

def test_concept_ordering():
    phi1 = models.Concept(mechanism=None, location=None, cause=None,
                          effect=None, phi=1.0)
    different_phi1 = models.Concept(mechanism='different', location=0,
                                    cause=None, effect=None, phi=1.0)
    phi2 = models.Concept(mechanism=0, location='stilldifferent', cause=None,
                          effect=None, phi=1.0 + constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_concept_equality():
    phi = 1.0
    concept = models.Concept(mechanism=None, location=None, cause=None,
                             effect=None, phi=phi)
    close_enough = models.Concept(mechanism=None, location=None, cause=None,
                                  effect=None, phi=(phi - constants.EPSILON/2))
    not_quite = models.Concept(mechanism=None, location=None, cause=None,
                               effect=None, phi=(phi - constants.EPSILON*2))
    assert concept == close_enough
    assert concept != not_quite

# }}}
# Test BigMip {{{
# ===============

def test_bigmip_ordering():
    phi1 = models.BigMip(partition=None, unpartitioned_constellation=None,
                         partitioned_constellation=None, phi=1.0,
                         subsystem=())
    different_phi1 = models.BigMip(partition='different',
                                   unpartitioned_constellation=0,
                                   partitioned_constellation=None, phi=1.0,
                                   subsystem=())
    phi2 = models.BigMip(partition=0,
                         unpartitioned_constellation='stilldifferent',
                         partitioned_constellation=None,
                         phi=1.0 + constants.EPSILON*2,
                         subsystem=())
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_bigmip_equality(s):
    phi = 1.0
    bigmip = models.BigMip(partition=None, unpartitioned_constellation=None,
                           partitioned_constellation=None, phi=phi,
                           subsystem=s)
    close_enough = models.BigMip(partition=None,
                                 unpartitioned_constellation=None,
                                 partitioned_constellation=None,
                                 phi=(phi - constants.EPSILON/2),
                                 subsystem=s)
    not_quite = models.BigMip(partition=None, unpartitioned_constellation=None,
                              partitioned_constellation=None,
                              phi=(phi - constants.EPSILON*2),
                              subsystem=s)
    assert bigmip == close_enough
    assert bigmip != not_quite

# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
