#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_models.py

from unittest import mock
from collections import namedtuple
import numpy as np

from pyphi import models, constants, config, Subsystem


# TODO: better way to build test objects than Mock?


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

# Test Cut {{{
# ============

def test_cut_splits_mechanism():
    cut = models.Cut((0,), (1, 2))
    assert cut.splits_mechanism((0, 1))
    assert not cut.splits_mechanism((0,))
    assert not cut.splits_mechanism((1, 2))


def test_cut_all_cut_mechanisms():
    cut = models.Cut((0,), (1, 2))
    assert cut.all_cut_mechanisms((0, 1, 2)) == ((0, 1), (0, 2), (0, 1, 2))
    assert cut.all_cut_mechanisms((0, 1)) == ((0, 1),)


def test_cut_matrix():

    cut = models.Cut((), (0,))
    matrix = np.array([[0]])
    assert np.array_equal(cut.cut_matrix(), matrix)

    cut = models.Cut((0,), (1,))
    matrix = np.array([
        [0, 1],
        [0, 0],
    ])
    assert np.array_equal(cut.cut_matrix(), matrix)

    cut = models.Cut((0, 2), (1, 2))
    matrix = np.array([
        [0, 1, 1],
        [0, 0, 0],
        [0, 1, 1],
    ])
    assert np.array_equal(cut.cut_matrix(), matrix)

    cut = models.Cut((), ())
    assert np.array_equal(cut.cut_matrix(), np.array([]))


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


def test_null_mip():
    direction = 'past'
    mechanism = (0,)
    purview = (1,)
    null_mip = models._null_mip(direction, mechanism, purview)
    assert null_mip.direction == direction
    assert null_mip.mechanism == mechanism
    assert null_mip.purview == purview
    assert null_mip.partition is None
    assert null_mip.unpartitioned_repertoire is None
    assert null_mip.partitioned_repertoire is None
    assert null_mip.phi == 0


def test_mip_repr_str():
    mip = models.Mip(direction=None, mechanism=(), purview=(),
                     unpartitioned_repertoire=None,
                     partitioned_repertoire=None, phi=0.0,
                     partition=(models.Part((), ()), models.Part((), ())))
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
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    print(repr(mice))
    print(str(mice))


def test_relevant_connections(s, subsys_n1n2):
    mip = mock.Mock(mechanism=(0,), purview=(1,), direction='past')
    mice = models.Mice(mip)
    answer = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    assert np.array_equal(mice._relevant_connections(s), answer)

    mip = mock.Mock(mechanism=(1,), purview=(1, 2), direction='future')
    mice = models.Mice(mip)
    answer = np.array([
        [1, 1],
        [0, 0],
    ])
    assert np.array_equal(mice._relevant_connections(subsys_n1n2), answer)


def test_damaged(s):
    # Build cut subsystem from s
    cut = models.Cut((0,), (1, 2))
    subsys = Subsystem(s.network, s.state, s.node_indices, cut=cut)

    # Cut splits mechanism:
    mip = mock.MagicMock(mechanism=(0, 1), purview=(1, 2), direction='future')
    mice = models.Mice(mip)
    assert mice.damaged_by_cut(subsys)
    assert not mice.damaged_by_cut(s)

    # Cut splits mechanism & purview (but not *only* mechanism)
    mip = mock.MagicMock(mechanism=(0,), purview=(1, 2), direction='future')
    mice = models.Mice(mip)
    assert mice.damaged_by_cut(subsys)
    assert not mice.damaged_by_cut(s)


# }}}

# Test Concept {{{
# ================

def test_concept_ordering(s):
    phi1 = models.Concept(
        mechanism=(0, 1), cause=1, effect=None, subsystem=s,
        phi=1.0)
    different_phi1 = models.Concept(
        mechanism=(), cause='different', effect=None, subsystem=s,
        phi=1.0)
    phi2 = models.Concept(
        mechanism=0, cause='stilldifferent', effect=None, subsystem=s,
        phi=1.0 + constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    assert phi1 <= different_phi1
    assert phi1 >= different_phi1


def test_concept_odering_by_mechanism(s):
    small = models.Concept(
        mechanism=(0, 1), cause=None, effect=None, subsystem=s,
        phi=1.0)
    big = models.Concept(
        mechanism=(0, 1, 3), cause=None, effect=None, subsystem=s,
        phi=1.0)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_concept_equality(s):
    phi = 1.0
    concept = models.Concept(mechanism=(), cause=None, effect=None,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(), cause=None, effect=None,
                             subsystem=s, phi=phi)
    assert concept == another


def test_concept_equality_phi(s):
    concept = models.Concept(mechanism=(), cause=None, effect=None,
                             subsystem=s, phi=1.0)
    another = models.Concept(mechanism=(), cause=None, effect=None,
                             subsystem=s, phi=0.0)
    assert concept != another


def test_concept_equality_mechanism(s):
    phi = 1.0
    concept = models.Concept(mechanism=(1,), cause=None, effect=None,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(), cause=None, effect=None,
                             subsystem=s, phi=phi)
    assert concept != another


def test_concept_equality_cause_purview_nodes(s):
    phi = 1.0
    mice1 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(1, 2),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    mice2 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(1,),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    concept = models.Concept(mechanism=(), cause=mice1, effect=None,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(), cause=mice2, effect=None,
                             subsystem=s, phi=phi)
    assert concept != another


def test_concept_equality_effect_purview_nodes(s):
    phi = 1.0
    mice1 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(1, 2),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    mice2 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(1,),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    concept = models.Concept(mechanism=(), cause=None, effect=mice1,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(), cause=None, effect=mice2,
                             subsystem=s, phi=phi)
    assert concept != another


def test_concept_equality_repertoires(s):
    phi = 1.0
    mice1 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=np.array([1, 2]), partitioned_repertoire=(),
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    mice2 = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=np.array([0, 0]), partitioned_repertoire=None,
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    concept = models.Concept(mechanism=(), cause=mice1, effect=mice2,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(), cause=mice2, effect=mice1,
                             subsystem=s, phi=phi)
    assert concept != another


def test_concept_equality_network(s, simple_subsys_all_off):
    phi = 1.0
    concept = models.Concept(mechanism=(), cause=None, effect=None,
                             subsystem=simple_subsys_all_off, phi=phi)
    another = models.Concept(mechanism=(), cause=None, effect=None,
                             subsystem=s, phi=phi)
    assert concept != another


def test_concept_equality_one_subsystem_is_subset_of_another(s, subsys_n1n2):
    phi = 1.0
    mice = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(1, 2),
        unpartitioned_repertoire=(), partitioned_repertoire=(),
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    concept = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=subsys_n1n2, phi=phi)
    assert concept == another


def test_concept_repr_str():
    mice = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    concept = models.Concept(
        mechanism=(), cause=mice, effect=mice,
        subsystem=None, phi=0.0)
    print(repr(concept))
    print(str(concept))


def test_concept_hashing(s):
    mice = models.Mice(models.Mip(
        direction=None, mechanism=(0, 1, 2), purview=(0, 1, 2),
        unpartitioned_repertoire=None, partitioned_repertoire=None,
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    concept = models.Concept(
        mechanism=(0, 1, 2), cause=mice, effect=mice, subsystem=s, phi=0.0)
    hash(concept)


def test_concept_hashing_one_subsystem_is_subset_of_another(s, subsys_n1n2):
    phi = 1.0
    mice = models.Mice(models.Mip(
        direction=None, mechanism=(), purview=(1, 2),
        unpartitioned_repertoire=(), partitioned_repertoire=(),
        phi=0.0, partition=(models.Part((), ()), models.Part((), ()))))
    concept = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=subsys_n1n2, phi=phi)
    assert hash(concept) == hash(another)
    assert(len(set([concept, another])) == 1)


# }}}

# Test Constellation {{{
# ======================

def test_constellation_is_still_a_tuple():
    c = models.Constellation([models.Concept()])
    assert len(c) == 1


@config.override(READABLE_REPRS=False)
def test_constellation_repr():
    c = models.Constellation()
    assert repr(c) == "Constellation(())"


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


def test_bigmip_repr_str(s):
    bigmip = models.BigMip(
        unpartitioned_constellation=None, partitioned_constellation=None,
        subsystem=s, cut_subsystem=s, phi=1.0)
    print(repr(bigmip))
    print(str(bigmip))


# }}}

# Test model __str__ and __reprs__ {{{
# ====================================

def test_indent():
    s = ("line1\n"
         "line2")
    answer = ("  line1\n"
              "  line2")
    assert models.indent(s) == answer


class ReadableReprClass:
    """Dummy class for make_repr tests"""
    some_attr = 3.14

    def __repr__(self):
        return models.make_repr(self, ['some_attr'])

    def __str__(self):
        return "A nice fat explicit string"


@config.override(READABLE_REPRS=False)
def test_make_reprs_uses___repr__():
    assert repr(ReadableReprClass()) == "ReadableReprClass(some_attr=3.14)"


@config.override(READABLE_REPRS=True)
def test_make_reprs_calls_out_to_string():
    assert repr(ReadableReprClass()) == "A nice fat explicit string"

# }}}

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
