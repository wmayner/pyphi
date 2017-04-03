#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_models.py

from unittest import mock
from collections import namedtuple
import numpy as np
import pytest

from pyphi import models, constants, config, Subsystem
from pyphi.constants import Direction


# TODO: better way to build test objects than Mock?


nt_attributes = ['this', 'that', 'phi', 'mechanism', 'purview']
nt = namedtuple('nt', nt_attributes)
a = nt(this=('consciousness', 'is phi'), that=np.arange(3), phi=0.5,
       mechanism=(0, 1, 2), purview=(2, 4))


# Test equality helpers {{{
# =========================

def test_phi_mechanism_ordering():

    class PhiThing(models.cmp._Orderable):
        def __init__(self, phi, mechanism):
            self.phi = phi
            self.mechanism = mechanism

        def _order_by(self):
            return [self.phi, self.mechanism]

        def __eq__(self, other):
            return self.phi == other.phi and self.mechanism == other.mechanism

    # assert PhiThing(1.0, (1,)) == PhiThing(1.0, (1,))
    # assert PhiThing(1.0, (1,)) == PhiThing(1.0, (1, 2))
    # assert PhiThing(1.0, (1,)) != PhiThing(2.0, (1, 2))
    assert PhiThing(1.0, (1,)) < PhiThing(2.0, (1,))
    assert PhiThing(1.0, (1,)) <= PhiThing(1.0, (1, 2))  # Smaller mechanism
    assert PhiThing(1.0, (1,)) <= PhiThing(2.0, (1,))
    assert PhiThing(2.0, (1,)) > PhiThing(1.0, (1,))
    assert PhiThing(2.0, (1,)) > PhiThing(1.0, (1, 2))  # Larger phi
    assert PhiThing(1.0, (1,)) >= PhiThing(1.0, (1,))
    assert PhiThing(1.0, (1, 2)) >= PhiThing(1.0, (1,))

    class PhiLikeThing(PhiThing):
        pass

    # Compared objects must be of the same class
    with pytest.raises(TypeError):  # TypeError: unorderable types
        PhiThing(1.0, (1, 2)) <= PhiLikeThing(1.0, (1, 2))

    class PhiThang(PhiThing):
        def __init__(self, phi, mechanism, purview):
            super().__init__(phi, mechanism)
            self.purview = purview

        def __eq__(self, other):
            return self.purview == other.purview

    assert PhiThang(1.0, (1,), (1,)) == PhiThang(2.0, (3,), (1,))
    assert PhiThang(1.0, (1,), (1,)) < PhiThang(2.0, (1,), (2,))


def test_sametype_decorator():

    class Thing:
        @models.cmp.sametype
        def do_it(self, other):
            return True

    assert Thing().do_it(object()) == NotImplemented


def test_numpy_aware_eq_noniterable():
    b = 1
    assert not models.cmp._numpy_aware_eq(a, b)


def test_numpy_aware_eq_nparray():
    b = np.arange(3)
    assert not models.cmp._numpy_aware_eq(a, b)


def test_numpy_aware_eq_tuple_nparrays():
    b = (np.arange(3), np.arange(3))
    assert not models.cmp._numpy_aware_eq(a, b)


def test_numpy_aware_eq_identical():
    b = a
    assert models.cmp._numpy_aware_eq(a, b)


def test_general_eq_different_attributes():
    similar_nt = namedtuple('nt', nt_attributes + ['supbro'])
    b = similar_nt(a.this, a.that, a.phi, a.mechanism, a.purview,
                   supbro="nothin' much")
    assert models.cmp._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_true():
    b = nt(a.this, a.that, (a.phi - constants.EPSILON/2), a.mechanism,
           a.purview)
    assert models.cmp._general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_false():
    b = nt(a.this, a.that, (a.phi - constants.EPSILON*2), a.mechanism,
           a.purview)
    assert not models.cmp._general_eq(a, b, nt_attributes)


def test_general_eq_different_mechanism_order():
    b = nt(a.this, a.that, a.phi, a.mechanism[::-1], a.purview)
    assert models.cmp._general_eq(a, b, nt_attributes)


def test_general_eq_different_purview_order():
    b = nt(a.this, a.that, a.phi, a.mechanism, a.purview[::-1])
    assert models.cmp._general_eq(a, b, nt_attributes)


def test_general_eq_different_mechanism_and_purview_order():
    b = nt(a.this, a.that, a.phi, a.mechanism[::-1], a.purview[::-1])
    assert models.cmp._general_eq(a, b, nt_attributes)


# }}}

# Test Cut {{{
# ============

def test_cut_splits_mechanism():
    cut = models.Cut((0,), (1, 2))
    assert cut.splits_mechanism((0, 1))
    assert not cut.splits_mechanism((0,))
    assert not cut.splits_mechanism((1, 2))


def test_cut_splits_connections():
    cut = models.Cut((0, 3), (1, 2))
    assert cut.cuts_connections((0,), (1, 2))
    assert cut.cuts_connections((0, 3), (1,))
    assert not cut.cuts_connections((1, 2), (0,))
    assert not cut.cuts_connections((1,), (0, 3))


def test_cut_all_cut_mechanisms():
    cut = models.Cut((0,), (1, 2))
    assert cut.all_cut_mechanisms() == ((0, 1), (0, 2), (0, 1, 2))

    cut = models.Cut((1,), (5,))
    assert cut.all_cut_mechanisms() == ((1, 5),)


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


def test_cut_indices():
    cut = models.Cut((0,), (1, 2))
    assert cut.indices == (0, 1, 2)
    cut = models.Cut((7,), (3, 1))
    assert cut.indices == (1, 3, 7)

# }}}


def mip(phi=1.0, dir=None, mech=(), purv=(), partition=None,
        unpartitioned_repertoire=None, partitioned_repertoire=None):
    return models.Mip(phi=phi, direction=dir, mechanism=mech,
                      purview=purv, partition=partition,
                      unpartitioned_repertoire=unpartitioned_repertoire,
                      partitioned_repertoire=partitioned_repertoire)


# Test MIP {{{
# ============

def test_mip_ordering_and_equality():
    assert mip(phi=1.0) < mip(phi=2.0)
    assert mip(phi=2.0) > mip(phi=1.0)
    assert mip(phi=1.0, mech=(1,)) < mip(phi=1.0, mech=(1, 2))
    assert mip(phi=1.0, mech=(1, 2)) >= mip(phi=1.0, mech=(1,))
    assert mip(phi=1.0, mech=(1,), purv=(1,)) < mip(phi=1.0, mech=(1,), purv=(1, 2))
    assert mip(phi=1.0, mech=(1,), purv=(1, 2)) >= mip(phi=1.0, mech=(1,), purv=(1,))

    assert mip(phi=1.0) == mip(phi=1.0)
    assert mip(phi=1.0) == mip(phi=(1.0 - constants.EPSILON/2))
    assert mip(phi=1.0) != mip(phi=(1.0 - constants.EPSILON * 2))
    assert mip(dir=Direction.PAST) != mip(dir=Direction.FUTURE)
    assert mip(mech=(1,)) != mip(mech=(1, 2))

    with pytest.raises(TypeError):
        mip(dir=Direction.PAST) < mip(dir=Direction.FUTURE)

    with pytest.raises(TypeError):
        mip(dir=Direction.PAST) >= mip(dir=Direction.FUTURE)


def test_null_mip():
    direction = Direction.PAST
    mechanism = (0,)
    purview = (1,)
    unpartitioned_repertoire = 'repertoire'
    null_mip = models._null_mip(direction, mechanism, purview,
                                unpartitioned_repertoire)
    assert null_mip.direction == direction
    assert null_mip.mechanism == mechanism
    assert null_mip.purview == purview
    assert null_mip.partition is None
    assert null_mip.unpartitioned_repertoire == 'repertoire'
    assert null_mip.partitioned_repertoire is None
    assert null_mip.phi == 0


def test_mip_repr_str():
    print(repr(mip()))
    print(str(mip()))


# }}}

# Test MICE {{{
# =============

def test_mice_ordering_by_phi():
    phi1 = models.Mice(mip())
    different_phi1 = models.Mice(mip(dir='different'))
    phi2 = models.Mice(mip(phi=(1.0 + constants.EPSILON * 2), partition=()))
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    with pytest.raises(TypeError):
        phi1 <= different_phi1

    with pytest.raises(TypeError):
        phi1 >= different_phi1


def test_mice_odering_by_mechanism():
    small = models.Mice(mip(mech=(1,)))
    big = models.Mice(mip(mech=(1, 2, 3)))
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_mice_ordering_by_purview():
    small = models.Mice(mip(purv=(1, 2)))
    big = models.Mice(mip(purv=(1, 2, 3)))
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small


def test_mice_equality():
    mice = models.Mice(mip(phi=1.0))
    close_enough = models.Mice(mip(phi=(1.0 - constants.EPSILON / 2)))
    not_quite = models.Mice(mip(phi=(1.0 - constants.EPSILON * 2)))
    assert mice == close_enough
    assert mice != not_quite


def test_mice_repr_str():
    mice = models.Mice(mip())
    print(repr(mice))
    print(str(mice))


def test_relevant_connections(s, subsys_n1n2):
    mice = models.Mice(mip(mech=(0,), purv=(1,), dir=Direction.PAST))
    answer = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    assert np.array_equal(mice._relevant_connections(s), answer)

    mice = models.Mice(mip(mech=(1,), purv=(1, 2), dir=Direction.FUTURE))
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
    mice = models.Mice(mip(mech=(0, 1), purv=(1, 2), dir=Direction.FUTURE))
    assert mice.damaged_by_cut(subsys)
    assert not mice.damaged_by_cut(s)

    # Cut splits mechanism & purview (but not *only* mechanism)
    mice = models.Mice(mip(mech=(0,), purv=(1, 2), dir=Direction.FUTURE))
    assert mice.damaged_by_cut(subsys)
    assert not mice.damaged_by_cut(s)


# }}}

# Test Concept {{{
# ================

def test_concept_ordering(s, micro_s):
    phi1 = models.Concept(
        mechanism=(0, 1), cause=1, effect=None, subsystem=s,
        phi=1.0)
    different_phi1 = models.Concept(
        mechanism=(0, 1), cause='different', effect=None, subsystem=micro_s,
        phi=1.0)
    phi2 = models.Concept(
        mechanism=(0,), cause='stilldifferent', effect=None, subsystem=s,
        phi=1.0 + constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    with pytest.raises(TypeError):
        phi1 <= different_phi1
    with pytest.raises(TypeError):
        phi1 > different_phi1


def test_concept_odering_by_mechanism(s):
    phi = 1.0
    small = models.Concept(mechanism=(0, 1), cause=None, effect=None,
                           subsystem=s, phi=phi)
    big = models.Concept(mechanism=(0, 1, 3), cause=None, effect=None,
                         subsystem=s, phi=phi)
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
    mice1 = models.Mice(mip(phi=phi, purv=(1, 2)))
    mice2 = models.Mice(mip(phi=phi, purv=(1,)))
    concept = models.Concept(mechanism=(), cause=mice1, effect=None,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(), cause=mice2, effect=None,
                             subsystem=s, phi=phi)
    assert concept != another


def test_concept_equality_effect_purview_nodes(s):
    phi = 1.0
    mice1 = models.Mice(mip(phi=phi, purv=(1, 2)))
    mice2 = models.Mice(mip(phi=phi, purv=(1,)))
    concept = models.Concept(mechanism=(), cause=None, effect=mice1,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(), cause=None, effect=mice2,
                             subsystem=s, phi=phi)
    assert concept != another


def test_concept_equality_repertoires(s):
    phi = 1.0
    mice1 = models.Mice(mip(phi=phi,
                            unpartitioned_repertoire=np.array([1, 2]),
                            partitioned_repertoire=()))
    mice2 = models.Mice(mip(phi=phi,
                            unpartitioned_repertoire=np.array([0, 0]), partitioned_repertoire=None))
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
    mice = models.Mice(mip(mech=(), purv=(1, 2), phi=phi))
    concept = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=subsys_n1n2, phi=phi)
    assert concept == another


def test_concept_repr_str():
    mice = models.Mice(mip())
    concept = models.Concept(mechanism=(), cause=mice, effect=mice,
                             subsystem=None, phi=0.0)
    print(repr(concept))
    print(str(concept))


def test_concept_hashing(s):
    mice = models.Mice(mip(mech=(0, 1, 2), purv=(0, 1, 2)))
    concept = models.Concept(mechanism=(0, 1, 2), cause=mice, effect=mice,
                             subsystem=s, phi=0.0)
    hash(concept)


def test_concept_hashing_one_subsystem_is_subset_of_another(s, subsys_n1n2):
    phi = 1.0
    mice = models.Mice(mip(mech=(), purv=(1, 2), phi=phi))
    concept = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=s, phi=phi)
    another = models.Concept(mechanism=(2,), cause=mice, effect=mice,
                             subsystem=subsys_n1n2, phi=phi)
    assert hash(concept) == hash(another)
    assert(len(set([concept, another])) == 1)


def test_concept_emd_eq(s, subsys_n1n2):
    mice = models.Mice(mip(mech=(1,)))
    concept = models.Concept(phi=1.0, mechanism=(1,), cause=mice, effect=mice,
                             subsystem=s)

    # Same repertoires, mechanism, phi
    another = models.Concept(phi=1.0, mechanism=(1,), cause=mice, effect=mice,
                             subsystem=subsys_n1n2)
    assert concept.emd_eq(another)

    # Everything equal except phi
    another = models.Concept(phi=2.0, mechanism=(1,), cause=mice, effect=mice,
                             subsystem=s)
    assert not concept.emd_eq(another)

    # TODO: test other expectations...

# }}}

# Test Constellation {{{
# ======================

def test_constellation_is_still_a_tuple():
    c = models.Constellation([models.Concept()])
    assert len(c) == 1


@config.override(REPR_VERBOSITY=0)
def test_constellation_repr():
    c = models.Constellation()
    assert repr(c) == "Constellation(())"


def test_normalize_constellation():
    c1 = models.Concept(mechanism=(2,))
    c2 = models.Concept(mechanism=(1, 3))
    assert (c2, c1) == models.normalize_constellation((c1, c2))

# }}}

# Test BigMip {{{
# ===============

def test_bigmip_ordering(s, s_noised):
    phi1 = models.BigMip(
        unpartitioned_constellation=None, partitioned_constellation=None,
        subsystem=s, cut_subsystem=(),
        phi=1.0)
    different_phi1 = models.BigMip(
        unpartitioned_constellation=0, partitioned_constellation=None,
        subsystem=s_noised, cut_subsystem=(),
        phi=1.0)
    phi2 = models.BigMip(
        unpartitioned_constellation='stilldifferent',
        partitioned_constellation=None, subsystem=s, cut_subsystem=(),
        phi=1.0 + constants.EPSILON*2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1
    with pytest.raises(TypeError):
        phi1 <= different_phi1
    with pytest.raises(TypeError):
        phi1 >= different_phi1


def test_bigmip_ordering_by_subsystem_size(s, s_single):
    small = models.BigMip(
        unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=s_single, cut_subsystem=(),
        phi=1.0)
    big = models.BigMip(
        unpartitioned_constellation=None,
        partitioned_constellation=None, subsystem=s, cut_subsystem=(),
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
    assert models.fmt.indent(s) == answer


class ReadableReprClass:
    """Dummy class for make_repr tests"""
    some_attr = 3.14

    def __repr__(self):
        return models.fmt.make_repr(self, ['some_attr'])

    def __str__(self):
        return "A nice fat explicit string"


@config.override(REPR_VERBOSITY=0)
def test_make_reprs_uses___repr__():
    assert repr(ReadableReprClass()) == "ReadableReprClass(some_attr=3.14)"


@config.override(REPR_VERBOSITY=2)
def test_make_reprs_calls_out_to_string():
    assert repr(ReadableReprClass()) == "A nice fat explicit string"

# }}}


# Test partitions
# ===============

@pytest.fixture
def bipartition():
    return models.Bipartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)))


def test_bipartition_properties(bipartition):
    assert bipartition.mechanism == (0,)
    assert bipartition.purview == (0, 1, 4)


def test_bipartition_str(bipartition):
    assert str(bipartition) == (
        " 0    []\n"
        "--- X --\n"
        "0,4   1 ")


@pytest.fixture
def tripartition():
    return models.Tripartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)),
        models.Part((2,), (2,)))


def test_tripartion_properties(tripartition):
    assert tripartition.mechanism == (0, 2)
    assert tripartition.purview == (0, 1, 2, 4)


def test_tripartition_str(tripartition):
    assert str(tripartition) == (
        " 0    []   2\n"
        "--- X -- X -\n"
        "0,4   1    2")

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
