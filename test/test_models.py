#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_models.py

from collections import namedtuple

import numpy as np
import pytest

from pyphi import Subsystem, config, constants, models
from pyphi.constants import Direction


# Helper functions for constructing PyPhi objects
# -----------------------------------------------

def mip(phi=1.0, direction=None, mechanism=(), purview=(), partition=None,
        unpartitioned_repertoire=None, partitioned_repertoire=None):
    '''Build a ``Mip``.'''
    return models.Mip(phi=phi, direction=direction, mechanism=mechanism,
                      purview=purview, partition=partition,
                      unpartitioned_repertoire=unpartitioned_repertoire,
                      partitioned_repertoire=partitioned_repertoire)


def mice(**kwargs):
    '''Build a ``Mice``.'''
    return models.Mice(mip(**kwargs))


def concept(mechanism=(0, 1), cause_purview=(1,), effect_purview=(1,), phi=1.0,
            subsystem=None):
    '''Build a ``Concept``.'''
    return models.Concept(
        mechanism=mechanism,
        cause=mice(mechanism=mechanism, purview=cause_purview, phi=phi),
        effect=mice(mechanism=mechanism, purview=effect_purview, phi=phi),
        subsystem=subsystem)


def bigmip(unpartitioned_constellation=(), partitioned_constellation=(),
           subsystem=None, cut_subsystem=None, phi=1.0):
    '''Build a ``BigMip``.'''
    cut_subsystem = cut_subsystem or subsystem

    return models.BigMip(
        unpartitioned_constellation=unpartitioned_constellation,
        partitioned_constellation=partitioned_constellation,
        subsystem=subsystem, cut_subsystem=cut_subsystem, phi=phi)


nt_attributes = ['this', 'that', 'phi', 'mechanism', 'purview']
nt = namedtuple('nt', nt_attributes)
a = nt(this=('consciousness', 'is phi'), that=np.arange(3), phi=0.5,
       mechanism=(0, 1, 2), purview=(2, 4))


# Test equality helpers
# {{{

def test_phi_mechanism_ordering():

    class PhiThing(models.cmp.Orderable):
        def __init__(self, phi, mechanism):
            self.phi = phi
            self.mechanism = mechanism

        def order_by(self):
            return [self.phi, self.mechanism]

        def __eq__(self, other):
            return self.phi == other.phi and self.mechanism == other.mechanism

    assert PhiThing(1.0, (1,)) == PhiThing(1.0, (1,))
    assert PhiThing(1.0, (1,)) != PhiThing(1.0, (1, 2))
    assert PhiThing(1.0, (1,)) != PhiThing(2.0, (1, 2))
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
    assert not models.cmp.numpy_aware_eq(a, b)


def test_numpy_aware_eq_nparray():
    b = np.arange(3)
    assert not models.cmp.numpy_aware_eq(a, b)


def test_numpy_aware_eq_tuple_nparrays():
    b = (np.arange(3), np.arange(3))
    assert not models.cmp.numpy_aware_eq(a, b)


def test_numpy_aware_eq_identical():
    b = a
    assert models.cmp.numpy_aware_eq(a, b)


def test_general_eq_different_attributes():
    similar_nt = namedtuple('nt', nt_attributes + ['supbro'])
    b = similar_nt(a.this, a.that, a.phi, a.mechanism, a.purview,
                   supbro="nothin' much")
    assert models.cmp.general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_true():
    b = nt(a.this, a.that, (a.phi - constants.EPSILON / 2), a.mechanism,
           a.purview)
    assert models.cmp.general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_false():
    b = nt(a.this, a.that, (a.phi - constants.EPSILON * 2), a.mechanism,
           a.purview)
    assert not models.cmp.general_eq(a, b, nt_attributes)


def test_general_eq_different_mechanism_order():
    b = nt(a.this, a.that, a.phi, a.mechanism[::-1], a.purview)
    assert models.cmp.general_eq(a, b, nt_attributes)


def test_general_eq_different_purview_order():
    b = nt(a.this, a.that, a.phi, a.mechanism, a.purview[::-1])
    assert models.cmp.general_eq(a, b, nt_attributes)


def test_general_eq_different_mechanism_and_purview_order():
    b = nt(a.this, a.that, a.phi, a.mechanism[::-1], a.purview[::-1])
    assert models.cmp.general_eq(a, b, nt_attributes)


# }}}

# Test Cut
# {{{

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


def test_apply_cut():
    cm = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    cut = models.Cut(from_nodes=(0, 3), to_nodes=(1, 2))
    cut_cm = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0]
    ])
    assert np.array_equal(cut.apply_cut(cm), cut_cm)


# }}}


# Test MIP
# {{{


def test_mip_ordering_and_equality():
    assert mip(phi=1.0) < mip(phi=2.0)
    assert mip(phi=2.0) > mip(phi=1.0)
    assert mip(mechanism=(1,)) < mip(mechanism=(1, 2))
    assert mip(mechanism=(1, 2)) >= mip(mechanism=(1,))
    assert mip(purview=(1,)) < mip(purview=(1, 2))
    assert mip(purview=(1, 2)) >= mip(purview=(1,))

    assert mip(phi=1.0) == mip(phi=1.0)
    assert mip(phi=1.0) == mip(phi=(1.0 - constants.EPSILON / 2))
    assert mip(phi=1.0) != mip(phi=(1.0 - constants.EPSILON * 2))
    assert mip(direction=Direction.PAST) != mip(direction=Direction.FUTURE)
    assert mip(mechanism=(1,)) != mip(mechanism=(1, 2))

    with config.override(PICK_SMALLEST_PURVIEW=True):
        assert mip(purview=(1, 2)) < mip(purview=(1,))

    with pytest.raises(TypeError):
        mip(direction=Direction.PAST) < mip(direction=Direction.FUTURE)

    with pytest.raises(TypeError):
        mip(direction=Direction.PAST) >= mip(direction=Direction.FUTURE)


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

# Test MICE
# {{{


def test_mice_ordering_by_phi():
    phi1 = mice()
    phi2 = mice(phi=(1.0 + constants.EPSILON * 2), partition=())
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    different_direction = mice(direction='different')

    with pytest.raises(TypeError):
        phi1 <= different_direction

    with pytest.raises(TypeError):
        phi1 >= different_direction


def test_mice_odering_by_mechanism():
    small = mice(mechanism=(1,))
    big = mice(mechanism=(1, 2, 3))
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_mice_ordering_by_purview():
    small = mice(purview=(1, 2))
    big = mice(purview=(1, 2, 3))
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small


def test_mice_equality():
    m = mice(phi=1.0)
    close_enough = mice(phi=(1.0 - constants.EPSILON / 2))
    not_quite = mice(phi=(1.0 - constants.EPSILON * 2))
    assert m == close_enough
    assert m != not_quite


def test_mice_repr_str():
    print(repr(mice()))
    print(str(mice()))


def test_relevant_connections(s, subsys_n1n2):
    m = mice(mechanism=(0,), purview=(1,), direction=Direction.PAST)
    answer = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    assert np.array_equal(m._relevant_connections(s), answer)

    m = mice(mechanism=(1,), purview=(1, 2), direction=Direction.FUTURE)
    answer = np.array([
        [1, 1],
        [0, 0],
    ])
    assert np.array_equal(m._relevant_connections(subsys_n1n2), answer)


def test_damaged(s):
    # Build cut subsystem from s
    cut = models.Cut((0,), (1, 2))
    cut_s = Subsystem(s.network, s.state, s.node_indices, cut=cut)

    # Cut splits mechanism:
    m1 = mice(mechanism=(0, 1), purview=(1, 2), direction=Direction.FUTURE)
    assert m1.damaged_by_cut(cut_s)
    assert not m1.damaged_by_cut(s)

    # Cut splits mechanism & purview (but not *only* mechanism)
    m2 = mice(mechanism=(0,), purview=(1, 2), direction=Direction.FUTURE)
    assert m2.damaged_by_cut(cut_s)
    assert not m2.damaged_by_cut(s)


# }}}


# Test Concept
# {{{


def test_concept_ordering(s, micro_s):
    phi1 = concept(subsystem=s)
    phi2 = concept(mechanism=(0,), phi=(1.0 + constants.EPSILON * 2),
                   subsystem=s)

    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    micro_phi1 = concept(subsystem=micro_s)

    with pytest.raises(TypeError):
        phi1 <= micro_phi1
    with pytest.raises(TypeError):
        phi1 > micro_phi1


def test_concept_ordering_by_mechanism(s):
    small = concept(mechanism=(0, 1), subsystem=s)
    big = concept(mechanism=(0, 1, 3), subsystem=s)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_concept_equality(s):
    assert concept(subsystem=s) == concept(subsystem=s)


def test_concept_equality_phi(s):
    assert concept(phi=1.0, subsystem=s) != concept(phi=0.0, subsystem=s)


def test_concept_equality_cause_purview_nodes(s):
    assert (concept(cause_purview=(1, 2), subsystem=s) !=
            concept(cause_purview=(1,), subsystem=s))


def test_concept_equality_effect_purview_nodes(s):
    assert (concept(effect_purview=(1, 2), subsystem=s) !=
            concept(effect_purview=(1,), subsystem=s))


def test_concept_equality_repertoires(s):
    phi = 1.0
    mice1 = mice(phi=phi, unpartitioned_repertoire=np.array([1, 2]),
                 partitioned_repertoire=())
    mice2 = mice(phi=phi, unpartitioned_repertoire=np.array([0, 0]),
                 partitioned_repertoire=None)
    concept = models.Concept(mechanism=(), cause=mice1, effect=mice2,
                             subsystem=s)
    another = models.Concept(mechanism=(), cause=mice2, effect=mice1,
                             subsystem=s)
    assert concept != another


def test_concept_equality_network(s, simple_subsys_all_off):
    assert concept(subsystem=simple_subsys_all_off) != concept(subsystem=s)


def test_concept_equality_one_subsystem_is_subset_of_another(s, subsys_n1n2):
    assert concept(subsystem=s) == concept(subsystem=subsys_n1n2)


def test_concept_repr_str():
    print(repr(concept()))
    print(str(concept()))


def test_concept_hashing(s):
    hash(concept(subsystem=s))


def test_concept_hashing_one_subsystem_is_subset_of_another(s, subsys_n1n2):
    c1 = concept(subsystem=s)
    c2 = concept(subsystem=subsys_n1n2)
    assert hash(c1) == hash(c2)
    assert len(set([c1, c2])) == 1


def test_concept_emd_eq(s, subsys_n1n2):
    c1 = concept(subsystem=s)

    # Same repertoires, mechanism, phi
    c2 = concept(subsystem=subsys_n1n2)
    assert c1.emd_eq(c2)

    # Everything equal except phi
    c3 = concept(phi=2.0, subsystem=s)
    assert not c1.emd_eq(c3)

    # TODO: test other expectations...

# }}}


# Test Constellation
# {{{

def test_constellation_is_still_a_tuple():
    c = models.Constellation([models.Concept()])
    assert len(c) == 1


@config.override(REPR_VERBOSITY=0)
def test_constellation_repr():
    c = models.Constellation()
    assert repr(c) == "Constellation()"


def test_normalize_constellation():
    c1 = models.Concept(mechanism=(1,))
    c2 = models.Concept(mechanism=(2,))
    c3 = models.Concept(mechanism=(1, 3))
    c4 = models.Concept(mechanism=(1, 2, 3))
    assert (c1, c2, c3, c4) == models.normalize_constellation((c3, c4, c2, c1))

# }}}


# Test BigMip
# {{{


def test_bigmip_ordering(s, s_noised):
    phi1 = bigmip(subsystem=s)
    phi2 = bigmip(subsystem=s, phi=1.0 + constants.EPSILON * 2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    different_system = bigmip(subsystem=s_noised)
    with pytest.raises(TypeError):
        phi1 <= different_system
    with pytest.raises(TypeError):
        phi1 >= different_system


def test_bigmip_ordering_by_subsystem_size(s, s_single):
    small = bigmip(subsystem=s_single)
    big = bigmip(subsystem=s)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_bigmip_equality(s):
    bm = bigmip(subsystem=s)
    close_enough = bigmip(subsystem=s, phi=(1.0 - constants.EPSILON / 2))
    not_quite = bigmip(subsystem=s, phi=(1.0 - constants.EPSILON * 2))
    assert bm == close_enough
    assert bm != not_quite


def test_bigmip_repr_str(s):
    bm = bigmip(subsystem=s)
    print(repr(bm))
    print(str(bm))


# }}}


# Test model __str__ and __reprs__
# {{{

def test_indent():
    s = ('line1\n'
         'line2')
    answer = ('  line1\n'
              '  line2')
    assert models.fmt.indent(s) == answer


class ReadableReprClass:
    '''Dummy class for make_repr tests'''
    some_attr = 3.14

    def __repr__(self):
        return models.fmt.make_repr(self, ['some_attr'])

    def __str__(self):
        return 'A nice fat explicit string'


@config.override(REPR_VERBOSITY=0)
def test_make_reprs_uses___repr__():
    assert repr(ReadableReprClass()) == 'ReadableReprClass(some_attr=3.14)'


@config.override(REPR_VERBOSITY=2)
def test_make_reprs_calls_out_to_string():
    assert repr(ReadableReprClass()) == 'A nice fat explicit string'

# }}}


# Test partitions
# {{{

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
        ' 0     ∅ \n'
        '─── ✕ ───\n'
        '0,4    1 ')


@pytest.fixture
def tripartition():
    return models.Tripartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)),
        models.Part((2,), (2,)))


def test_tripartion_properties(tripartition):
    assert tripartition.mechanism == (0, 2)
    assert tripartition.purview == (0, 1, 2, 4)


def test_tripartion_str(tripartition):
    assert str(tripartition) == (
        ' 0     ∅     2 \n'
        '─── ✕ ─── ✕ ───\n'
        '0,4    1     2 ')

# }}}

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
