#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_models.py

from collections import namedtuple

import numpy as np
import pytest

from pyphi import Direction, Subsystem, config, constants, exceptions, models
from pyphi.labels import NodeLabels

# Helper functions for constructing PyPhi objects
# -----------------------------------------------

def ria(phi=1.0, direction=None, mechanism=(), purview=(), partition=None,
        repertoire=None, partitioned_repertoire=None):
    """Build a ``RepertoireIrreducibilityAnalysis``."""
    return models.RepertoireIrreducibilityAnalysis(
        phi=phi, direction=direction, mechanism=mechanism, purview=purview,
        partition=partition, repertoire=repertoire,
        partitioned_repertoire=partitioned_repertoire
    )


def mice(**kwargs):
    """Build a ``MaximallyIrreducibleCauseOrEffect``."""
    return models.MaximallyIrreducibleCauseOrEffect(ria(**kwargs))


def mic(**kwargs):
    """Build a ``MIC``."""
    return models.MaximallyIrreducibleCause(ria(**kwargs))


def mie(**kwargs):
    """Build a ``MIE``."""
    return models.MaximallyIrreducibleEffect(ria(**kwargs))


def concept(mechanism=(0, 1), cause_purview=(1,), effect_purview=(1,), phi=1.0,
            subsystem=None):
    """Build a ``Concept``."""
    return models.Concept(
        mechanism=mechanism,
        cause=mic(mechanism=mechanism, purview=cause_purview, phi=phi,
                  direction=Direction.CAUSE),
        effect=mie(mechanism=mechanism, purview=effect_purview, phi=phi,
                   direction=Direction.EFFECT),
        subsystem=subsystem)


def sia(ces=(), partitioned_ces=(), subsystem=None, cut_subsystem=None,
        phi=1.0):
    """Build a ``SystemIrreducibilityAnalysis``."""
    cut_subsystem = cut_subsystem or subsystem

    return models.SystemIrreducibilityAnalysis(
        ces=ces,
        partitioned_ces=partitioned_ces,
        subsystem=subsystem, cut_subsystem=cut_subsystem, phi=phi)


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


nt_attributes = ['this', 'that', 'phi', 'mechanism', 'purview']
nt = namedtuple('nt', nt_attributes)
a = nt(this=('consciousness', 'is phi'), that=np.arange(3), phi=0.5,
       mechanism=(0, 1, 2), purview=(2, 4))


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


def test_general_eq_purview_mechanism_none():
    b = nt(a.this, a.that, a.phi, None, None)
    assert models.cmp.general_eq(b, b, nt_attributes)
    c = nt(a.this, a.that, a.phi, a.mechanism, None)
    assert not models.cmp.general_eq(a, b, nt_attributes)
    c = nt(a.this, a.that, a.phi, None, a.purview)
    assert not models.cmp.general_eq(a, c, nt_attributes)


def test_general_eq_attribute_missing():
    b = namedtuple('no_purview', nt_attributes[:-1])(
        a.this, a.that, a.phi, a.mechanism)
    assert not models.cmp.general_eq(a, b, nt_attributes)


# }}}

# Test Cut
# {{{

def test_cut_equality():
    cut1 = models.Cut((0,), (1,))
    cut2 = models.Cut((0,), (1,))
    assert cut1 == cut2
    assert hash(cut1) == hash(cut2)


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
    assert list(cut.all_cut_mechanisms()) == [(0, 1), (0, 2), (0, 1, 2)]

    cut = models.Cut((1,), (5,))
    assert list(cut.all_cut_mechanisms()) == [(1, 5)]


def test_cut_matrix():
    cut = models.Cut((), (0,))
    matrix = np.array([[0]])
    assert np.array_equal(cut.cut_matrix(1), matrix)

    cut = models.Cut((0,), (1,))
    matrix = np.array([
        [0, 1],
        [0, 0],
    ])
    assert np.array_equal(cut.cut_matrix(2), matrix)

    cut = models.Cut((0, 2), (1, 2))
    matrix = np.array([
        [0, 1, 1],
        [0, 0, 0],
        [0, 1, 1],
    ])
    assert np.array_equal(cut.cut_matrix(3), matrix)

    cut = models.Cut((), ())
    assert np.array_equal(cut.cut_matrix(0), np.ndarray(shape=(0, 0)))


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


def test_cut_is_null():
    cut = models.Cut((0,), (1, 2))
    assert not cut.is_null


def test_null_cut():
    cut = models.NullCut((2, 3))
    assert cut.indices == (2, 3)
    assert cut.is_null
    assert np.array_equal(cut.cut_matrix(4), np.zeros((4, 4)))


def test_null_cut_str():
    cut = models.NullCut((2, 3))
    assert str(cut) == 'NullCut((2, 3))'


def test_null_cut_equality():
    cut = models.NullCut((2, 3))
    other = models.NullCut((2, 3))
    assert cut == other
    assert hash(cut) == hash(other)


def test_cuts_can_have_node_labels(node_labels):
    models.NullCut((0, 1), node_labels=node_labels)
    models.Cut((0,), (1,), node_labels=node_labels)

    k_partition = models.KPartition(
        models.Part((0, 1), (0,)),
        models.Part((), (1,)),
        node_labels=node_labels)
    models.KCut(Direction.CAUSE, k_partition, node_labels=node_labels)

# }}}


# Test ria
# {{{


def test_ria_ordering_and_equality():
    assert ria(phi=1.0) < ria(phi=2.0)
    assert ria(phi=2.0) > ria(phi=1.0)
    assert ria(mechanism=(1,)) < ria(mechanism=(1, 2))
    assert ria(mechanism=(1, 2)) >= ria(mechanism=(1,))
    assert ria(purview=(1,)) < ria(purview=(1, 2))
    assert ria(purview=(1, 2)) >= ria(purview=(1,))

    assert ria(phi=1.0) == ria(phi=1.0)
    assert ria(phi=1.0) == ria(phi=(1.0 - constants.EPSILON / 2))
    assert ria(phi=1.0) != ria(phi=(1.0 - constants.EPSILON * 2))
    assert ria(direction=Direction.CAUSE) != ria(direction=Direction.EFFECT)
    assert ria(mechanism=(1,)) != ria(mechanism=(1, 2))

    with config.override(PICK_SMALLEST_PURVIEW=True):
        assert ria(purview=(1, 2)) < ria(purview=(1,))

    with pytest.raises(TypeError):
        ria(direction=Direction.CAUSE) < ria(direction=Direction.EFFECT)

    with pytest.raises(TypeError):
        ria(direction=Direction.CAUSE) >= ria(direction=Direction.EFFECT)


def test_null_ria():
    direction = Direction.CAUSE
    mechanism = (0,)
    purview = (1,)
    repertoire = 'repertoire'
    null_ria = models._null_ria(direction, mechanism, purview,
                                repertoire)
    assert null_ria.direction == direction
    assert null_ria.mechanism == mechanism
    assert null_ria.purview == purview
    assert null_ria.partition is None
    assert null_ria.repertoire == 'repertoire'
    assert null_ria.partitioned_repertoire is None
    assert null_ria.phi == 0


def test_ria_repr_str():
    print(repr(ria()))
    print(str(ria()))


# }}}

# Test MaximallyIrreducibleCauseOrEffect
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
    m = mice(mechanism=(0,), purview=(1,), direction=Direction.CAUSE)
    answer = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    assert np.array_equal(m._relevant_connections(s), answer)

    m = mice(mechanism=(1,), purview=(1, 2), direction=Direction.EFFECT)
    answer = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ])
    assert np.array_equal(m._relevant_connections(subsys_n1n2), answer)


def test_damaged(s):
    # Build cut subsystem from s
    cut = models.Cut((0,), (1, 2))
    cut_s = Subsystem(s.network, s.state, s.node_indices, cut=cut)

    # Cut splits mechanism:
    m1 = mice(mechanism=(0, 1), purview=(1, 2), direction=Direction.EFFECT)
    assert m1.damaged_by_cut(cut_s)
    assert not m1.damaged_by_cut(s)

    # Cut splits mechanism & purview (but not *only* mechanism)
    m2 = mice(mechanism=(0,), purview=(1, 2), direction=Direction.EFFECT)
    assert m2.damaged_by_cut(cut_s)
    assert not m2.damaged_by_cut(s)


# }}}


# Test MIC and MIE {{{

def test_mic_raises_wrong_direction():
    mic(direction=Direction.CAUSE, mechanism=(0,), purview=(1,))
    with pytest.raises(exceptions.WrongDirectionError):
        mic(direction=Direction.EFFECT, mechanism=(0,), purview=(1,))


def test_mie_raises_wrong_direction():
    mie(direction=Direction.EFFECT, mechanism=(0,), purview=(1,))
    with pytest.raises(exceptions.WrongDirectionError):
        mie(direction=Direction.CAUSE, mechanism=(0,), purview=(1,))

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
    big = concept(mechanism=(0, 1, 2), subsystem=s)
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
    mice1 = mice(phi=phi, repertoire=np.array([1, 2]),
                 partitioned_repertoire=())
    mice2 = mice(phi=phi, repertoire=np.array([0, 0]),
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


def test_concept_repr_str(s):
    print(repr(concept(subsystem=s)))
    print(str(concept(subsystem=s)))


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


# Test CauseEffectStructure
# {{{

def test_ces_is_still_a_tuple(s):
    c = models.CauseEffectStructure([concept(subsystem=s)], subsystem=s)
    assert len(c) == 1


def test_ces_repr_str(s):
    c = models.CauseEffectStructure([concept(subsystem=s)])
    repr(c)
    str(c)


def test_ces_are_always_normalized(s):
    c1 = concept(mechanism=(0,), subsystem=s)
    c2 = concept(mechanism=(1,), subsystem=s)
    c3 = concept(mechanism=(0, 2), subsystem=s)
    c4 = concept(mechanism=(0, 1, 2), subsystem=s)
    assert (c1, c2, c3, c4) == models.CauseEffectStructure((c3, c4, c2, c1)).concepts


def test_ces_labeled_mechanisms(s):
    c = models.CauseEffectStructure([concept(subsystem=s)], subsystem=s)
    assert c.labeled_mechanisms == (['A', 'B'],)


def test_ces_ordering(s):
    assert (models.CauseEffectStructure([concept(subsystem=s)], subsystem=s) ==
            models.CauseEffectStructure([concept(subsystem=s)], subsystem=s))

    assert (models.CauseEffectStructure([concept(phi=1, subsystem=s)],
                                        subsystem=s) >
            models.CauseEffectStructure([concept(phi=0, subsystem=s)],
                                        subsystem=s))

# }}}


# Test SystemIrreducibilityAnalysis
# {{{


def test_sia_ordering(s, s_noised, subsys_n0n2, subsys_n1n2):
    phi1 = sia(subsystem=s)
    phi2 = sia(subsystem=s, phi=1.0 + constants.EPSILON * 2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    assert sia(subsystem=subsys_n0n2) < sia(subsystem=subsys_n1n2)

    different_system = sia(subsystem=s_noised)
    with pytest.raises(TypeError):
        phi1 <= different_system
    with pytest.raises(TypeError):
        phi1 >= different_system


def test_sia_ordering_by_subsystem_size(s, s_single):
    small = sia(subsystem=s_single)
    big = sia(subsystem=s)
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert big != small


def test_sia_equality(s):
    bm = sia(subsystem=s)
    close_enough = sia(subsystem=s, phi=(1.0 - constants.EPSILON / 2))
    not_quite = sia(subsystem=s, phi=(1.0 - constants.EPSILON * 2))
    assert bm == close_enough
    assert bm != not_quite


def test_sia_repr_str(s):
    bm = sia(subsystem=s)
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
    """Dummy class for make_repr tests"""
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
def node_labels():
    return NodeLabels('ABCDE', tuple(range(5)))


@pytest.fixture
def bipartition(node_labels):
    return models.Bipartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)),
        node_labels=node_labels)


def test_bipartition_properties(bipartition):
    assert bipartition.mechanism == (0,)
    assert bipartition.purview == (0, 1, 4)


def test_bipartition_str(bipartition):
    assert str(bipartition) == (
        ' A     ∅ \n'
        '─── ✕ ───\n'
        'A,E    B ')


@pytest.fixture
def tripartition(node_labels):
    return models.Tripartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)),
        models.Part((2,), (2,)),
        node_labels=node_labels)


def test_tripartion_properties(tripartition):
    assert tripartition.mechanism == (0, 2)
    assert tripartition.purview == (0, 1, 2, 4)


def test_tripartion_str(tripartition):
    assert str(tripartition) == (
        ' A     ∅     C \n'
        '─── ✕ ─── ✕ ───\n'
        'A,E    B     C ')


def k_partition(node_labels=None):
    return models.KPartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)),
        models.Part((6,), (5,)),
        models.Part((2,), (2,)),
        node_labels=node_labels
    )


def test_partition_normalize():
    assert k_partition().normalize() == models.KPartition(
        models.Part((), (1,)),
        models.Part((0,), (0, 4)),
        models.Part((2,), (2,)),
        models.Part((6,), (5,))
    )


def test_partition_normalize_preserves_labels(node_labels):
    k = k_partition(node_labels=node_labels)
    assert k.normalize().node_labels == k.node_labels


def test_partition_eq_hash():
    assert k_partition() == k_partition()
    assert hash(k_partition()) == hash(k_partition())

# }}}

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
