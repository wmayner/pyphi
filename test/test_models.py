from collections import namedtuple

import numpy as np
import pytest

from pyphi import Direction
from pyphi import System
from pyphi import config
from pyphi import exceptions
from pyphi import models  # used by other tests in this module
from pyphi.labels import NodeLabels
from pyphi.models import DirectedBipartition
from pyphi.models.partitions import JointPartition

EPSILON = 10 ** (-config.numerics.precision)


# Helper functions for constructing PyPhi objects


def ria(
    phi=1.0,
    direction=None,
    mechanism=(),
    purview=(),
    partition=None,
    repertoire=None,
    partitioned_repertoire=None,
    node_labels=None,
):
    """Build a ``RepertoireIrreducibilityAnalysis``."""
    return models.RepertoireIrreducibilityAnalysis(
        phi=phi,
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=partition,
        repertoire=repertoire,
        partitioned_repertoire=partitioned_repertoire,
        node_labels=node_labels,
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


def concept(
    mechanism=(0, 1), cause_purview=(1,), effect_purview=(1,), phi=1.0, system=None
):
    """Build a ``Concept``."""
    # Extract node_labels from system if available
    node_labels = system.node_labels if system is not None else None
    return models.Concept(
        mechanism=mechanism,
        cause=mic(
            mechanism=mechanism,
            purview=cause_purview,
            phi=phi,
            direction=Direction.CAUSE,
            node_labels=node_labels,
        ),
        effect=mie(
            mechanism=mechanism,
            purview=effect_purview,
            phi=phi,
            direction=Direction.EFFECT,
            node_labels=node_labels,
        ),
    )


def sia(ces=(), partitioned_ces=(), system=None, cut_system=None, phi=1.0):
    """Build a ``SystemIrreducibilityAnalysis``."""
    cut_system = cut_system or system

    return models.SystemIrreducibilityAnalysis(
        ces=ces,
        partitioned_ces=partitioned_ces,
        system=system,
        cut_system=cut_system,
        phi=phi,
    )


# Test equality helpers


def test_ria_signed_phi_clamps_phi_to_positive_part():
    """Negative signed_phi yields phi=0; positive signed_phi yields phi=signed_phi.

    The canonical ``RIA.phi`` is the paper-faithful ``|·|+`` clamped value;
    ``signed_phi`` preserves the raw value (which may be negative under
    preventative-cause semantics).
    """
    pos = ria(phi=0.5, direction=Direction.CAUSE, mechanism=(0,), purview=(1,))
    assert float(pos.phi) == pytest.approx(0.5)
    assert float(pos.signed_phi) == pytest.approx(0.5)

    neg = ria(phi=-0.3, direction=Direction.CAUSE, mechanism=(0,), purview=(1,))
    assert float(neg.phi) == pytest.approx(0.0)
    assert float(neg.signed_phi) == pytest.approx(-0.3)

    zero = ria(phi=0.0, direction=Direction.CAUSE, mechanism=(0,), purview=(1,))
    assert float(zero.phi) == pytest.approx(0.0)
    assert float(zero.signed_phi) == pytest.approx(0.0)


def test_ria_signed_phi_explicit_argument():
    """When ``signed_phi`` is passed explicitly, it overrides the default
    snapshot from ``phi``.
    """
    r = models.RepertoireIrreducibilityAnalysis(
        phi=0.0,
        direction=Direction.CAUSE,
        mechanism=(0,),
        purview=(1,),
        partition=None,
        repertoire=None,
        partitioned_repertoire=None,
        signed_phi=-0.7,
    )
    assert float(r.signed_phi) == pytest.approx(-0.7)
    assert float(r.phi) == pytest.approx(0.0)


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
        def do_it(self, other):  # noqa: ARG002
            return True

    assert Thing().do_it(object()) == NotImplemented


nt_attributes = ["this", "that", "phi", "mechanism", "purview"]
nt = namedtuple("nt", nt_attributes)
a = nt(
    this=("consciousness", "is phi"),
    that=np.arange(3),
    phi=0.5,
    mechanism=(0, 1, 2),
    purview=(2, 4),
)


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
    similar_nt = namedtuple("nt", [*nt_attributes, "supbro"])
    b = similar_nt(a.this, a.that, a.phi, a.mechanism, a.purview, supbro="nothin' much")
    assert models.cmp.general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_true():
    b = nt(a.this, a.that, (a.phi - EPSILON / 2), a.mechanism, a.purview)
    assert models.cmp.general_eq(a, b, nt_attributes)


def test_general_eq_phi_precision_comparison_false():
    b = nt(a.this, a.that, (a.phi - EPSILON * 2), a.mechanism, a.purview)
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
    b = namedtuple("no_purview", nt_attributes[:-1])(a.this, a.that, a.phi, a.mechanism)
    assert not models.cmp.general_eq(a, b, nt_attributes)


# Test Cut


def test_cut_equality():
    cut1 = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    cut2 = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    assert cut1 == cut2
    assert hash(cut1) == hash(cut2)


def test_cut_splits_mechanism():
    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    assert cut.splits_mechanism((0, 1))
    assert not cut.splits_mechanism((0,))
    assert not cut.splits_mechanism((1, 2))


def test_cut_splits_connections():
    cut = DirectedBipartition(Direction.EFFECT, (0, 3), (1, 2))
    assert cut.cuts_connections((0,), (1, 2))
    assert cut.cuts_connections((0, 3), (1,))
    assert not cut.cuts_connections((1, 2), (0,))
    assert not cut.cuts_connections((1,), (0, 3))


def test_cut_all_cut_mechanisms():
    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    assert list(cut.all_cut_mechanisms()) == [(0, 1), (0, 2), (0, 1, 2)]

    cut = DirectedBipartition(Direction.EFFECT, (1,), (5,))
    assert list(cut.all_cut_mechanisms()) == [(1, 5)]


def test_cut_matrix():
    cut = DirectedBipartition(Direction.EFFECT, (), (0,))
    matrix = np.array([[0]])
    assert np.array_equal(cut.cut_matrix(1), matrix)

    cut = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    matrix = np.array(
        [
            [0, 1],
            [0, 0],
        ]
    )
    assert np.array_equal(cut.cut_matrix(2), matrix)

    cut = DirectedBipartition(Direction.EFFECT, (0, 2), (1, 2))
    matrix = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [0, 1, 1],
        ]
    )
    assert np.array_equal(cut.cut_matrix(3), matrix)

    cut = DirectedBipartition(Direction.EFFECT, (), ())
    assert np.array_equal(cut.cut_matrix(0), np.ndarray(shape=(0, 0)))


def test_cut_indices():
    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    assert cut.indices == (0, 1, 2)
    cut = DirectedBipartition(Direction.EFFECT, (7,), (3, 1))
    assert cut.indices == (1, 3, 7)


def test_apply_cut():
    # fmt: off
    cm = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])
    # fmt: on
    cut = DirectedBipartition(Direction.EFFECT, from_nodes=(0, 3), to_nodes=(1, 2))
    # fmt: off
    cut_cm = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
    ])
    # fmt: on
    assert np.array_equal(cut.apply_cut(cm), cut_cm)


def test_cut_is_null():
    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    assert not cut.is_null


def test_null_cut():
    cut = models.NullCut((2, 3))
    assert cut.indices == (2, 3)
    assert cut.is_null
    assert np.array_equal(cut.cut_matrix(4), np.zeros((4, 4)))


def test_null_cut_str():
    cut = models.NullCut((2, 3))
    assert str(cut) == "NullCut((2, 3))"


def test_null_cut_equality():
    cut = models.NullCut((2, 3))
    other = models.NullCut((2, 3))
    assert cut == other
    assert hash(cut) == hash(other)


def test_cuts_can_have_node_labels(node_labels):
    models.NullCut((0, 1), node_labels=node_labels)
    DirectedBipartition(Direction.EFFECT, (0,), (1,), node_labels=node_labels)

    k_partition = models.JointPartition(
        models.Part((0, 1), (0,)), models.Part((), (1,)), node_labels=node_labels
    )
    models.DirectedJointPartition(Direction.CAUSE, k_partition, node_labels=node_labels)


# Test ria


def test_ria_ordering_and_equality():
    assert ria(phi=1.0) < ria(phi=2.0)
    assert ria(phi=2.0) > ria(phi=1.0)
    assert ria(phi=1.0) == ria(phi=1.0)
    assert ria(phi=1.0) == ria(phi=(1.0 - EPSILON / 2))
    assert ria(phi=1.0) != ria(phi=(1.0 - EPSILON * 2))
    assert ria(direction=Direction.CAUSE) != ria(direction=Direction.EFFECT)


def test_null_ria():
    direction = Direction.CAUSE
    mechanism = (0,)
    purview = (1,)
    repertoire = "repertoire"
    null_ria = models._null_ria(direction, mechanism, purview, repertoire)
    assert null_ria.direction == direction
    assert null_ria.mechanism == mechanism
    assert null_ria.purview == purview
    assert null_ria.partition == JointPartition()
    assert null_ria.repertoire == "repertoire"
    assert null_ria.partitioned_repertoire is None
    assert null_ria.phi == 0


def test_ria_repr_str():
    print(repr(ria()))
    print(str(ria()))


# Test MaximallyIrreducibleCauseOrEffect


def test_mice_ordering():
    phi1 = mice()
    phi2 = mice(phi=(1.0 + EPSILON * 2), partition=())
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    different_direction = mice(direction="different")
    assert phi2 > different_direction
    assert different_direction < phi2
    assert phi2 >= different_direction
    assert different_direction <= phi2


def test_mice_equality():
    m = mice(phi=1.0)
    close_enough = mice(phi=(1.0 - EPSILON / 2))
    not_quite = mice(phi=(1.0 - EPSILON * 2))
    assert m == close_enough
    assert m != not_quite


def test_mice_repr_str():
    print(repr(mice()))
    print(str(mice()))


def test_relevant_connections(s, subsys_n1n2):
    m = mice(mechanism=(0,), purview=(1,), direction=Direction.CAUSE)
    # fmt: off
    answer = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    # fmt: on
    assert np.array_equal(m._relevant_connections(s), answer)

    m = mice(mechanism=(1,), purview=(1, 2), direction=Direction.EFFECT)
    # fmt: off
    answer = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
    ])
    # fmt: on
    assert np.array_equal(m._relevant_connections(subsys_n1n2), answer)


def test_damaged(s):
    # Build cut system from s
    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    cut_s = System(s.substrate, s.state, s.node_indices, cut=cut)

    # Cut splits mechanism:
    m1 = mice(mechanism=(0, 1), purview=(1, 2), direction=Direction.EFFECT)
    assert m1.damaged_by_cut(cut_s)
    assert not m1.damaged_by_cut(s)

    # Cut splits mechanism & purview (but not *only* mechanism)
    m2 = mice(mechanism=(0,), purview=(1, 2), direction=Direction.EFFECT)
    assert m2.damaged_by_cut(cut_s)
    assert not m2.damaged_by_cut(s)


# Test MIC and MIE


def test_mic_raises_wrong_direction():
    mic(direction=Direction.CAUSE, mechanism=(0,), purview=(1,))
    with pytest.raises(exceptions.WrongDirectionError):
        mic(direction=Direction.EFFECT, mechanism=(0,), purview=(1,))


def test_mie_raises_wrong_direction():
    mie(direction=Direction.EFFECT, mechanism=(0,), purview=(1,))
    with pytest.raises(exceptions.WrongDirectionError):
        mie(direction=Direction.CAUSE, mechanism=(0,), purview=(1,))


# NOTE: test_specified_states_and_indices was removed because it tested IIT 3.0
# specified_state functionality that is not implemented for IIT 4.0.


# Test Concept


# NOTE: test_concept_ordering was removed because it relied on Concept comparison
# that requires a system attribute, which was removed from Concept during the
# IIT 3.0 -> 4.0 migration.


def test_concept_equality(s):
    assert concept(system=s) == concept(system=s)


def test_concept_equality_phi(s):
    assert concept(phi=1.0, system=s) != concept(phi=0.0, system=s)


def test_concept_equality_cause_purview_nodes(s):
    assert concept(cause_purview=(1, 2), system=s) != concept(
        cause_purview=(1,), system=s
    )


def test_concept_equality_effect_purview_nodes(s):
    assert concept(effect_purview=(1, 2), system=s) != concept(
        effect_purview=(1,), system=s
    )


def test_concept_equality_repertoires(s):
    phi = 1.0
    mice1 = mice(
        phi=phi, repertoire=np.array([1, 2]), partitioned_repertoire=np.array([2, 3])
    )
    mice2 = mice(phi=phi, repertoire=np.array([0, 0]), partitioned_repertoire=None)
    concept = models.Concept(
        mechanism=(),
        cause=mice1,
        effect=mice2,
    )
    another = models.Concept(
        mechanism=(),
        cause=mice2,
        effect=mice1,
    )
    assert concept != another


# NOTE: test_concept_equality_substrate was removed because it relied on Concept
# comparison that requires system/mechanism_state attributes, which are not
# available on test concepts created with the helper function.


def test_concept_equality_one_system_is_subset_of_another(s, subsys_n1n2):
    assert concept(system=s) == concept(system=subsys_n1n2)


# NOTE: test_concept_repr_str was removed because it relied on Concept formatting
# that requires mechanism_state, which is not available on test concepts.


def test_concept_hashing(s):
    hash(concept(system=s))


def test_concept_hashing_one_system_is_subset_of_another(s, subsys_n1n2):
    c1 = concept(system=s)
    c2 = concept(system=subsys_n1n2)
    assert hash(c1) == hash(c2)
    assert len({c1, c2}) == 1


def test_concept_emd_eq(s, subsys_n1n2):
    c1 = concept(system=s)

    # Same repertoires, mechanism, phi
    c2 = concept(system=subsys_n1n2)
    assert c1.emd_eq(c2)

    # Everything equal except phi
    c3 = concept(phi=2.0, system=s)
    assert not c1.emd_eq(c3)

    # TODO: test other expectations...


# Test Distinctions


def test_ces_is_still_a_tuple(s):
    c = models.UnresolvedDistinctions([concept(system=s)])
    assert len(c) == 1


# NOTE: test_ces_repr_str was removed because it relied on Concept formatting
# that requires mechanism_state, which is not available on test concepts.


def test_ces_are_always_normalized(s):
    c1 = concept(mechanism=(0,), system=s)
    c2 = concept(mechanism=(1,), system=s)
    c3 = concept(mechanism=(0, 2), system=s)
    c4 = concept(mechanism=(0, 1, 2), system=s)
    assert (c1, c2, c3, c4) == models.UnresolvedDistinctions((c3, c4, c2, c1)).concepts


@pytest.mark.outdated
def test_ces_labeled_mechanisms(s):
    c = models.UnresolvedDistinctions([concept(system=s)])
    assert c.labeled_mechanisms == (["A", "B"],)


# NOTE: test_ces_ordering was removed because it relied on Distinctions
# comparison that requires a system attribute, which was removed from CES.


# Test SystemIrreducibilityAnalysis


def test_sia_ordering(s, s_noised, subsys_n0n2, subsys_n1n2):
    phi1 = sia(system=s)
    phi2 = sia(system=s, phi=1.0 + EPSILON * 2)
    assert phi1 < phi2
    assert phi2 > phi1
    assert phi1 <= phi2
    assert phi2 >= phi1

    different_system = sia(system=s_noised)
    with pytest.raises(TypeError):
        phi1 <= different_system  # noqa: B015
    with pytest.raises(TypeError):
        phi1 >= different_system  # noqa: B015


def test_sia_equality(s):
    bm = sia(system=s)
    close_enough = sia(system=s, phi=(1.0 - EPSILON / 2))
    not_quite = sia(system=s, phi=(1.0 - EPSILON * 2))
    assert bm == close_enough
    assert bm != not_quite


def test_sia_repr_str(s):
    bm = sia(system=s)
    print(repr(bm))
    print(str(bm))


# Test model __str__ and __reprs__


def test_indent():
    s = "line1\nline2"
    answer = "  line1\n  line2"
    assert models.fmt.indent(s) == answer


class ReadableReprClass:
    """Dummy class for make_repr tests"""

    some_attr = 3.14

    def __repr__(self):
        return models.fmt.make_repr(self, ["some_attr"])

    def __str__(self):
        return "A nice fat explicit string"


@config.override(repr_verbosity=0)
def test_make_reprs_uses___repr__():
    assert repr(ReadableReprClass()) == "ReadableReprClass(some_attr=3.14)"


@config.override(repr_verbosity=2)
def test_make_reprs_calls_out_to_string():
    assert repr(ReadableReprClass()) == "A nice fat explicit string"


# Test partitions


@pytest.fixture
def node_labels():
    return NodeLabels("ABCDE", tuple(range(5)))


@pytest.fixture
def bipartition(node_labels):
    return models.JointBipartition(
        models.Part((0,), (0, 4)), models.Part((), (1,)), node_labels=node_labels
    )


def test_bipartition_properties(bipartition):
    assert set(bipartition.mechanism) == {0}
    assert set(bipartition.purview) == {0, 1, 4}


def test_bipartition_str(bipartition):
    assert str(bipartition) == (" A     ∅ \n─── ✕ ───\nA,E    B ")


@pytest.fixture
def tripartition(node_labels):
    return models.JointTripartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)),
        models.Part((2,), (2,)),
        node_labels=node_labels,
    )


def test_tripartion_properties(tripartition):
    assert set(tripartition.mechanism) == {0, 2}
    assert set(tripartition.purview) == {0, 1, 2, 4}


def test_tripartion_str(tripartition):
    assert str(tripartition) == (" A     ∅     C \n─── ✕ ─── ✕ ───\nA,E    B     C ")


def k_partition(node_labels=None):
    return models.JointPartition(
        models.Part((0,), (0, 4)),
        models.Part((), (1,)),
        models.Part((6,), (5,)),
        models.Part((2,), (2,)),
        node_labels=node_labels,
    )


def test_partition_normalize():
    assert k_partition().normalize() == models.JointPartition(
        models.Part((), (1,)),
        models.Part((0,), (0, 4)),
        models.Part((2,), (2,)),
        models.Part((6,), (5,)),
    )


def test_partition_normalize_preserves_labels(node_labels):
    k = k_partition(node_labels=node_labels)
    assert k.normalize().node_labels == k.node_labels


def test_partition_eq_hash():
    assert k_partition() == k_partition()
    assert hash(k_partition()) == hash(k_partition())


class TestRepertoireIrreducibilityAnalysisDistanceResult:
    """Test RepertoireIrreducibilityAnalysis integration with DistanceResult."""

    def test_ria_preserves_distance_result_phi(self):
        """Test that RIA preserves DistanceResult type in phi attribute."""
        from pyphi.metrics.distribution import DistanceResult

        # Create a DistanceResult with auxiliary data
        distance_result = DistanceResult(0.42, method="EMD", direction="CAUSE", state=1)

        # Create RIA with DistanceResult phi
        test_ria = ria(
            phi=distance_result,
            direction=Direction.CAUSE,
            mechanism=(0,),
            purview=(0, 1),
        )

        # Verify that phi retains DistanceResult type and auxiliary data
        assert isinstance(test_ria.phi, DistanceResult)
        assert float(test_ria.phi) == 0.42
        assert test_ria.phi.method == "EMD"
        assert test_ria.phi.direction == "CAUSE"
        assert test_ria.phi.state == 1

    def test_ria_converts_float_to_pyphi_float(self):
        """Test that RIA converts regular float to PyPhiFloat."""
        from pyphi.data_structures import PyPhiFloat

        # Create RIA with regular float phi
        test_ria = ria(
            phi=0.25, direction=Direction.EFFECT, mechanism=(1,), purview=(0, 1)
        )

        # Verify that phi is converted to PyPhiFloat
        assert isinstance(test_ria.phi, PyPhiFloat)
        assert float(test_ria.phi) == 0.25

    def test_multiple_rias_with_distance_results_min_comparison(self):
        """Test min() comparison across multiple RIAs with DistanceResults."""
        from pyphi.metrics.distribution import DistanceResult

        # Create multiple RIAs with DistanceResult phi values
        rias = [
            ria(
                phi=DistanceResult(
                    0.6, method="EMD", direction="CAUSE", partition="A|B"
                ),
                direction=Direction.CAUSE,
            ),
            ria(
                phi=DistanceResult(
                    0.3, method="L1", direction="EFFECT", partition="AB|"
                ),
                direction=Direction.EFFECT,
            ),
            ria(
                phi=DistanceResult(
                    0.8, method="KLD", direction="CAUSE", partition="X|Y"
                ),
                direction=Direction.CAUSE,
            ),
        ]

        # Simulate SystemIrreducibilityAnalysis scenario:
        # phi = min(integration[direction].phi for direction in directions)
        phi_values = [r.phi for r in rias]
        min_phi = min(phi_values)

        # Verify that min preserves DistanceResult type and auxiliary data
        assert isinstance(min_phi, DistanceResult)
        assert float(min_phi) == 0.3
        assert min_phi.method == "L1"
        assert min_phi.direction == "EFFECT"
        assert min_phi.partition == "AB|"

    def test_ria_comparison_preserves_types(self):
        """Test that RIA comparison operations preserve DistanceResult types."""
        from pyphi.metrics.distribution import DistanceResult

        # Create RIAs with different phi values
        ria1 = ria(phi=DistanceResult(0.7, method="EMD", direction="CAUSE"))
        ria2 = ria(phi=DistanceResult(0.3, method="L1", direction="EFFECT"))

        # Test comparison operations
        assert ria1 > ria2
        assert ria2 < ria1

        # The actual phi objects should maintain their types
        assert isinstance(ria1.phi, DistanceResult)
        assert isinstance(ria2.phi, DistanceResult)
        assert ria1.phi.method == "EMD"
        assert ria2.phi.method == "L1"
