import numpy as np
import pytest

from pyphi import Direction
from pyphi import config
from pyphi.formalism import iit3
from pyphi.models import KCut
from pyphi.models import KPartition
from pyphi.models import Part

from .test_models import sia


@pytest.fixture()
def kcut_cause():
    partition = KPartition(Part((0, 2), (0,)), Part((), (2,)), Part((3,), (3,)))
    return KCut(Direction.CAUSE, partition)


@pytest.fixture()
def kcut_effect():
    partition = KPartition(Part((0, 2), (0,)), Part((), (2,)), Part((3,), (3,)))
    return KCut(Direction.EFFECT, partition)


def test_cut_indices(kcut_cause, kcut_effect):
    assert kcut_cause.indices == (0, 2, 3)
    assert kcut_effect.indices == (0, 2, 3)


def test_apply_cut(kcut_cause, kcut_effect):
    cm = np.ones((4, 4))
    # fmt: off
    cut_cm = np.array([
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
    ])
    # fmt: on
    assert np.array_equal(kcut_cause.apply_cut(cm), cut_cm)

    cm = np.ones((4, 4))
    # fmt: off
    cut_cm = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
    ])
    # fmt: on
    assert np.array_equal(kcut_effect.apply_cut(cm), cut_cm)


def test_cut_matrix(kcut_cause, kcut_effect):
    # fmt: off
    answer = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 0],
    ])
    # fmt: on
    assert np.array_equal(kcut_cause.cut_matrix(4), answer)

    # fmt: off
    answer = np.array([
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0]
    ])
    # fmt: on
    assert np.array_equal(kcut_effect.cut_matrix(4), answer)


def test_splits_mechanism(kcut_cause):
    assert kcut_cause.splits_mechanism((0, 3))
    assert kcut_cause.splits_mechanism((2, 3))
    assert not kcut_cause.splits_mechanism((0,))
    assert not kcut_cause.splits_mechanism((3,))


def test_all_cut_mechanisms(kcut_cause):
    assert list(kcut_cause.all_cut_mechanisms()) == [
        (2,),
        (0, 2),
        (0, 3),
        (2, 3),
        (0, 2, 3),
    ]


@config.override(partition_type="TRI")
def test_concept_style_cuts():
    assert list(iit3.concept_cuts(Direction.CAUSE, (0,))) == [
        KCut(Direction.CAUSE, KPartition(Part((), ()), Part((), (0,)), Part((0,), ())))
    ]
    assert list(iit3.concept_cuts(Direction.EFFECT, (0,))) == [
        KCut(Direction.EFFECT, KPartition(Part((), ()), Part((), (0,)), Part((0,), ())))
    ]


def test_kcut_equality(kcut_cause, kcut_effect):
    other = KCut(
        Direction.CAUSE,
        KPartition(Part((0, 2), (0,)), Part((), (2,)), Part((3,), (3,))),
    )
    assert kcut_cause == other
    assert hash(kcut_cause) == hash(other)
    assert hash(kcut_cause) != hash(kcut_cause.partition)

    assert kcut_cause != kcut_effect
    assert hash(kcut_cause) != hash(kcut_effect)


def test_system_accessors(s):
    cut_cause = KCut(Direction.CAUSE, KPartition(Part((0, 2), (0, 1)), Part((1,), (2,))))
    cs_cause = iit3.ConceptStyleSystem(s, Direction.CAUSE, cut_cause)
    assert cs_cause.cause_system.cut == cut_cause
    assert not cs_cause.effect_system.is_cut

    cut_effect = KCut(
        Direction.EFFECT, KPartition(Part((0, 2), (0, 1)), Part((1,), (2,)))
    )
    cs_effect = iit3.ConceptStyleSystem(s, Direction.EFFECT, cut_effect)
    assert not cs_effect.cause_system.is_cut
    assert cs_effect.effect_system.cut == cut_effect


def sia_cs(phi=1.0, system=None):
    return iit3.SystemIrreducibilityAnalysisConceptStyle(
        sia_cause=sia(phi=phi, system=system),
        sia_effect=sia(phi=phi, system=system),
    )


@pytest.mark.slow
def test_sia_concept_style_ordering(s, subsys_n0n2, s_noised):
    assert sia_cs(system=s) == sia_cs(system=s)
    assert sia_cs(phi=1, system=s) < sia_cs(phi=2, system=s)

    assert sia_cs(system=s) >= sia_cs(system=subsys_n0n2)

    with pytest.raises(TypeError):
        sia_cs(system=s) < sia_cs(system=s_noised)  # noqa: B015


# NOTE: test_sia_concept_style, test_unpickle, and test_concept_style_phi were removed
# because they had outdated expected values and were incompatible with IIT 4.0.
