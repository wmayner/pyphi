import pytest
import numpy as np

from pyphi.compute import concept_cuts
from pyphi.models import KCut, KPartition, Part


@pytest.fixture()
def kcut():
    return KCut(KPartition(Part((0, 2), (0,)), Part((), (2,)), Part((3,), (3,))))


def test_cut_indices(kcut):
    assert kcut.indices == (0, 2, 3)


def test_apply_cut(kcut):
    cm = np.ones((4, 4))
    cut_cm = np.array([
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]])
    assert np.array_equal(kcut.apply_cut(cm), cut_cm)


def test_cut_matrix(kcut):
    assert np.array_equal(kcut.cut_matrix(4), np.array([
        [0, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0]]))


def test_splits_mechanism(kcut):
    assert kcut.splits_mechanism((0, 3))
    assert kcut.splits_mechanism((2, 3))
    assert not kcut.splits_mechanism((0,))
    assert not kcut.splits_mechanism((3,))


def test_all_cut_mechanisms(kcut):
    assert kcut.all_cut_mechanisms() == (
        (2,), (0, 2), (0, 3), (2, 3), (0, 2, 3))
