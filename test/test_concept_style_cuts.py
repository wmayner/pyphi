import pytest
import numpy as np

from pyphi import config, compute
from pyphi.constants import Direction
from pyphi.compute import concept_cuts, ConceptStyleSystem, BigMipConceptStyle
from pyphi.models import KCut, KPartition, Part
from test_models import bigmip


@pytest.fixture()
def kcut():
    return KCut(KPartition(Part((0, 2), (0,)), Part((), (2,)), Part((3,), (3,))))


def test_cut_indices(kcut):
    assert kcut.indices == (0, 2, 3)


def test_apply_cut(kcut):
    cm = np.ones((4, 4))
    cut_cm = np.array([
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1]])
    assert np.array_equal(kcut.apply_cut(cm), cut_cm)


def test_cut_matrix(kcut):
    assert np.array_equal(kcut.cut_matrix(4), np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [1, 0, 1, 0]]))


def test_splits_mechanism(kcut):
    assert kcut.splits_mechanism((0, 3))
    assert kcut.splits_mechanism((2, 3))
    assert not kcut.splits_mechanism((0,))
    assert not kcut.splits_mechanism((3,))


def test_all_cut_mechanisms(kcut):
    assert kcut.all_cut_mechanisms() == (
        (2,), (0, 2), (0, 3), (2, 3), (0, 2, 3))


def test_system_accessors(s):
    cut = KCut(KPartition(Part((0, 2), (0, 1)), Part((1,), (2,))))

    cs_past = ConceptStyleSystem(s, Direction.PAST, cut)
    assert cs_past.cause_system.cut == cut
    assert cs_past.effect_system.cut == s.null_cut

    cs_future = ConceptStyleSystem(s, Direction.FUTURE, cut)
    assert cs_future.cause_system.cut == s.null_cut
    assert cs_future.effect_system.cut == cut


def big_mip_cs(phi=1.0, subsystem=None):
    return BigMipConceptStyle(
        subsystem=subsystem,
        mip_past=bigmip(subsystem=subsystem, phi=phi),
        mip_future=bigmip(subsystem=subsystem, phi=phi))


def test_big_mip_concept_style_ordering(s, subsys_n0n2, s_noised):
    assert big_mip_cs(subsystem=s) == big_mip_cs(subsystem=s)
    assert big_mip_cs(phi=1, subsystem=s) < big_mip_cs(phi=2, subsystem=s)
    assert big_mip_cs(subsystem=s) >= big_mip_cs(subsystem=subsys_n0n2)

    with pytest.raises(TypeError):
        big_mip_cs(subsystem=s) < big_mip_cs(subsystem=s_noised)


@config.override(SYSTEM_CUTS='CONCEPT_STYLE', PARALLEL_CUT_EVALUATION=True)
def test_unpickling_in_parallel_computations(s, flushcache, restore_fs_cache):
    assert compute.big_phi(s) == 0.6875
