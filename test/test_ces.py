#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_ces.py

import pytest
from unittest.mock import patch

from pyphi import compute, config, models


@patch('pyphi.compute.distance._ces_distance_simple')
@patch('pyphi.compute.distance._ces_distance_emd')
def test_ces_distance_uses_simple_vs_emd(mock_emd_distance,
                                         mock_simple_distance, s):
    """Quick check that we use the correct CES distance function.

    If the two CESs differ only in that some concepts have
    moved to the null concept and all other concepts are the same then
    we use the simple CES distance. Otherwise, use the EMD.
    """
    mock_emd_distance.return_value = float()
    mock_simple_distance.return_value = float()

    make_mice = lambda: models.MaximallyIrreducibleCauseOrEffect(
        models.RepertoireIrreducibilityAnalysis(
            phi=0, direction=None, mechanism=None, purview=None,
            partition=None, repertoire=None, partitioned_repertoire=None))

    lone_concept = models.Concept(cause=make_mice(), effect=make_mice(),
                                  mechanism=(0, 1), subsystem=s)
    # lone concept -> null concept
    compute.ces_distance((lone_concept,), ())
    assert mock_emd_distance.called is False
    assert mock_simple_distance.called is True
    mock_simple_distance.reset_mock()

    other_concept = models.Concept(cause=make_mice(), effect=make_mice(),
                                   mechanism=(0, 1, 2), subsystem=s)
    # different concepts in CES
    compute.ces_distance((lone_concept,), (other_concept,))
    assert mock_emd_distance.called is True
    assert mock_simple_distance.called is False


def test_ces_distance_switches_to_small_phi_difference(s):
    sia = compute.sia(s)
    ce_structures = (sia.ces, sia.partitioned_ces)

    with config.override(USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE=False):
        assert compute.ces_distance(*ce_structures) == 2.3125

    with config.override(USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE=True):
        assert compute.ces_distance(*ce_structures) == 1.083333


def test_parallel_and_sequential_ces_are_equal(s, micro_s, macro_s):
    with config.override(PARALLEL_CONCEPT_EVALUATION=False):
        c = compute.ces(s)
        c_micro = compute.ces(micro_s)
        c_macro = compute.ces(macro_s)

    with config.override(PARALLEL_CONCEPT_EVALUATION=True):
        assert set(c) == set(compute.ces(s))
        assert set(c_micro) == set(compute.ces(micro_s))
        assert set(c_macro) == set(compute.ces(macro_s))


@pytest.mark.parametrize('parallel', [False, True])
def test_ces_concepts_share_the_same_subsystem(parallel, s):
    with config.override(PARALLEL_CONCEPT_EVALUATION=parallel):
        ces = compute.ces(s)
        for concept in ces:
            assert concept.subsystem is ces.subsystem
