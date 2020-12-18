#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_compute_distance.py

from unittest.mock import patch

import numpy as np
import pytest

from pyphi import compute, config, metrics, models


def test_system_repertoire_distance_must_be_symmetric():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    with config.override(REPERTOIRE_DISTANCE="KLD"):
        with pytest.raises(ValueError):
            metrics.ces.system_repertoire_distance(a, b)


@patch("pyphi.metrics.ces._ces_distance_simple")
@patch("pyphi.metrics.ces._ces_distance_emd")
def test_ces_distance_uses_simple_vs_emd(mock_emd_distance, mock_simple_distance, s):
    """Quick check that we use the correct CES distance function.

    If the two CESs differ only in that some concepts have
    moved to the null concept and all other concepts are the same then
    we use the simple CES distance. Otherwise, use the EMD.
    """
    mock_emd_distance.return_value = float()
    mock_simple_distance.return_value = float()

    make_mice = lambda: models.MaximallyIrreducibleCauseOrEffect(
        models.RepertoireIrreducibilityAnalysis(
            phi=0,
            direction=None,
            mechanism=None,
            purview=None,
            partition=None,
            repertoire=None,
            partitioned_repertoire=None,
        )
    )

    lone_concept = models.Concept(
        cause=make_mice(), effect=make_mice(), mechanism=(0, 1), subsystem=s
    )
    # lone concept -> null concept
    compute.distance.ces_distance((lone_concept,), ())
    assert mock_emd_distance.called is False
    assert mock_simple_distance.called is True
    mock_simple_distance.reset_mock()

    other_concept = models.Concept(
        cause=make_mice(), effect=make_mice(), mechanism=(0, 1, 2), subsystem=s
    )
    # different concepts in CES
    compute.distance.ces_distance((lone_concept,), (other_concept,))
    assert mock_emd_distance.called is True
    assert mock_simple_distance.called is False


def test_ces_distance_switches_to_small_phi_difference(s):
    sia = compute.subsystem.sia(s)
    ce_structures = (sia.ces, sia.partitioned_ces)

    with config.override(USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE=False):
        assert compute.distance.ces_distance(*ce_structures) == 2.3125

    with config.override(USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE=True):
        assert compute.distance.ces_distance(*ce_structures) == 1.083333
