#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_metrics_ces.py

import numpy as np
import pytest

from pyphi import compute, config, metrics
from pyphi.compute.distance import ces_distance


def test_emd_ground_distance_must_be_symmetric():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    with config.override(REPERTOIRE_DISTANCE="KLD"):
        with pytest.raises(ValueError):
            metrics.ces.emd_ground_distance(a, b)


def test_ces_distances(s):
    with config.override(REPERTOIRE_DISTANCE="EMD"):
        sia = compute.subsystem.sia(s)

    with config.override(CES_DISTANCE="EMD"):
        assert compute.distance.ces_distance(sia.ces, sia.partitioned_ces) == 2.3125

    with config.override(CES_DISTANCE="SUM_SMALL_PHI"):
        assert compute.distance.ces_distance(sia.ces, sia.partitioned_ces) == 1.083333


def test_sia_uses_ces_distances(s):
    with config.override(REPERTOIRE_DISTANCE="EMD", CES_DISTANCE="EMD"):
        sia = compute.subsystem.sia(s)
        assert sia.phi == 2.3125

    with config.override(REPERTOIRE_DISTANCE="EMD", CES_DISTANCE="SUM_SMALL_PHI"):
        sia = compute.subsystem.sia(s)
        assert sia.phi == 1.083333