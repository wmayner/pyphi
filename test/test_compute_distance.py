#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_compute_distance.py

import numpy as np
import pytest

from pyphi import config
from pyphi.compute import distance


def test_system_repertoire_distance_must_be_symmetric():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    with config.override(REPERTOIRE_DISTANCE="KLD"):
        with pytest.raises(ValueError):
            distance.system_repertoire_distance(a, b)