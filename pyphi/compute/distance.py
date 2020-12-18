#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/distance.py

"""
Functions for computing distances between various PyPhi objects.
"""

import numpy as np
from pyemd import emd

from .. import config, metrics


def repertoire_distance(r1, r2, direction):
    """Compute the distance between two repertoires for the given direction.

    Args:
        r1 (np.ndarray): The first repertoire.
        r2 (np.ndarray): The second repertoire.
        direction (Direction): |CAUSE| or |EFFECT|.

    Returns:
        float: The distance between ``r1`` and ``r2``, rounded to |PRECISION|.
    """
    func = metrics.distribution.measures[config.REPERTOIRE_DISTANCE]
    try:
        distance = func(r1, r2, direction)
    except TypeError:
        distance = func(r1, r2)
    return round(distance, config.PRECISION)


def ces_distance(C1, C2, measure=None):
    """Return the distance between two cause-effect structures.

    Args:
        C1 (CauseEffectStructure): The first |CauseEffectStructure|.
        C2 (CauseEffectStructure): The second |CauseEffectStructure|.

    Returns:
        float: The distance between the two cause-effect structures.
    """
    measure = config.CES_DISTANCE if measure is None else measure
    dist = metrics.ces.measures[measure](C1, C2)
    return round(dist, config.PRECISION)