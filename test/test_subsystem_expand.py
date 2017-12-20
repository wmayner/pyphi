#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_subsystem_expand.py

import numpy as np
import pytest

from pyphi import Direction, compute
from pyphi.constants import EPSILON

CD = (2, 3)
BCD = (1, 2, 3)
ABCD = (0, 1, 2, 3)


def test_expand_cause_repertoire(micro_s_all_off):
    sia = compute.sia(micro_s_all_off)
    A = sia.ces[0]
    cause = A.cause_repertoire

    assert np.all(abs(A.expand_cause_repertoire(CD) - cause) < EPSILON)
    assert np.all(abs(
        A.expand_cause_repertoire(BCD).flatten(order='F') -
        np.array([1 / 6 if i < 6 else 0 for i in range(8)])) < EPSILON)
    assert np.all(abs(
        A.expand_cause_repertoire(ABCD).flatten(order='F') -
        np.array([1 / 12 if i < 12 else 0 for i in range(16)])) < EPSILON)
    assert np.all(abs(A.expand_cause_repertoire(ABCD) -
                      A.expand_cause_repertoire()) < EPSILON)


def test_expand_effect_repertoire(micro_s_all_off):
    sia = compute.sia(micro_s_all_off)
    A = sia.ces[0]
    effect = A.effect_repertoire

    assert np.all(abs(A.expand_effect_repertoire(CD) - effect) < EPSILON)
    assert np.all(abs(A.expand_effect_repertoire(BCD).flatten(order='F') -
                      np.array([.25725, .23275, .11025, .09975,
                                .11025, .09975, .04725, .04275])) < EPSILON)
    assert np.all(abs(
        A.expand_effect_repertoire(ABCD).flatten(order='F') -
        np.array([.13505625, .12219375, .12219375, .11055625,
                  .05788125, .05236875, .05236875, .04738125,
                  .05788125, .05236875, .05236875, .04738125,
                  .02480625, .02244375, .02244375, .02030625])) < EPSILON)
    assert np.all(abs(A.expand_effect_repertoire(ABCD) -
                      A.expand_effect_repertoire()) < EPSILON)


def test_expand_repertoire_purview_must_be_subset_of_new_purview(s):
    mechanism = (0, 1)
    purview = (0, 1)
    new_purview = (1,)
    cause_repertoire = s.cause_repertoire(mechanism, purview)
    with pytest.raises(ValueError):
        s.expand_repertoire(Direction.CAUSE, cause_repertoire, new_purview)
