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


def close(a, b):
    return np.all(abs(a - b) < EPSILON)


def test_expand_cause_repertoire(micro_s_all_off):
    sub = micro_s_all_off
    sia = compute.sia(sub)
    cause = sia.ces[0].cause_repertoire

    assert close(sub.expand_cause_repertoire(cause, CD), cause)
    assert close(sub.expand_cause_repertoire(cause, BCD).flatten(order='F'),
                 np.array([1 / 6 if i < 6 else 0 for i in range(8)]))
    assert close(sub.expand_cause_repertoire(cause, ABCD).flatten(order='F'),
                 np.array([1 / 12 if i < 12 else 0 for i in range(16)]))
    assert close(sub.expand_cause_repertoire(cause, ABCD),
                 sub.expand_cause_repertoire(cause))


def test_expand_effect_repertoire(micro_s_all_off):
    sub = micro_s_all_off
    sia = compute.sia(sub)
    effect = sia.ces[0].effect_repertoire

    assert close(sub.expand_effect_repertoire(effect, CD), effect)
    assert close(sub.expand_effect_repertoire(effect, BCD).flatten(order='F'),
                 np.array([.25725, .23275, .11025, .09975,
                           .11025, .09975, .04725, .04275]))
    assert close(sub.expand_effect_repertoire(effect, ABCD).flatten(order='F'),
                 np.array([.13505625, .12219375, .12219375, .11055625,
                           .05788125, .05236875, .05236875, .04738125,
                           .05788125, .05236875, .05236875, .04738125,
                           .02480625, .02244375, .02244375, .02030625]))
    assert close(sub.expand_effect_repertoire(effect, ABCD),
                 sub.expand_effect_repertoire(effect))


def test_expand_repertoire_purview_must_be_subset_of_new_purview(s):
    mechanism = (0, 1)
    purview = (0, 1)
    new_purview = (1,)
    cause_repertoire = s.cause_repertoire(mechanism, purview)
    with pytest.raises(ValueError):
        s.expand_repertoire(Direction.CAUSE, cause_repertoire, new_purview)
