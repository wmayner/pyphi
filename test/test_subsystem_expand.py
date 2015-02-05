#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyphi.compute import big_mip
from pyphi import Subsystem
import numpy as np
import example_networks
from pyphi.constants import EPSILON

micro = example_networks.micro()
micro.current_state = (0, 0, 0, 0)
micro.past_state = (0, 0, 0, 0)
micro_subsystem = Subsystem(range(micro.size), micro)
mip = big_mip(micro_subsystem)

CD = micro_subsystem.nodes[2:4]
BCD = micro_subsystem.nodes[1:4]
ABCD = micro_subsystem.nodes[0:4]

A = mip.unpartitioned_constellation[0]

cause = A.cause.mip.unpartitioned_repertoire
effect = A.effect.mip.unpartitioned_repertoire


def test_expand_cause_repertoire():
    assert np.all(abs(A.expand_cause_repertoire(CD) - cause) < EPSILON)
    assert np.all(abs(
        A.expand_cause_repertoire(BCD).flatten(order='F') -
        np.array([1/6 if i < 6 else 0 for i in range(8)])) < EPSILON)
    assert np.all(abs(
        A.expand_cause_repertoire(ABCD).flatten(order='F') -
        np.array([1/12 if i < 12 else 0 for i in range(16)])) < EPSILON)
    assert np.all(abs(A.expand_cause_repertoire(ABCD) -
                      A.expand_cause_repertoire()) < EPSILON)


def test_expand_effect_repertoire():
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
