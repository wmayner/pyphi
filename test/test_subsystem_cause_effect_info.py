#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_subsystem_cause_effect_info.py

from pyphi.distance import hamming_emd


def test_cause_info(s):
    mechanism = (0, 1)
    purview = (0, 2)
    answer = hamming_emd(
        s.cause_repertoire(mechanism, purview),
        s.unconstrained_cause_repertoire(purview))
    assert s.cause_info(mechanism, purview) == answer


def test_effect_info(s):
    mechanism = (0, 1)
    purview = (0, 2)
    answer = hamming_emd(
        s.effect_repertoire(mechanism, purview),
        s.unconstrained_effect_repertoire(purview))
    assert s.effect_info(mechanism, purview) == answer


def test_cause_effect_info(s):
    mechanism = (0, 1)
    purview = (0, 2)
    answer = min(s.cause_info(mechanism, purview),
                 s.effect_info(mechanism, purview))
    assert (s.cause_effect_info(mechanism, purview) == answer)
