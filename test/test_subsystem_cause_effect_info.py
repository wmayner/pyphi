#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cyphi.utils import hamming_emd


def test_cause_info(m):
    mechanism = [m.nodes[0], m.nodes[1]]
    purview = [m.nodes[0], m.nodes[2]]
    answer = hamming_emd(
        m.subsys_all.cause_repertoire(mechanism, purview),
        m.subsys_all.unconstrained_cause_repertoire(purview))
    assert m.subsys_all.cause_info(mechanism, purview) == answer


def test_effect_info(m):
    mechanism = [m.nodes[0], m.nodes[1]]
    purview = [m.nodes[0], m.nodes[2]]
    answer = hamming_emd(
        m.subsys_all.effect_repertoire(mechanism, purview),
        m.subsys_all.unconstrained_effect_repertoire(purview))
    assert m.subsys_all.effect_info(mechanism, purview) == answer


def test_cause_effect_info(m):
    mechanism = [m.nodes[0], m.nodes[1]]
    purview = [m.nodes[0], m.nodes[2]]
    answer = min(m.subsys_all.cause_info(mechanism, purview),
                 m.subsys_all.effect_info(mechanism, purview))
    assert (m.subsys_all.cause_effect_info(mechanism, purview) == answer)
