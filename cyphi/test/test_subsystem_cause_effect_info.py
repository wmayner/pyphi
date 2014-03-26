#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cyphi.utils import emd
from .example_networks import WithExampleNetworks


class TestCauseEffectInformation(WithExampleNetworks):

    def test_cause_info(self):
        mechanism = [self.m0, self.m1]
        purview = [self.m0, self.m2]
        answer = emd(
            self.m_subsys_all.cause_repertoire(mechanism, purview),
            self.m_subsys_all.unconstrained_cause_repertoire(purview))
        assert self.m_subsys_all.cause_info(mechanism, purview) == answer

    def test_effect_info(self):
        mechanism = [self.m0, self.m1]
        purview = [self.m0, self.m2]
        answer = emd(
            self.m_subsys_all.effect_repertoire(mechanism, purview),
            self.m_subsys_all.unconstrained_effect_repertoire(purview))
        assert self.m_subsys_all.effect_info(mechanism, purview) == answer

    def test_cause_effect_info(self):
        mechanism = [self.m0, self.m1]
        purview = [self.m0, self.m2]
        answer = min(self.m_subsys_all.cause_info(mechanism, purview),
                     self.m_subsys_all.effect_info(mechanism, purview))
        assert (self.m_subsys_all.cause_effect_info(mechanism, purview) ==
                answer)
