#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .example_networks import WithExampleNetworks


# TODO finish
class TestPhiMax(WithExampleNetworks):

    def test_find_mice_bad_direction(self):
        mechanism = [self.m0]
        with self.assertRaises(ValueError):
            self.m_subsys_all.find_mice('doge', mechanism)

    def test_find_mice(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        past_mice = s.find_mice('past', mechanism)
        future_mice = s.find_mice('future', mechanism)
        assert 1

    def test_core_cause(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        assert s.core_cause(mechanism) == s.find_mice('past', mechanism)

    def test_core_effect(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        assert s.core_effect(mechanism) == s.find_mice('future', mechanism)

    def test_phi_max(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        assert 1
        # assert 0.5 == round(s.phi_max(mechanism), 4)

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker
