#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .example_networks import WithExampleNetworks
from cyphi.subsystem import Subsystem, a_cut


# TODO test against other matlab examples
class TestSubsystem(WithExampleNetworks):

    def test_empty_init(self):
        # Empty mechanism
        s = Subsystem([],
                      self.m_network.current_state,
                      self.m_network.past_state,
                      self.m_network)
        assert s.nodes == ()

    def test_eq(self):
        a = self.m_subsys_n0n2
        b = self.m_subsys_n0n2
        assert a == b

    def test_hash(self):
        print(hash(self.m_subsys_all))

    def test_cut_bad_input(self):
        s = self.m_subsys_all
        with self.assertRaises(ValueError):
            s.cut((), ())
        with self.assertRaises(ValueError):
            s.cut(self.m0, self.m1)
        with self.assertRaises(ValueError):
            s.cut(self.m0, (self.m1, self.m1))

    def test_cut_single_node(self):
        s = self.m_subsys_all
        s.cut(self.m0, (self.m1, self.m2))
        assert s._cut == a_cut((self.m0,), (self.m1, self.m2))

    def test_cut(self):
        s = self.m_subsys_all
        s.cut((self.m0,), (self.m1, self.m2))
        assert s._cut == a_cut((self.m0,), (self.m1, self.m2))
