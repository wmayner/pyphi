#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cyphi.utils import tuple_eq
from .example_networks import WithExampleNetworks
from cyphi.subsystem import a_mip, a_part


# TODO test against other matlab examples
class TestSmallPhi(WithExampleNetworks):

    # MIP tests {{{
    # =============

        # Validation {{{
        # ~~~~~~~~~~~~~~

    def test_find_mip_bad_direction(self):
        mechanism = [self.m0]
        purview = [self.m0]
        with self.assertRaises(ValueError):
            self.m_subsys_all.find_mip('doge', mechanism, purview)

        # }}}

        # Past {{{
        # ~~~~~~~~

    def test_find_mip_past_reducible(self):
        mechanism = [self.m0]
        purview = [self.m0]
        mip = self.m_subsys_all.find_mip('past', mechanism, purview)
        assert mip is None

    def test_find_mip_past_irreducible_1(self):
        s = self.m_subsys_all
        s.cut(self.m0, (self.m1, self.m2))

        mechanism = [self.m1]
        purview = [self.m2]
        mip = s.find_mip('past', mechanism, purview)

        part0 = a_part(mechanism=(),
                       purview=(self.m2,))
        part1 = a_part(mechanism=(self.m1,),
                       purview=())
        partitioned_repertoire = np.array([0.5, 0.5]).reshape(1, 1, 2)
        phi = 0.5
        assert mip_eq(mip, a_mip(partition=(part0, part1),
                                 repertoire=partitioned_repertoire,
                                 difference=phi))

        # }}}

        # Future {{{
        # ~~~~~~~~~~

    def test_find_mip_future_irreducible_1(self):
        s = self.m_subsys_all
        s.cut((self.m1, self.m2), self.m0)

        mechanism = [self.m2]
        purview = [self.m1]
        mip = s.find_mip('future', mechanism, purview)

        part0 = a_part(mechanism=(),
                       purview=(self.m1,))
        part1 = a_part(mechanism=(self.m2,),
                       purview=())
        partitioned_repertoire = np.array([0.5, 0.5]).reshape(1, 2, 1)
        phi = 0.5
        assert mip_eq(mip, a_mip(partition=(part0, part1),
                                 repertoire=partitioned_repertoire,
                                 difference=phi))

    def test_find_mip_future_irreducible_2(self):
        s = self.m_subsys_all
        s.cut((self.m0, self.m2), self.m1)
        print(s._cut)

        mechanism = [self.m2]
        purview = [self.m0]
        mip = s.find_mip('future', mechanism, purview)

        part0 = a_part(mechanism=(),
                       purview=(self.m0,))
        part1 = a_part(mechanism=(self.m2,),
                       purview=())
        partitioned_repertoire = np.array([0.25, 0.75]).reshape(2, 1, 1)
        phi = 0.25
        assert mip_eq(mip, a_mip(partition=(part0, part1),
                                 repertoire=partitioned_repertoire,
                                 difference=phi))

    def test_find_mip_future_irreducible_3(self):
        s = self.m_subsys_all
        s.cut((self.m0, self.m2), self.m1)

        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m2]
        mip = s.find_mip('future', mechanism, purview)

        part0 = a_part(mechanism=(self.m0,),
                       purview=(self.m2,))
        part1 = a_part(mechanism=(self.m1, self.m2),
                       purview=(self.m0,))
        partitioned_repertoire = \
            np.array([0.5, 0.5, 0.0, 0.0]).reshape(2, 1, 2)
        phi = 0.5
        assert mip_eq(mip, a_mip(partition=(part0, part1),
                                 repertoire=partitioned_repertoire,
                                 difference=phi))

    def test_find_mip_future_irreducible_4(self):
        s = self.m_subsys_all
        s.cut((self.m0, self.m1), self.m2)

        mechanism = [self.m1]
        purview = [self.m0]
        mip = s.find_mip('future', mechanism, purview)

        part0 = a_part(mechanism=(),
                       purview=(self.m0,))
        part1 = a_part(mechanism=(self.m1,),
                       purview=())
        partitioned_repertoire = np.array([0.25, 0.75]).reshape(2, 1, 1)
        phi = 0.25
        assert mip_eq(mip, a_mip(partition=(part0, part1),
                                 repertoire=partitioned_repertoire,
                                 difference=phi))

        # }}}

    # }}}

    # Wrapper method tests {{{
    # ========================

    def test_mip_past(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        mip_past = s.find_mip('past', mechanism, purview)
        assert tuple_eq(mip_past, s.mip_past(mechanism, purview))

    def test_mip_future(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        mip_future = s.find_mip('future', mechanism, purview)
        assert tuple_eq(mip_future, s.mip_future(mechanism, purview))

    def test_phi_mip_past(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m1, self.m2]
        assert (s.phi_mip_past(mechanism, purview) ==
                s.mip_past(mechanism, purview).difference)

    def test_phi_mip_future(self):
        s = self.m_subsys_all
        mechanism = [self.m0, self.m1, self.m2]
        purview = [self.m0, self.m2]
        assert (s.phi_mip_future(mechanism, purview) ==
                s.mip_future(mechanism, purview).difference)

    # }}}


# Helper for checking MIP equality {{{
# ====================================

def mip_eq(a, b):
    """Return whether two MIPs are equal."""
    if not a or not b:
        return a == b
    return ((a.partition == b.partition or a.partition == (b.partition[1],
                                                           b.partition[0])) and
            (a.difference == b.difference) and
            (np.array_equal(a.repertoire, b.repertoire)))

# }}}

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker
