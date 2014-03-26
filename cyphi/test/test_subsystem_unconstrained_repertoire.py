#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .example_networks import WithExampleNetworks


class TestUnconstrainedRepertoires(WithExampleNetworks):

    # Unconstrained cause repertoire tests {{{{
    # =========================================

        # Matlab default network {{{
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_unconstrained_cause_repertoire_matlab_0(self):
        # Purview {m0}
        assert np.array_equal(
            self.m_subsys_all.unconstrained_cause_repertoire(
                [self.m0]),
            np.array([[[0.5]], [[0.5]]]))

    def test_unconstrained_cause_repertoire_matlab_1(self):
        purview = [self.m0, self.m1]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_cause_repertoire(purview),
            np.array([[[0.25], [0.25]], [[0.25], [0.25]]]))

    def test_unconstrained_cause_repertoire_matlab_2(self):
        purview = [self.m0, self.m1, self.m2]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_cause_repertoire(purview),
            np.array([[[0.125, 0.125],
                       [0.125, 0.125]],
                      [[0.125, 0.125],
                       [0.125, 0.125]]]))

        # }}}

    # }}}

    # Unconstrained effect repertoire tests {{{
    # =========================================

        # Matlab default network {{{
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~

    def test_unconstrained_effect_repertoire_matlab_0(self):
        purview = [self.m0]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_effect_repertoire(purview),
            np.array([[[0.25]], [[0.75]]]))

    def test_unconstrained_effect_repertoire_matlab_1(self):
        purview = [self.m0, self.m1]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_effect_repertoire(purview),
            np.array([[[0.125], [0.125]], [[0.375], [0.375]]]))

    def test_unconstrained_effect_repertoire_matlab_2(self):
        purview = [self.m0, self.m1, self.m2]
        assert np.array_equal(
            self.m_subsys_all.unconstrained_effect_repertoire(purview),
            np.array([[[0.0625, 0.0625],
                       [0.0625, 0.0625]],
                      [[0.1875, 0.1875],
                       [0.1875, 0.1875]]]))

        # }}}

    # }}}

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
