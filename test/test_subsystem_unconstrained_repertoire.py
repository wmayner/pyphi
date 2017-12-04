#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Unconstrained cause repertoire tests {{{{
# =========================================
#   Matlab default network {{{
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_unconstrained_cause_repertoire_matlab_0(s):
    # Purview {m0}
    purview = (0,)
    assert np.array_equal(
        s.unconstrained_cause_repertoire(purview, ),
        np.array([[[0.5]], [[0.5]]]))


def test_unconstrained_cause_repertoire_matlab_1(s):
    purview = (0, 1)
    assert np.array_equal(
        s.unconstrained_cause_repertoire(purview),
        np.array([[[0.25], [0.25]], [[0.25], [0.25]]]))


def test_unconstrained_cause_repertoire_matlab_2(s):
    purview = (0, 1, 2)
    assert np.array_equal(
        s.unconstrained_cause_repertoire(purview),
        np.array([[[0.125, 0.125],
                   [0.125, 0.125]],
                  [[0.125, 0.125],
                   [0.125, 0.125]]]))


#   }}}
# }}}
# Unconstrained effect repertoire tests {{{
# =========================================
#   Matlab default network {{{
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_unconstrained_effect_repertoire_matlab_0(s):
    purview = (0,)
    assert np.array_equal(
        s.unconstrained_effect_repertoire(purview),
        np.array([[[0.25]], [[0.75]]]))


def test_unconstrained_effect_repertoire_matlab_1(s):
    purview = (0, 1)
    assert np.array_equal(
        s.unconstrained_effect_repertoire(purview),
        np.array([[[0.125], [0.125]], [[0.375], [0.375]]]))


def test_unconstrained_effect_repertoire_matlab_2(s):
    purview = (0, 1, 2)
    assert np.array_equal(
        s.unconstrained_effect_repertoire(purview),
        np.array([[[0.0625, 0.0625],
                   [0.0625, 0.0625]],
                  [[0.1875, 0.1875],
                   [0.1875, 0.1875]]]))


#   }}}
# }}}

# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
