#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cyphi import options


def test_epsilon():
    assert options.EPSILON == 10**-options.PRECISION


def test_set_precision():
    initial_precision = options.PRECISION
    options.PRECISION = 1
    assert options.EPSILON == 10**-1
    options.PRECISION = initial_precision
