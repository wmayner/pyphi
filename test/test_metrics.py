#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_metrics.py

from pyphi import metrics


def test_default_distribution_measures():
    assert set(metrics.distribution.measures.all()) == set(
        [
            "EMD",
            "L1",
            "KLD",
            "ENTROPY_DIFFERENCE",
            "PSQ2",
            "MP2Q",
            "AID",
            "KLM",
            "BLD",
            "ID",
        ]
    )


def test_default_asymmetric_distribution_measures():
    assert set(metrics.distribution.measures.asymmetric()) == set(
        ["KLD", "MP2Q", "AID", "KLM", "BLD", "ID"]
    )


def test_default_ces_measures():
    assert set(metrics.ces.measures.all()) == set(
        [
            "EMD",
            "SUM_SMALL_PHI",
        ]
    )


def test_default_actual_causation_measures():
    assert set(metrics.actual.measures.all()) == set(
        [
            "PMI",
            "WPMI",
        ]
    )