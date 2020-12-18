#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_metrics.py

from pyphi import metrics


def test_default_measures():
    assert set(metrics.measures.all()) == set(
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
            "PMI",
            "WPMI",
        ]
    )


def test_default_asymmetric_measures():
    assert set(metrics.measures.asymmetric()) == set(
        ["KLD", "MP2Q", "AID", "KLM", "BLD", "ID", "PMI", "WPMI"]
    )