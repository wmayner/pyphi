#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_relations.py

import numpy as np

from pyphi import compute, config, examples, relations, utils
from pyphi.models.subsystem import FlatCauseEffectStructure


def test_PQR_relations():
    with config.override(
        PARTITION_TYPE="TRI",
        REPERTOIRE_DISTANCE="BLD",
    ):
        PQR = examples.PQR()
        ces = compute.subsystem.ces(PQR)
        separated_ces = FlatCauseEffectStructure(ces)
        results = list(relations.relations(PQR, ces))

        # NOTE: these phi values are in nats, not bits!
        answers = [
            [(0, 4), 0.6931471805599452, [(2,)]],
            [(0, 6), 0.6931471805599452, [(2,)]],
            [(1, 2), 0.3465735902799726, [(0,)]],
            [(1, 3), 0.3465735902799726, [(0,)]],
            [(1, 7), 0.3465735902799726, [(0,)]],
            [(2, 3), 0.3465735902799726, [(0,), (1,), (0, 1)]],
            [(2, 4), 0.3465735902799726, [(1,)]],
            [(2, 6), 0.3465735902799726, [(0,), (1,), (0, 1)]],
            [(2, 7), 0.3465735902799726, [(0,), (1,), (0, 1)]],
            [(3, 7), 0.693147180559945, [(0, 1)]],
            [(4, 6), 1.3862943611198901, [(1, 2)]],
            [(5, 7), 0.6931471805599452, [(2,)]],
            [(0, 4, 6), 0.6931471805599452, [(2,)]],
            [(1, 2, 3), 0.3465735902799726, [(0,)]],
            [(1, 2, 7), 0.3465735902799726, [(0,)]],
            [(1, 3, 7), 0.3465735902799726, [(0,)]],
            [(2, 3, 7), 0.3465735902799726, [(0,), (1,), (0, 1)]],
            [(2, 4, 6), 0.3465735902799726, [(1,)]],
            [(1, 2, 3, 7), 0.3465735902799726, [(0,)]],
        ]
        return results, answers

        def base2(x):
            return x / np.log(2.0)

        for result, answer in zip(results, answers):
            subset, phi, purviews = answer
            subset = tuple(separated_ces[i] for i in subset)
            relata = relations.Relata(PQR, subset)
            assert set(purviews) == set(result.ties)
            assert utils.eq(base2(phi), result.phi)
            assert relata == result.relata
