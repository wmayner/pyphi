#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_relations.py

from pyphi.relations.partition import wedge_partitions, rwedge_partitions
from pyphi.relations import relations

import pyphi


def test_maximal_states():
    subsystem = pyphi.examples.PQR()
    ces = relations.separate_ces(pyphi.compute.ces(subsystem))
    result = [relations.maximal_states(subsystem, ce) for ce in ces]
    assert result == [
        {(0, 0, 0)},
        {(0, 0, 0)},
        {(0, 0, 0), (1, 1, 0)},
        {(0, 0, 0)},
        {(0, 1, 0)},
        {(0, 0, 1)},
        {(1, 1, 0)},
        {(0, 0, 1)}
    ]
    # network = pyphi.examples.rule154_network()
    # subsystem = pyphi.Subsystem(network, (0,) * network.size)
    # ces = relations.separate_ces(pyphi.compute.ces(subsystem))
    # result = [relations.maximal_states(subsystem, ce) for ce in ces]


def test_PQR_relations():
    PQR = pyphi.examples.PQR()
    with pyphi.config.override(
            PARTITION_TYPE='TRI', MEASURE='BLD', PRECISION=5):
        ces = pyphi.compute.ces(PQR)
        separated_ces = list(relations.separate_ces(ces))
        results = list(relations.relations(PQR, ces))
        # answers = [
        #     [(0, 4), 0.240228, {(2,)}],
        #     [(0, 6), 0.120114, {(2,)}],
        #     [(1, 3), 0.060055, {(0,)}],
        #     [(1, 7), 0.120111, {(0,)}],
        #     [(2, 3), 0.120110, {(0, 1)}],
        #     [(2, 6), 0.080075, {(0, 1)}],
        #     [(2, 7), 0.120112, {(0, 1)}],
        #     [(3, 7), 0.120112, {(0,),
        #                         (0, 1)}],
        #     [(4, 6), 0.240227, {(1, 2)}],
        #     [(5, 7), 0.060057, {(2,)}],
        #     [(0, 4, 6), 0.083257, {(2,)}],
        #     [(1, 3, 7), 0.041627, {(0,)}],
        #     [(2, 3, 7), 0.041627, {(0, 1)}]
        # ]
        answers = [
            [(0, 4), 0.240227, {(2,)}],
            [(0, 6), 0.120113, {(2,)}],
            [(1, 3), 0.060057, {(0,)}],
            [(1, 7), 0.120113, {(0,)}],
            [(2, 3), 0.120113, {(0, 1)}],
            [(2, 6), 0.080076, {(0, 1)}],
            [(2, 7), 0.120113, {(0, 1)}],
            [(3, 7), 0.120113, {(0,), (0, 1)}],
            [(4, 6), 0.240227, {(1, 2)}],
            [(5, 7), 0.060057, {(2,)}],
            [(0, 4, 6), 0.083256, {(2,)}],
            [(1, 3, 7), 0.041628, {(0,)}],
            [(2, 3, 7), 0.041628, {(0, 1)}]
        ]
        for result, answer in zip(results, answers):
            subset, phi, purview = answer
            relata = tuple(separated_ces[i] for i in subset)
            relation = relations.Relation(relata, phi, purview)
            assert result == relation
