#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_relation.py

import numpy as np
import pytest

import pyphi
from pyphi import relation


@pytest.fixture()
def concept_set(s):
    c = pyphi.compute.constellation(s)

    # Concept 1: cause=(0, 1), mechanism=(2,), effect=(1,)
    # Concept 3: cause=(0, 1, 2), mechanism=(0, 1, 2), effect=(0, 1, 2)

    return relation.ConceptSet([c[1], c[3]])


def test_concepts_must_be_nonempty():
    with pytest.raises(ValueError):
        relation.ConceptSet([])


def test_concept_set_iteration(concept_set):
    list(concept_set)


def test_purview_overlap_elements(concept_set):
    assert concept_set.purview_overlap('past') == (0, 1)
    assert concept_set.purview_overlap('future') == (1,)


def test_possible_purviews(concept_set):
    assert list(concept_set.possible_purviews('past')) == [(0,), (1,), (0, 1)]
    assert list(concept_set.possible_purviews('future')) == [(1,)]


def test_maximal_overlap_purviews(concept_set):
    mo = relation.MaximalOverlap(0, (0, 1), 'past', concept_set.concepts)
    assert mo.concept_purviews == [(0, 1), (0, 1, 2)]


def test_relation_partitions(concept_set):
    purview = (0, 1)
    concept = concept_set.concepts[0]
    partitions = relation.relation_partitions('past', concept, purview)

    def _partition(p0m, p0p, p1m, p1p):
        return ((p0m, p0p), (p1m, p1p))

    assert np.array_equal(list(partitions), [
        _partition((2,), (1,), (), (0,)),  # subset (0,)
        _partition((2,), (0,), (), (1,)),  # subset (1,)
        _partition((2,), (), (), (0, 1)),  # subset (0, 1)
    ])


def test_find_relation(concept_set):
    print(relation.find_relation('past', concept_set.concepts))
    print(relation.find_relation('future', concept_set.concepts))
    assert False
