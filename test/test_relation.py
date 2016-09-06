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


def test_shared_purview_elements(concept_set):
    assert concept_set.shared_purview('past') == (0, 1)
    assert concept_set.shared_purview('future') == (1,)


def test_possible_purviews(concept_set):
    assert concept_set.possible_purviews('past') == [(0,), (1,), (0, 1)]
    assert concept_set.possible_purviews('future') == [(1,)]


def test_find_relation(concept_set):
    print(relation.find_relation('past', concept_set.concepts))
    print(relation.find_relation('future', concept_set.concepts))
    assert False
