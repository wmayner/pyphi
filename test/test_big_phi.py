#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cyphi.models import Concept


def test_null_concept(m):
    s = m.subsys_all
    assert (s.null_concept() == Concept(
        mechanism=(),
        location=np.array([s.unconstrained_cause_repertoire(s.nodes),
                           s.unconstrained_effect_repertoire(s.nodes)]),
        size=0, cause=None, effect=None))

def test_concept_nonexistent(m):
     s = m.subsys_all
     assert not s.concept((m.nodes[0], m.nodes[2]))


def test_concept(m):
    pass
