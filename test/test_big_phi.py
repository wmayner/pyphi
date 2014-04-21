#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from cyphi import constants
from cyphi import utils
from cyphi.models import Concept
from cyphi import compute


def test_null_concept(s):
    assert (s.null_concept() == Concept(
        mechanism=(),
        location=np.array([s.unconstrained_cause_repertoire(s.nodes),
                           s.unconstrained_effect_repertoire(s.nodes)]),
        phi=0, cause=None, effect=None))


def test_concept_nonexistent(s):
    assert not s.concept((s.nodes[0], s.nodes[2]))


# TODO finish
def test_concept(s):
    pass


@pytest.mark.slow
def test_big_phi_standard_example(s):
    initial_precision = constants.PRECISION
    constants.PRECISION = 4
    assert utils.phi_eq(compute.big_phi(s), 2.3125)
    constants.PRECISION = initial_precision
