#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from cyphi.models import Cut
from cyphi import validate


def test_validate_nodelist_noniterable():
    with pytest.raises(ValueError):
        validate.nodelist(2, "it's a doge")


def test_validate_nodelist_nonnode():
    with pytest.raises(ValueError):
        validate.nodelist([0, 1, 2], 'invest in dogecoin!')


def test_validate_direction():
    with pytest.raises(ValueError):
        validate.direction("dogeeeee")


def test_validate_cm_valid(s):
    assert validate.connectivity_matrix(s.network.connectivity_matrix)


def test_validate_cm_not_square():
    cm = np.random.binomial(1, 0.5, (4, 5))
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_cm_not_2D():
    cm = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_cm_not_binary():
    cm = np.arange(16).reshape(4, 4)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)
