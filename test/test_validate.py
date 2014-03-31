#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from cyphi.validate import Validate


def test_validate_nodelist_noniterable():
    with pytest.raises(ValueError):
        Validate.nodelist(2, "it's a doge")


def test_validate_nodelist_nonnode():
    with pytest.raises(ValueError):
        Validate.nodelist([0, 1, 2], 'invest in dogecoin!')


def test_validate_direction():
    with pytest.raises(ValueError):
        Validate.direction("dogeeeee")
