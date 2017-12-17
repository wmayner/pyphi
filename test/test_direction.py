#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_direction.py

import pytest

from pyphi import Direction


def test_direction_order():
    mechanism = (0,)
    purview = (1, 2)
    assert Direction.CAUSE.order(mechanism, purview) == (purview, mechanism)
    assert Direction.EFFECT.order(mechanism, purview) == (mechanism, purview)

    with pytest.raises(ValueError):
        Direction.BIDIRECTIONAL.order(mechanism, purview)
