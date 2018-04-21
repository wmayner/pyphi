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


@pytest.mark.parametrize('direction,json_dict', [
    (Direction.CAUSE, {'direction': 'CAUSE'}),
    (Direction.EFFECT, {'direction': 'EFFECT'}),
    (Direction.BIDIRECTIONAL, {'direction': 'BIDIRECTIONAL'})])
def test_direction_json(direction, json_dict):
    assert direction.to_json() == json_dict
    assert Direction.from_json(json_dict) == direction


def test_direction_str():
    assert str(Direction.CAUSE) == 'CAUSE'
    assert str(Direction.EFFECT) == 'EFFECT'
    assert str(Direction.BIDIRECTIONAL) == 'BIDIRECTIONAL'
