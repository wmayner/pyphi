#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test/test_registry.py

import pytest

from pyphi import distance


def test_registry():
    registry = distance.Registry()

    assert 'DIFF' not in registry
    assert len(registry) == 0

    @registry.register('DIFF')
    def difference(a, b):
        return a - b

    assert 'DIFF' in registry
    assert len(registry) == 1
    assert registry['DIFF'] == difference

    with pytest.raises(KeyError):
        registry['HEIGHT']
