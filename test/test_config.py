#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_config.py

from pyphi import config


def test_override_config():
    # Given some config value
    config.TEST_CONFIG = 1

    @config.override(TEST_CONFIG=1000)
    def return_test_config(arg, kwarg=None):
        # Decorator should still pass args
        assert arg == 'arg'
        assert kwarg == 3
        return config.TEST_CONFIG

    # Should override config value in function
    assert return_test_config('arg', kwarg=3) == 1000
    # and revert the initial config value
    assert config.TEST_CONFIG == 1


def test_override_config_cleans_up_after_exception():
    config.TEST_CONFIG = 1

    @config.override(TEST_CONFIG=1000)
    def raise_exception():
        raise ValueError('elephants')

    try:
        raise_exception()
    except ValueError as e:
        # Decorator should reraise original exception
        assert e.args == ('elephants',)

    # and reset original config value
    assert config.TEST_CONFIG == 1
