#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_config.py

import os

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


def test_override_config_is_a_context_manager():
    config.TEST_CONFIG = 1

    with config.override(TEST_CONFIG=1000):
        # Overriden
        assert config.TEST_CONFIG == 1000

    # Reverts original value
    assert config.TEST_CONFIG == 1


EXAMPLE_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'example_config.yml')


@config.override(PRECISION=6)
def test_load_config_file():
    config.load_config_file(EXAMPLE_CONFIG_FILE)
    assert config.PRECISION == 100
    assert config.SOME_OTHER_CONFIG == 'loaded'
