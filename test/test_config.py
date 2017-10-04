#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_config.py

import logging
import os

from pyphi import config
from pyphi.conf import Config


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


def test_direct_assignment():
    c = Config()
    c.KEY = 'VALUE'
    assert c._values['KEY'] == 'VALUE'


def test_load_config_dict():
    c = Config()
    c.load_config_dict({'KEY': 'VALUE'})
    assert c.KEY == 'VALUE'


EXAMPLE_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'example_config.yml')

def test_load_config_file():
    c = Config()
    c.load_config_file(EXAMPLE_CONFIG_FILE)
    assert c.PRECISION == 100
    assert c.SOME_OTHER_CONFIG == 'loaded'


def test_log_through_progress_handler(capsys):
    log = logging.getLogger('pyphi.config')
    with config.override(LOG_STDOUT_LEVEL='INFO'):
        config.configure_logging()
        log.warning('Just a warning, folks.')

    out, err = capsys.readouterr()
    assert 'Just a warning, folks.' in err

    # Reset logging
    # TODO: handle automatically
    config.configure_logging()
