#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_config.py

import logging
import os

import pytest

from pyphi import config
from pyphi.conf import Config, option


@pytest.fixture
def c():
    return Config()


def test_load_config_dict(c):
    c.load_config_dict({'KEY': 'VALUE'})
    assert c.KEY == 'VALUE'


def test_snapshot(c):
    c.KEY = 'VALUE'
    snapshot = c.snapshot()
    assert snapshot == {'KEY': 'VALUE'}
    c.KEY = 'ANOTHER'
    assert snapshot == {'KEY': 'VALUE'}


EXAMPLE_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'example_config.yml')


def test_load_config_file(c):
    c.load_config_file(EXAMPLE_CONFIG_FILE)
    assert c.PRECISION == 100
    assert c.SOME_OTHER_CONFIG == 'loaded'


def test_str(c):
    c.KEY = 'VALUE'
    assert str(c) == "{'KEY': 'VALUE'}"


def test_override(c):
    # Given some config value
    c.TEST_CONFIG = 1

    @c.override(TEST_CONFIG=1000)
    def return_test_config(arg, kwarg=None):
        # Decorator should still pass args
        assert arg == 'arg'
        assert kwarg == 3
        return c.TEST_CONFIG

    # Should override config value in function
    assert return_test_config('arg', kwarg=3) == 1000
    # and revert the initial config value
    assert c.TEST_CONFIG == 1


def test_override_cleans_up_after_exception(c):
    c.TEST_CONFIG = 1

    @c.override(TEST_CONFIG=1000)
    def raise_exception():
        raise ValueError('elephants')

    try:
        raise_exception()
    except ValueError as e:
        # Decorator should reraise original exception
        assert e.args == ('elephants',)

    # and reset original config value
    assert c.TEST_CONFIG == 1


def test_override_config_is_a_context_manager(c):
    c.TEST_CONFIG = 1

    with c.override(TEST_CONFIG=1000):
        # Overriden
        assert c.TEST_CONFIG == 1000

    # Reverts original value
    assert c.TEST_CONFIG == 1


class ExampleConfig(Config):
    SPEED = option('default', values=['default', 'slow', 'fast'])


def test_option_descriptor():
    c = ExampleConfig()
    assert c.SPEED == 'default'

    assert c.__class__.__dict__['SPEED'].name == 'SPEED'

    c.SPEED = 'slow'
    assert c.SPEED == 'slow'

    with pytest.raises(ValueError):
        c.SPEED = 'medium'


def test_config_defaults():
    c = ExampleConfig()
    assert c.defaults() == {'SPEED': 'default'}
    c.SPEED = 'slow'
    assert c.defaults() == {'SPEED': 'default'}


def test_option_on_change():
    class Event:
        def notify(self, config):
            self.notified = config.SPEED
    event = Event()

    class AnotherConfig(Config):
        SPEED = option('default', on_change=event.notify)

    c = AnotherConfig()
    c.SPEED = 'slow'
    assert event.notified == 'slow'

    c.load_config_dict({'SPEED': 'fast'})
    assert event.notified == 'fast'


def test_logging_is_reconfigured_on_change(capsys):
    log = logging.getLogger('pyphi.config')

    with config.override(LOG_STDOUT_LEVEL='WARNING'):
        log.warning('Just a warning, folks.')
    out, err = capsys.readouterr()
    assert 'Just a warning, folks.' in err

    with config.override(LOG_STDOUT_LEVEL='ERROR'):
        log.warning('Another warning.')
    out, err = capsys.readouterr()
    assert err == ''
