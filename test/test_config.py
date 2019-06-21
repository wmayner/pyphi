#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_config.py

import logging
import os
from pathlib import Path
import shutil

import pytest

from pyphi import config, constants
from pyphi.conf import Config, Option


class ExampleConfig(Config):
    SPEED = Option('default', values=['default', 'slow', 'fast'])


@pytest.fixture
def c():
    return ExampleConfig()


def test_load_dict(c):
    c.load_dict({'SPEED': 'slow'})
    assert c.SPEED == 'slow'


def test_snapshot(c):
    c.SPEED = 'slow'
    snapshot = c.snapshot()
    assert snapshot == {'SPEED': 'slow'}
    c.SPEED = 'fast'
    assert snapshot == {'SPEED': 'slow'}


EXAMPLE_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'example_config.yml')


def test_load_file(c):
    c.load_file(EXAMPLE_CONFIG_FILE)
    assert c.SPEED == 'slow'
    assert c._loaded_files == [EXAMPLE_CONFIG_FILE]


def test_str(c):
    c.SPEED = 'slow'
    assert str(c) == "{'SPEED': 'slow'}"


def test_override(c):
    @c.override(SPEED='slow')
    def return_test_config(arg, kwarg=None):
        # Decorator should still pass args
        assert arg == 'arg'
        assert kwarg == 3
        return c.SPEED

    # Should override config value in function
    assert return_test_config('arg', kwarg=3) == 'slow'
    # and revert the initial config value
    assert c.SPEED == 'default'


def test_override_cleans_up_after_exception(c):
    @c.override(SPEED='slow')
    def raise_exception():
        raise ValueError('elephants')

    try:
        raise_exception()
    except ValueError as e:
        # Decorator should reraise original exception
        assert e.args == ('elephants',)

    # and reset original config value
    assert c.SPEED == 'default'


def test_override_is_a_context_manager(c):
    c.SPEED = 'slow'

    with c.override(SPEED='fast'):
        # Overriden
        assert c.SPEED == 'fast'

    # Reverts original value
    assert c.SPEED == 'slow'


def test_option_descriptor(c):
    assert c.SPEED == 'default'
    assert c.__class__.__dict__['SPEED'].name == 'SPEED'

    c.SPEED = 'slow'
    assert c.SPEED == 'slow'

    with pytest.raises(ValueError):
        c.SPEED = 'medium'


def test_defaults(c):
    assert c.defaults() == {'SPEED': 'default'}
    c.SPEED = 'slow'
    assert c.defaults() == {'SPEED': 'default'}


def test_only_set_public__attributes_that_are_options(c):
    with pytest.raises(ValueError):
        c.another_attribute = 2


def test_can_set_private_attributes(c):
    c._private = 2
    assert c._private == 2


def test_on_change():
    class Event:
        def notify(self, config):
            self.notified = config.SPEED
    event = Event()

    class AnotherConfig(Config):
        SPEED = Option('default', on_change=event.notify)

    c = AnotherConfig()
    assert event.notified == 'default'

    c.SPEED = 'slow'
    assert event.notified == 'slow'

    c.load_dict({'SPEED': 'fast'})
    assert event.notified == 'fast'


def test_reconfigure_logging_on_change(capsys):
    log = logging.getLogger('pyphi.config')

    with config.override(LOG_STDOUT_LEVEL='WARNING'):
        log.warning('Just a warning, folks.')
    out, err = capsys.readouterr()
    assert 'Just a warning, folks.' in err

    with config.override(LOG_STDOUT_LEVEL='ERROR'):
        log.warning('Another warning.')
    out, err = capsys.readouterr()
    assert err == ''


def test_reconfigure_precision_on_change():
    with config.override(PRECISION=100):
        assert constants.EPSILON == 1e-100

    with config.override(PRECISION=3):
        assert constants.EPSILON == 1e-3

    with config.override(PRECISION=123):
        assert constants.EPSILON == 1e-123


def test_reconfigure_joblib_on_change(capsys):
    cachedir = './__testing123__'
    try:
        with config.override(FS_CACHE_DIRECTORY=cachedir):
            assert constants.joblib_memory.location == cachedir
            assert Path(cachedir).exists()
    finally:
        shutil.rmtree(cachedir)

    def f(x):
        return x + 1

    with config.override(FS_CACHE_VERBOSITY=0):
        constants.joblib_memory.cache(f)(42)
    out, err = capsys.readouterr()
    assert len(out) == 0

    with config.override(FS_CACHE_VERBOSITY=100):
        constants.joblib_memory.cache(f)(42)
    out, err = capsys.readouterr()
    assert len(out) > 0


@config.override()
@pytest.mark.parametrize('name,valid,invalid', [
    ('SYSTEM_CUTS', ['3.0_STYLE', 'CONCEPT_STYLE'], ['OTHER']),
    ('REPR_VERBOSITY', [0, 1, 2], [-1, 3])])
def test_config_validation(name, valid, invalid):
    for value in valid:
        setattr(config, name, value)

    for value in invalid:
        with pytest.raises(ValueError):
            setattr(config, name, value)
