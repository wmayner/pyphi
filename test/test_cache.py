import functools
import multiprocessing
from unittest import mock

import pytest
import redis

from pyphi import Direction, Subsystem, cache, config, examples, models


def test_cache():
    c = cache.DictCache()
    key = (0, 1)
    value = "value"

    assert c.get(key) is None
    assert c.hits == 0
    assert c.misses == 1
    assert c.info() == (0, 1, 0)
    assert c.size() == 0

    c.set(key, value)

    assert c.get(key) == value
    assert c.hits == 1
    assert c.misses == 1
    assert c.info() == (1, 1, 1)
    assert c.size() == 1

    c.clear()
    assert c.size() == 0
    assert c.hits == 0
    assert c.misses == 0


class SomeObject:
    """Object for testing cache decorator"""

    def __init__(self):
        self.my_cache = cache.DictCache()

    @cache.method("my_cache", "key_prefix")
    def cached_method(self, some_arg):
        return "expensive computation"


def test_cache_decorator():
    o = SomeObject()
    assert o.cached_method(1) == "expensive computation"
    # generated from the key prefix and method arguments
    expected_key = ("key_prefix", 1)
    assert expected_key in o.my_cache.cache


def test_cache_key_generation():
    c = cache.DictCache()
    assert c.key("arg", _prefix="CONSTANT") == ("CONSTANT", "arg")


def factory():
    """This function is necessary because CACHE_REPERTOIRES does not have an
    effect if changed at runtime.

    .. TODO:
        fix that
    """

    class SomeObject:
        """Object for testing CACHE_REPERTOIRES config option"""

        def __init__(self):
            self.repertoire_cache = cache.DictCache()

        @cache.method("repertoire_cache", "cause")
        def cause_repertoire(self, some_arg):
            return "expensive computation"

        @cache.method("repertoire_cache", "effect")
        def effect_repertoire(self, some_arg):
            return "expensive computation"

    return SomeObject


def test_cache_repertoires_config_option():

    with config.override(CACHE_REPERTOIRES=True):
        SomeObject = factory()
        o = SomeObject()
        assert o.cause_repertoire(1) == "expensive computation"
        assert o.effect_repertoire(1) == "expensive computation"
        expected_key = ("cause", 1)
        assert expected_key in o.repertoire_cache.cache
        expected_key = ("effect", 1)
        assert expected_key in o.repertoire_cache.cache

    with config.override(CACHE_REPERTOIRES=False):
        SomeObject = factory()
        o = SomeObject()
        assert o.cause_repertoire(1) == "expensive computation"
        assert o.effect_repertoire(1) == "expensive computation"
        # Repertoire cache should be empty
        assert not o.repertoire_cache.cache



# Test purview cache
# ==================


@config.override(CACHE_POTENTIAL_PURVIEWS=True)
def test_purview_cache(standard):
    purviews = standard.potential_purviews(Direction.EFFECT, (0,))
    assert standard.purview_cache.size() == 1
    assert purviews in standard.purview_cache.cache.values()


@config.override(CACHE_POTENTIAL_PURVIEWS=False)
def test_only_cache_purviews_if_configured():
    c = cache.PurviewCache()
    c.set(c.key(Direction.CAUSE, (0,)), ("some purview"))
    assert c.size() == 0
