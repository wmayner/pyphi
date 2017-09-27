import functools
from unittest import mock

import pytest
import redis

from pyphi import Direction, Subsystem, cache, config, examples, models


def test_cache():
    c = cache.DictCache()
    key = (0, 1)
    value = 'value'

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
    '''Object for testing cache decorator'''
    def __init__(self):
        self.my_cache = cache.DictCache()

    @cache.method('my_cache', 'key_prefix')
    def cached_method(self, some_arg):
        return 'expensive computation'


def test_cache_decorator():
    o = SomeObject()
    assert o.cached_method(1) == 'expensive computation'
    # generated from the key prefix and method arguments
    expected_key = ('key_prefix', 1)
    assert expected_key in o.my_cache.cache


def test_cache_key_generation():
    c = cache.DictCache()
    assert c.key('arg', _prefix='CONSTANT') == ('CONSTANT', 'arg')


# Test MICE caching
# ========================

# NOTE: All subsystems are loaded from `examples` internally instead of by
# pytest fixture because they must be constructed with the correct cache
# config.

try:
    redis_available = cache.RedisConn().ping()
except redis.exceptions.ConnectionError:
    redis_available = False

# Decorator to skip a test if Redis is not available
require_redis = pytest.mark.skipif(not redis_available,
                                   reason="requires a running Redis server")

# Decorator to force a test to use the local cache
local_cache = config.override(REDIS_CACHE=False)

# Decorator to force a test to use Redis cache; skip test if Redis is not available
redis_cache = lambda f: config.override(REDIS_CACHE=True)(require_redis(f))


def all_caches(test_func):
    '''Decorator to run a test twice: once with the local cache and once with Redis.

    Any decorated test must add a `redis_cache` argument.
    '''
    @pytest.mark.parametrize("redis_cache,", [
        require_redis((True,)),
        (False,),
    ])
    def wrapper(redis_cache, *args, **kwargs):
        with config.override(REDIS_CACHE=redis_cache[0]):
            return test_func(redis_cache, *args, **kwargs)

    return functools.wraps(test_func)(wrapper)


@pytest.fixture
def flush_redis():
    '''Fixture to flush and reset the Redis cache.'''
    try:
        conn = cache.RedisConn()
        conn.flushall()
        conn.config_resetstat()
    except redis.exceptions.ConnectionError:
        pass


@require_redis
def test_redis_singleton_connection():
    conn = cache.RedisConn()
    assert conn.ping() is True


@require_redis
def test_redis_cache_info(flush_redis):
    c = cache.RedisCache()
    assert c.info() == (0, 0, 0)
    key = 'key'
    c.get(key)  # miss
    c.set(key, 'value')
    c.get(key)  # hit
    assert c.size() == 1
    assert c.info() == (1, 1, 1)


@redis_cache
def test_use_redis_mice_cache(s):
    c = cache.MiceCache(s)
    assert isinstance(c, cache.RedisMiceCache)


@local_cache
def test_use_dict_mice_cache(s):
    c = cache.MiceCache(s)
    assert isinstance(c, cache.DictMiceCache)


def test_mice_cache_keys(s):
    c = cache.DictMiceCache(s)
    answer = (None, Direction.PAST, (0,), (0, 1))
    assert c.key(Direction.PAST, (0,), purviews=(0, 1)) == answer

    c = cache.RedisMiceCache(s)
    answer = 'subsys:{}:None:Direction.PAST:(0,):(0, 1)'.format(hash(s))
    assert c.key(Direction.PAST, (0,), purviews=(0, 1)) == answer


@all_caches
def test_mice_cache(redis_cache, flush_redis):
    s = examples.basic_subsystem()
    mechanism = (1,)  # has a core cause
    mice = s.find_mice(Direction.PAST, mechanism)
    key = s._mice_cache.key(Direction.PAST, mechanism)
    assert s._mice_cache.get(key) == mice


@local_cache
def test_do_not_cache_phi_zero_mice():
    s = examples.basic_subsystem()
    mechanism = ()  # zero phi
    mice = s.find_mice(Direction.PAST, mechanism)
    assert mice.phi == 0
    # don't cache anything because mice.phi == 0
    assert s._mice_cache.size() == 0


@all_caches
def test_only_cache_uncut_subsystem_mices(redis_cache, flush_redis, s):
    s = Subsystem(s.network, (1, 0, 0), s.node_indices,
                  cut=models.Cut((1,), (0, 2)))
    mechanism = (1,)  # has a core cause
    s.find_mice(Direction.PAST, mechanism)
    # don't cache anything because subsystem is cut
    assert s._mice_cache.size() == 0


@all_caches
def test_split_mechanism_mice_is_not_reusable(redis_cache, flush_redis):
    '''If mechanism is split, then cached mice are not usable
    when a cache is built from a parent cache.'''
    s = examples.basic_subsystem()
    mechanism = (0, 1)
    mice = s.find_mice(Direction.PAST, mechanism)
    assert s._mice_cache.size() == 1  # cached
    assert mice.purview == (1, 2)

    # Splits mechanism, but not relevant connections:
    cut = models.Cut((0,), (1, 2))
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    key = cut_s._mice_cache.key(Direction.PAST, mechanism)
    assert cut_s._mice_cache.get(key) is None


@all_caches
def test_cut_relevant_connections_mice_is_not_reusable(redis_cache, flush_redis):
    '''If relevant connections are cut, cached mice are not usable
    when a cache is built from a parent cache.'''
    s = examples.basic_subsystem()
    mechanism = (1,)
    mice = s.find_mice(Direction.PAST, mechanism)
    assert s._mice_cache.size() == 1  # cached
    assert mice.purview == (2,)

    # Cuts connections from 2 -> 1
    cut = models.Cut((0, 2), (1,))
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    key = cut_s._mice_cache.key(Direction.PAST, mechanism)
    assert cut_s._mice_cache.get(key) is None


@all_caches
def test_inherited_mice_cache_keeps_unaffected_mice(redis_cache, flush_redis):
    '''Cached Mice are saved from the parent cache if both
    the mechanism and the relevant connections are not cut.'''
    s = examples.basic_subsystem()
    mechanism = (1,)
    mice = s.find_mice(Direction.PAST, mechanism)
    assert s._mice_cache.size() == 1  # cached
    assert mice.purview == (2,)

    # Does not cut from 0 -> 1 or split mechanism
    cut = models.Cut((0, 1), (2,))
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    key = cut_s._mice_cache.key(Direction.PAST, mechanism)
    assert cut_s._mice_cache.get(key) == mice


@all_caches
def test_inherited_cache_must_come_from_uncut_subsystem(redis_cache, flush_redis):
    s = examples.basic_subsystem()
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=models.Cut((0, 2), (1,)))
    with pytest.raises(ValueError):
        cache.MiceCache(s, cut_s._mice_cache)


@local_cache
@config.override(MAXIMUM_CACHE_MEMORY_PERCENTAGE=0)
def test_mice_cache_respects_cache_memory_limits():
    s = examples.basic_subsystem()
    c = cache.MiceCache(s)
    mice = mock.Mock(phi=1)  # dummy Mice
    c.set(c.key(Direction.PAST, ()), mice)
    assert c.size() == 0


# Test purview cache
# ==================

@config.override(CACHE_POTENTIAL_PURVIEWS=True)
def test_purview_cache(standard):
    purviews = standard.potential_purviews(Direction.FUTURE, (0,))
    assert standard.purview_cache.size() == 1
    assert purviews in standard.purview_cache.cache.values()


@config.override(CACHE_POTENTIAL_PURVIEWS=False)
def test_only_cache_purviews_if_configured():
    c = cache.PurviewCache()
    c.set(c.key(Direction.PAST, (0,)), ('some purview'))
    assert c.size() == 0
