import functools
from unittest import mock
import pytest
from pyphi import cache, config, models, Subsystem


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
    """Object for testing cache decorator"""
    def __init__(self):
        self.my_cache = cache.DictCache()

    @cache.method_cache('my_cache', 'key_prefix')
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

# TODO: only run redis tests if there is a running server

def test_redis_singleton_connection():
    conn = cache.RedisConn()
    assert conn.ping() is True


def all_caches(test_func):
    """Decorator to run a test function with local and Redis caches"""
    @pytest.mark.parametrize("redis_cache", [
        (True,), (False,),
    ])
    def wrapper(redis_cache, *args, **kwargs):
        override = config.override(REDIS_CACHE=redis_cache)
        return override(test_func)(redis_cache, *args, **kwargs)

    return functools.wraps(test_func)(wrapper)

redis_cache = config.override(REDIS_CACHE=True)
local_cache = config.override(REDIS_CACHE=False)


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
    assert c.key('past', (0,), purviews=(0, 1)) == (None, 'past', (0,), (0, 1))

    c = cache.RedisMiceCache(s)
    answer = 'subsys:{}:None:past:(0,):(0, 1)'.format(hash(s))
    assert c.key('past', (0,), purviews=(0, 1)) == answer


@all_caches
def test_mice_cache(redis_cache, s):
    mechanism = (1,)  # has a core cause
    mice = s.find_mice('past', mechanism)
    key = s._mice_cache.key('past', mechanism)
    assert s._mice_cache.get(key) == mice


@local_cache
def test_do_not_cache_phi_zero_mice(s):
    mechanism = ()  # zero phi
    mice = s.find_mice('past', mechanism)
    assert mice.phi == 0
    # don't cache anything because mice.phi == 0
    assert s._mice_cache.size() == 0


@all_caches
def test_only_cache_uncut_subsystem_mices(redis_cache, standard):
    s = Subsystem(standard, (1, 0, 0), range(standard.size),
                  cut=models.Cut((1,), (0, 2)))
    mechanism = (1,)  # has a core cause
    s.find_mice('past', mechanism)
    # don't cache anything because subsystem is cut
    assert s._mice_cache.size() == 0


@all_caches
def test_split_mechanism_mice_is_not_reusable(redis_cache, s):
    """If mechanism is split, then cached mice are not usable
    when a cache is built from a parent cache."""
    mechanism = (0, 1)
    mice = s.find_mice('past', mechanism)
    assert s._mice_cache.size() == 1  # cached
    assert mice.purview == (1, 2)

    # Splits mechanism, but not relevant connections:
    cut = models.Cut((0,), (1, 2))
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    key = cut_s._mice_cache.key('past', mechanism)
    assert cut_s._mice_cache.get(key) is None


@all_caches
def test_cut_relevant_connections_mice_is_not_reusable(redis_cache, s):
    """If relevant connections are cut, cached mice are not usable
    when a cache is built from a parent cache."""
    mechanism = (1,)
    mice = s.find_mice('past', mechanism)
    assert s._mice_cache.size() == 1  # cached
    assert mice.purview == (2,)

    # Cuts connections from 2 -> 1
    cut = models.Cut((0, 2), (1,))
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    key = cut_s._mice_cache.key('past', mechanism)
    assert cut_s._mice_cache.get(key) is None


@all_caches
def test_inherited_mice_cache_keeps_unaffected_mice(redis_cache, s):
    """Cached Mice are saved from the parent cache if both
    the mechanism and the relevant connections are not cut."""
    mechanism = (1,)
    mice = s.find_mice('past', mechanism)
    assert s._mice_cache.size() == 1  # cached
    assert mice.purview == (2,)

    # Does not cut from 0 -> 1 or split mechanism
    cut = models.Cut((0, 1), (2,))
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    key = cut_s._mice_cache.key('past', mechanism)
    assert cut_s._mice_cache.get(key) is mice


@all_caches
def test_inherited_cache_must_come_from_uncut_subsystem(redis_cache, s):
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=models.Cut((0, 2), (1,)))
    with pytest.raises(ValueError):
        cache.MiceCache(s, cut_s._mice_cache)


@local_cache
@config.override(MAXIMUM_CACHE_MEMORY_PERCENTAGE=0)
def test_mice_cache_respects_cache_memory_limits(s):
    c = cache.MiceCache(s)
    mice = mock.Mock(phi=1)  # dummy Mice
    c.set(c.key('past', ()), mice)
    assert c.size() == 0


# Test purview cache
# ==================

@config.override(CACHE_POTENTIAL_PURVIEWS=True)
def test_purview_cache(standard):
    purviews = standard._potential_purviews('future', (0,))
    assert standard.purview_cache.size() == 1
    assert purviews in standard.purview_cache.cache.values()


@config.override(CACHE_POTENTIAL_PURVIEWS=False)
def test_only_cache_purviews_if_configured():
    c = cache.PurviewCache()
    c.set(c.key('past', (0,)), ('some purview'))
    assert c.size() == 0
