
from unittest import mock
from pyphi import cache, config, models, Subsystem


def test_cache():
    c = cache.DictCache()
    key = (0, 1)
    value = 'value'

    assert c.get(key) is None
    assert c.hits == 0
    assert c.misses == 1
    assert c.info() == (0, 1, 0)

    c.set(key, value)

    assert c.get(key) == value
    assert c.hits == 1
    assert c.misses == 1
    assert c.info() == (1, 1, 1)


class TestObject:
    """Object for testing cache decorator"""
    def __init__(self):
        self.my_cache = cache.DictCache()

    @cache.method_cache('my_cache', 'key_prefix')
    def cached_method(self, some_arg):
        return 'expensive computation'


def test_cache_decorator():
    o = TestObject()
    assert o.cached_method(1) == 'expensive computation'
    # generated from the key prefix and method arguments
    expected_key = ('key_prefix', 1)
    assert expected_key in o.my_cache.cache


# Test MICE caching
# ========================


def test_mice_cache(s):
    mechanism = (1,)  # has a core cause
    mice = s.find_mice('past', mechanism)
    assert s._mice_cache.get(('past', mechanism)) == mice


def test_do_not_cache_phi_zero_mice(s):
    mechanism = ()  # zero phi
    mice = s.find_mice('past', mechanism)
    assert mice.phi == 0
    # don't cache anything because mice.phi == 0
    assert len(s._mice_cache.cache) == 0


def test_only_cache_uncut_subsystem_mices(standard):
    s = Subsystem(standard, (1, 0, 0), range(standard.size),
                  cut=models.Cut((1,), (0, 2)))
    mechanism = (1,)  # has a core cause
    s.find_mice('past', mechanism)
    # don't cache anything because subsystem is cut
    assert len(s._mice_cache.cache) == 0


def test_inherited_mice_cache_does_not_return_split_mice(s):
    # If mechanism is split, then cached mice are not usable
    mechanism = (0, 1, 2)
    cut = models.Cut((1,), (0, 2))  # splits mechanism
    mice = s.find_mice('past', mechanism)
    assert mice.phi > 0  # gets cached
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    assert cut_s._mice_cache.get(('past', mechanism)) is None


def test_inherited_mice_cache_does_not_contain_cut_mice(s):
    # If relevant connections are cut, cached mice are not usable
    mechanism = (1,)
    mice = s.find_mice('past', mechanism)
    assert mice.phi > 0  # gets cached
    assert mice.purview == (2,)
    cut = models.Cut((0, 2), (1,))  # cuts connection from 0 -> 1
    cut_s = Subsystem(s.network, s.state, s.node_indices,
                      cut=cut, mice_cache=s._mice_cache)
    assert cut_s._mice_cache.get(('past', mechanism)) is None


@config.override(MAXIMUM_CACHE_MEMORY_PERCENTAGE=0)
def test_mice_cache_respects_cache_memory_limits(s):
    c = cache.MiceCache(s)
    mice = mock.Mock(phi=1)  # dummy Mice
    c.set(('past', ()), mice)
    assert len(c.cache) == 0


# TODO: test purview=False cache behavior
