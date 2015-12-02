
from pyphi import cache


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
