# cache/redis.py
"""Provides a Redis backend for caching."""

import pickle

from .. import constants
from ..conf import config

from .cache_utils import _CacheInfo

try:
    import redis

    NO_REDIS = False
except ModuleNotFoundError as exc:
    NO_REDIS = True


def init(db):
    if NO_REDIS:
        return None
    return redis.StrictRedis(
        host=config.REDIS_CONFIG["host"], port=config.REDIS_CONFIG["port"], db=db
    )


# Expose the StrictRedis API, maintaining one connection pool
# The connection pool is multi-process safe, and is reinitialized when the
# client detects a fork. See:
# https://github.com/andymccurdy/redis-py/blob/5109cb4f/redis/connection.py#L950
#
# TODO(redis): rebuild connection after config changes and warn in on_change if
# set to True
conn = init(config.REDIS_CONFIG["db"])


def available():
    """Check if the Redis server is connected."""
    try:
        return conn.ping()
    except redis.exceptions.ConnectionError:
        return False


# TODO: use a cache prefix?
# TODO: key schema for easy access/queries
class RedisCache:
    def clear(self):
        """Flush the cache."""
        conn.flushdb()
        conn.config_resetstat()

    @staticmethod
    def size():
        """Size of the Redis cache.

        .. note:: This is the size of the entire Redis database.
        """
        return conn.dbsize()

    def info(self):
        """Return cache information.

        .. note:: This is not the cache info for the entire Redis key space.
        """
        info = conn.info()
        return _CacheInfo(info["keyspace_hits"], info["keyspace_misses"], self.size())

    def get(self, key):
        """Get a value from the cache.

        Returns None if the key is not in the cache.
        """
        value = conn.get(key)

        if value is not None:
            value = pickle.loads(value)

        return value

    def set(self, key, value):
        """Set a value in the cache."""
        value = pickle.dumps(value, protocol=constants.PICKLE_PROTOCOL)
        conn.set(key, value)

    def key(self):
        """Delegate to subclasses."""
        raise NotImplementedError
