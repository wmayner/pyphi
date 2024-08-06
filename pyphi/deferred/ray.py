# deferred/ray.py
"""Provide the optional Ray dependency."""


def identity_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class NoRay:
    """Provides a no-op implementation of the portion of the Ray API that is
    used at import time, to allow Ray to be an optional dependency."""

    def remote(self, func):
        return identity_decorator

    def init(self, *args, **kwargs):
        return None

    def is_initialized(self):
        return False

    def __repr__(self):
        return "<NoRay(): " + self.__doc__ + ">"


try:
    import ray

    NO_RAY = False
except ModuleNotFoundError:
    ray = NoRay()
    NO_RAY = True
