#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# registry.py

"""
A function registry for storing custom measures and partition strategies.
"""

import collections


class Registry(collections.Mapping):
    """Generic registry for user-supplied functions.

    See ``pyphi.subsystem.PartitionRegistry`` and
    ``pyphi.distance.MeasureRegistry`` for concrete usage examples.
    """
    desc = ''

    def __init__(self):
        self.store = {}

    def register(self, name):
        """Decorator for registering a function with PyPhi.

        Args:
            name (string): The name of the function
        """
        def register_func(func):
            self.store[name] = func
            return func
        return register_func

    def all(self):
        """Return a list of all registered functions"""
        return list(self)

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __getitem__(self, name):
        try:
            return self.store[name]
        except KeyError:
            raise KeyError(
                '"{}" not found. Try using one of the installed {} {} or '
                'register your own.'.format(name, self.desc, self.all()))
