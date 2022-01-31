#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data_structures.py

from collections.abc import Hashable
from ordered_set import OrderedSet


class HashableOrderedSet(Hashable, OrderedSet):
    """An OrderedSet that implements the hash method.

    For efficiency the hash is computed only once, when first called.

    The user is responsible for not mutating the set so that the hash value
    remains valid.
    """

    def __hash__(self):
        try:
            return self._precomputed_hash
        except AttributeError:
            self._precomputed_hash = self._hash()
            return self._precomputed_hash

    def __eq__(self, other):
        """Returns true if the containers have the same items.

        Example:
            >>> oset = OrderedSet([1, 3, 2])
            >>> oset == [1, 3, 2]
            True
            >>> oset == [1, 2, 3]
            False
            >>> oset == [2, 3]
            False
            >>> oset == OrderedSet([3, 2, 1])
            False
        """
        try:
            return hash(self) == hash(other)
        except TypeError:
            # If `other` can't be compared, it's not equal.
            return False
