#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# data_structures.py

from ordered_set import OrderedSet
import collections.abc
import typing


class HashableOrderedSet(OrderedSet):
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

    def __getstate__(self):
        # In pickle, the state can't be an empty list.
        # We need to return a truthy value, or else __setstate__ won't be run.
        # This ensures a truthy value even if the set is empty.
        return (list(self),)

    def __setstate__(self, state):
        self.__init__(state[0])


K = typing.TypeVar("K")
V = typing.TypeVar("V")


class FrozenMap(typing.Generic[K, V], collections.abc.Mapping[K, V]):
    __slots__ = ("_dict", "_hash")

    def __init__(self, *args, **kwargs):
        self._dict: typing.Dict[K, V] = dict(*args, **kwargs)
        self._hash: typing.Optional[int] = None

    def __getitem__(self, key: K) -> V:
        return self._dict[key]

    def __contains__(self, key: K) -> bool:
        return key in self._dict

    def __iter__(self) -> typing.Iterator[K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f"FrozenMap({repr(self._dict)})"

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(
                (frozenset(self._dict), frozenset(iter(self._dict.values())))
            )
        return self._hash

    def replace(self, /, **changes):
        return self.__class__(self, **changes)
