# data_structures/frozen_map.py

import typing

K = typing.TypeVar("K")
V = typing.TypeVar("V")


class FrozenMap(typing.Mapping[K, V]):
    """An immutable mapping from keys to values."""

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
