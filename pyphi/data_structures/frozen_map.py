# pyright: strict
# data_structures/frozen_map.py

import typing

K = typing.TypeVar("K")
V = typing.TypeVar("V")


class FrozenMap(typing.Mapping[K, V]):
    """An immutable mapping from keys to values.

    Notes
    -----
    Instances are used as cache keys (repertoires are memoized on the
    mapping from mechanism node to that node's state), so the hash must
    distinguish mappings that differ only in which key holds which value.
    Hashing the key set and the value set separately does not: over binary
    units every mapping on a given mechanism shares one key set and draws
    its values from ``{0, 1}``, collapsing all 2ⁿ mappings onto three
    hashes. Every lookup then degenerates to a linear scan of the bucket
    under :meth:`~collections.abc.Mapping.__eq__`, making cache operations
    quadratic in the number of entries.
    """

    __slots__ = ("_dict", "_hash")

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # Type ignore needed because dict(*args, **kwargs) can create various key types
        self._dict: dict[K, V] = dict(*args, **kwargs)  # type: ignore[assignment]
        self._hash: int | None = None

    def __getitem__(self, key: K) -> V:
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __iter__(self) -> typing.Iterator[K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return f"FrozenMap({self._dict!r})"

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash

    def replace(self, /, **changes: V) -> "FrozenMap[K, V]":
        return self.__class__(self, **changes)
