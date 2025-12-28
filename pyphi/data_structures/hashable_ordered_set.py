# pyright: strict
# data_structures/hashable_ordered_set.py


from ordered_set import OrderedSet


class HashableOrderedSet(OrderedSet):
    """An OrderedSet that implements the hash method.

    For efficiency the hash is computed only once, when first called.

    The user is responsible for not mutating the set so that the hash value
    remains valid.
    """

    _precomputed_hash: int

    def __hash__(self) -> int:
        try:
            return self._precomputed_hash
        except AttributeError:
            self._precomputed_hash = self._hash()
            return self._precomputed_hash

    def __eq__(self, other: object) -> bool:
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

    def __getstate__(self) -> tuple[list, ...]:  # type: ignore[override]
        # In pickle, the state can't be an empty list.
        # We need to return a truthy value, or else __setstate__ won't be run.
        # This ensures a truthy value even if the set is empty.
        # NOTE: Override return type from OrderedSet for pickle compatibility
        return (list(self),)

    def __setstate__(self, state: tuple[list, ...]) -> None:
        super().__init__(state[0])
