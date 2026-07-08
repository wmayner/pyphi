"""Unit value type — atomic node in a substrate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Unit:
    """An atomic node in a substrate.

    An immutable, hashable value holding the node's index, label, and
    alphabet size (the number of distinct states the node can take).

    Attributes
    ----------
    index : int
        The node's position within its substrate.
    label : str
        The node's display name.
    alphabet_size : int
        The number of distinct states the node can take. Defaults to 2
        (binary); multi-valued substrates use larger sizes.
    """

    index: int
    label: str
    alphabet_size: int = 2
