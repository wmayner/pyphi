"""Unit value type — atomic node in a substrate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Unit:
    """An atomic node in a substrate.

    Holds the node's index, label, and alphabet size (number of distinct
    states the node can take). Alphabet size defaults to 2 (binary). Math
    operations against ``Unit`` are parameterized by ``alphabet_size``;
    multi-valued substrates pass non-2 values.
    """

    index: int
    label: str
    alphabet_size: int = 2
