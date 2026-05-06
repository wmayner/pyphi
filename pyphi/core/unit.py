"""Unit value type — atomic node in a substrate.

Roughly today's :class:`pyphi.node.Node` minus the per-instance TPM
caching, but layered as a pure value type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Unit:
    """An atomic node in a substrate.

    P7: alphabet is implicit binary (0 or 1). P12 adds ``alphabet_size``.
    """

    index: int
    label: str
