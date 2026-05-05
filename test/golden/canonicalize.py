"""Canonicalization helpers for golden fixtures.

The IIT formalism produces several quantities that are mathematically equivalent
under reordering or tie-breaking. Without canonicalization, golden fixtures would
fail under benign refactors (e.g., changes in iteration order). These helpers
turn each such quantity into a deterministic representation.

Key invariant: ``canonical(x) == canonical(y)`` iff ``x`` and ``y`` are
mathematically equivalent IIT objects.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def canonical_state_set(states: Iterable[tuple[int, ...]]) -> list[list[int]]:
    """Lexicographically sort a set of tied states.

    IIT specified-state computation can produce ties when multiple states
    achieve the same intrinsic information. The implementation picks one
    arbitrarily; the canonical form is the lex-sorted set of all valid ties.

    Returns a list of lists (JSON-serializable), not tuples.
    """
    return sorted([list(s) for s in states])


def canonical_partition(partition: Any) -> list:
    """Reduce a partition object to a deterministic structural form.

    Accepts:
    - ``Cut`` (from_nodes, to_nodes)
    - ``KCut`` / ``KPartition`` / ``Bipartition`` (sequence of Parts)
    - ``SystemPartition`` (parts with directions)
    - ``NullCut`` / ``CompleteSystemPartition`` (markers)

    Returns nested lists of int sequences. Each "part" is internally sorted;
    parts are sorted among themselves. For directed cuts, the (from, to)
    structure is preserved as a 2-tuple.

    Examples:
        Bipartition over {0,1,2}/{3} → [[0,1,2], [3]]
        Cut from {0} to {1,2} → [[0], [1,2]] with direction sentinel
    """
    if partition is None:
        return []

    cls_name = type(partition).__name__

    # NullCut / CompleteSystemPartition: marker classes
    if cls_name in {"NullCut", "CompleteSystemPartition", "_NullCut"}:
        return [["@null"]]

    # Cut: (from_nodes, to_nodes)
    if hasattr(partition, "from_nodes") and hasattr(partition, "to_nodes"):
        return [
            sorted(int(n) for n in partition.from_nodes),
            sorted(int(n) for n in partition.to_nodes),
        ]

    # KPartition / Bipartition / KCut: iterable of Parts
    parts = []
    try:
        for part in partition:
            if hasattr(part, "mechanism") and hasattr(part, "purview"):
                # Part with mechanism/purview attrs
                parts.append(
                    [
                        sorted(int(n) for n in part.mechanism),
                        sorted(int(n) for n in part.purview),
                    ]
                )
            elif hasattr(part, "__iter__"):
                # Iterable of indices
                parts.append([sorted(int(n) for n in part)])
            else:
                parts.append([[int(part)]])
        # Sort parts canonically
        return sorted(parts)
    except TypeError:
        # Fallback: stringify
        return [[str(partition)]]


def canonical_purview(purview: Iterable[int] | None) -> list[int] | None:
    """Sort a purview's node indices."""
    if purview is None:
        return None
    return sorted(int(n) for n in purview)


def canonical_mechanism(mechanism: Iterable[int]) -> list[int]:
    """Sort a mechanism's node indices."""
    return sorted(int(n) for n in mechanism)
