"""Bounded search for intrinsic units and complexes (Marshall et al.
2024, Sec. 2.2.2).

The recursion starts from the micro units, which are axiomatically
valid (Eqs. 15-16 gate macroing only). Each level derives candidate
decompositions ``V`` from the previous level's pool of valid units and
judges each ``(V, W)`` pair once -- validity is a property of the
decomposition, independent of the candidate's own mapping and update
grain. Valid decompositions emit their mapped and grained variants
into the pool. Footprints are processed smallest-first, so the
competitor set ``f(U^J, W^J)`` always draws on every unit already
validated at strictly finer footprints.

``f(U^J, W^J)`` is the set of systems assembled from valid units whose
micro constituents are proper subsets of ``U^J`` and whose background
apportionments are non-overlapping subsets of ``W^J``, excluding the
candidate's own constituent system. The set ``P(u)`` extends the same
assembly to the whole universe (Eq. 18), and a member is a complex if
it strictly beats every other member whose micro constituents overlap
its own (Eq. 19). Candidate systems whose state is unreachable under
their own TPM specify no cause and cannot exist; they are dropped.

All ``phi_s`` evaluations within one driver run share a memo keyed on
the hashable :class:`~pyphi.macro.system.MacroSystem`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain

_MAPPING_POLICIES = ("FAMILIES", "EXHAUSTIVE")
_APPORTIONMENT_POLICIES = ("NONE", "ENUMERATE")


@dataclass(frozen=True)
class SearchBounds:
    """Bounds on the intrinsic-unit search space.

    Attributes:
        max_constituents: Cap on ``|U^J|`` per candidate unit.
        max_update_grain: Largest update grain ``tau'`` per level.
        max_depth: Macroing levels above micro.
        mappings: ``"FAMILIES"`` (coarse-grainings and black-boxings)
            or ``"EXHAUSTIVE"`` (every surjective table, capped).
        exhaustive_cap: Largest sequence-state count for EXHAUSTIVE.
        apportionment: ``"NONE"`` or ``"ENUMERATE"`` (assign background
            micro units to derived candidates).
        max_background: Cap on apportioned units when enumerating.
    """

    max_constituents: int = 4
    max_update_grain: int = 1
    max_depth: int = 1
    mappings: str = "FAMILIES"
    exhaustive_cap: int = 8
    apportionment: str = "NONE"
    max_background: int = 0

    def __post_init__(self) -> None:
        if self.max_constituents < 1:
            raise ValueError(
                f"max_constituents must be >= 1; got {self.max_constituents}"
            )
        if self.max_update_grain < 1:
            raise ValueError(
                f"max_update_grain must be >= 1; got {self.max_update_grain}"
            )
        if self.max_depth < 0:
            raise ValueError(f"max_depth must be >= 0; got {self.max_depth}")
        if self.mappings not in _MAPPING_POLICIES:
            raise ValueError(
                f"unknown mappings policy {self.mappings!r}; "
                f"expected one of {_MAPPING_POLICIES}"
            )
        if self.apportionment not in _APPORTIONMENT_POLICIES:
            raise ValueError(
                f"unknown apportionment policy {self.apportionment!r}; "
                f"expected one of {_APPORTIONMENT_POLICIES}"
            )
        if self.apportionment == "ENUMERATE" and self.max_background == 0:
            raise ValueError('apportionment="ENUMERATE" requires max_background >= 1')

    @property
    def max_micro_grain(self) -> int:
        """Largest micro grain a derived unit can reach (grains compose
        down the hierarchy)."""
        return self.max_update_grain**self.max_depth


def _canonical_table(table: tuple[int, ...]) -> tuple[int, ...]:
    """The representative of ``{table, complement}``.

    A mapping and its complement define the same partition of the
    constituents' sequence-states into two classes; the macro unit's
    two state labels are conventional and the analysis is invariant
    under relabeling. The representative maps the all-OFF sequence to
    macro state 0.
    """
    if table[0] == 1:
        return tuple(1 - entry for entry in table)
    return table


def candidate_mappings(
    num_constituents: int, update_grain: int, bounds: SearchBounds
) -> tuple[tuple[int, ...], ...]:
    """Deduplicated candidate truth tables for a unit shape.

    FAMILIES: every non-degenerate ``coarse_grain`` on-count set
    (update grain 1 only, by the family's definition) plus every
    nonempty ``blackbox`` output subset (any grain). EXHAUSTIVE: every
    surjective table when the sequence-state count is within
    ``exhaustive_cap``; ``ValueError`` above it.

    Tables are canonicalized up to state-label complementation (the
    all-OFF sequence maps to macro state 0) and deduplicated,
    preserving first-seen order.
    """
    tables: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(table: tuple[int, ...]) -> None:
        table = _canonical_table(table)
        if table not in seen:
            seen.add(table)
            tables.append(table)

    if bounds.mappings == "FAMILIES":
        if update_grain == 1:
            counts = tuple(range(num_constituents + 1))
            for size in range(1, len(counts)):
                for on_counts in itertools.combinations(counts, size):
                    add(coarse_grain(num_constituents, on_counts))
        for size in range(1, num_constituents + 1):
            for outputs in itertools.combinations(range(num_constituents), size):
                add(blackbox(num_constituents, update_grain, outputs))
    else:  # EXHAUSTIVE
        num_states = (2**num_constituents) ** update_grain
        if num_states > bounds.exhaustive_cap:
            raise ValueError(
                f"EXHAUSTIVE mappings for {num_constituents} constituents "
                f"at update grain {update_grain} require {num_states} "
                f"sequence-states, above exhaustive_cap={bounds.exhaustive_cap}"
            )
        for index in range(1, 2**num_states - 1):
            add(tuple((index >> k) & 1 for k in range(num_states)))
    return tuple(tables)
