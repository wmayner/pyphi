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

from dataclasses import dataclass

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
