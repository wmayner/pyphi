"""Macro unit value objects (Marshall et al. 2024, Eq. 11).

A macro unit ``J = (U^J, V^J, tau'_J, g'_J, W^J)`` is specified by its
direct constituents ``V^J`` (micro unit indices or meso ``MacroUnit``
objects), an update grain ``tau'_J`` counted in constituent updates, a
state mapping ``g'_J``, and a background apportionment ``W^J``. The
micro constituents ``U^J`` are derived recursively.

Truth-table indexing convention: the mapping is a flat tuple over the
joint sequence-states of the direct constituents. Within an update the
first constituent varies fastest (little-endian, matching pyphi's state
convention); updates are ordered oldest first, with newer updates
varying slower.

All index arithmetic is mixed-radix, keyed to per-constituent alphabet
tuples. Binary alphabets are enforced by validation at both the micro
and macro level.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


def _mixed_radix_index(digits, radices):
    """Index of mixed-radix ``digits``; the first digit varies fastest."""
    index = 0
    for digit, radix in zip(reversed(digits), reversed(radices), strict=True):
        if not 0 <= digit < radix:
            raise ValueError(f"digit {digit} out of range for radix {radix}")
        index = index * radix + digit
    return index


def _mixed_radix_digits(index, radices):
    """Digits of ``index`` in mixed radix; the first digit varies fastest."""
    digits = []
    for radix in radices:
        digits.append(index % radix)
        index //= radix
    return tuple(digits)


@dataclass(frozen=True)
class MacroUnit:
    """A macro unit ``J = (U^J, V^J, tau'_J, g'_J, W^J)`` (Eq. 11).

    Args:
        constituents: Direct constituents ``V^J`` — micro unit indices
            or meso ``MacroUnit`` objects. Order fixes the truth-table
            digit order.
        update_grain: ``tau'_J`` — constituent updates per unit update.
        mapping: ``g'_J`` as a flat truth table of 0/1 entries over the
            ``prod(alphabets) ** update_grain`` joint sequence-states of
            the constituents (see module docstring for digit order).
        background_apportionment: ``W^J`` — universe indices apportioned
            to this unit.
    """

    constituents: tuple[MacroUnit | int, ...]
    update_grain: int
    mapping: tuple[int, ...]
    background_apportionment: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "constituents", tuple(self.constituents))
        object.__setattr__(self, "mapping", tuple(self.mapping))
        object.__setattr__(
            self,
            "background_apportionment",
            tuple(self.background_apportionment),
        )
        if not self.constituents:
            raise ValueError("a macro unit requires at least one constituent")
        if self.update_grain < 1:
            raise ValueError(f"update grain must be >= 1; got {self.update_grain}")
        micro_sets = []
        grains = set()
        for c in self.constituents:
            if isinstance(c, MacroUnit):
                micro_sets.append(set(c.micro_constituents))
                grains.add(c.micro_grain)
            elif isinstance(c, int) and not isinstance(c, bool):
                if c < 0:
                    raise ValueError(f"negative micro unit index: {c}")
                micro_sets.append({c})
                grains.add(1)
            else:
                raise TypeError(f"constituents must be ints or MacroUnits; got {c!r}")
        union: set[int] = set()
        for s in micro_sets:
            if union & s:
                raise ValueError(
                    "constituents overlap in their micro constituents: "
                    f"{sorted(union & s)}"
                )
            union |= s
        if len(grains) > 1:
            raise ValueError(
                f"constituents must share a single micro grain; got {sorted(grains)}"
            )
        expected = 1
        for size in self.constituent_alphabet_sizes:
            expected *= size
        expected **= self.update_grain
        if len(self.mapping) != expected:
            raise ValueError(
                f"mapping must have {expected} entries for "
                f"{len(self.constituents)} constituents at update grain "
                f"{self.update_grain}; got {len(self.mapping)}"
            )
        if not set(self.mapping) <= {0, 1}:
            raise ValueError("mapping entries must be 0 or 1")
        if 0 not in self.mapping or 1 not in self.mapping:
            raise ValueError("mapping must produce both macro states")
        apportionment = self.background_apportionment
        if len(set(apportionment)) != len(apportionment):
            raise ValueError(f"duplicate background apportionment: {apportionment}")
        if set(apportionment) & union:
            raise ValueError(
                "background apportionment overlaps the unit's micro "
                f"constituents: {sorted(set(apportionment) & union)}"
            )

    @property
    def alphabet_size(self) -> int:
        """Number of unit states (binary)."""
        return 2

    @cached_property
    def constituent_alphabet_sizes(self) -> tuple[int, ...]:
        """Alphabet size of each direct constituent."""
        return tuple(
            c.alphabet_size if isinstance(c, MacroUnit) else 2 for c in self.constituents
        )

    @cached_property
    def micro_constituents(self) -> tuple[int, ...]:
        """``U^J``: the sorted union of micro constituents."""
        out: set[int] = set()
        for c in self.constituents:
            if isinstance(c, MacroUnit):
                out |= set(c.micro_constituents)
            else:
                out.add(c)
        return tuple(sorted(out))

    @cached_property
    def constituent_micro_grain(self) -> int:
        """The common micro grain of the direct constituents."""
        first = self.constituents[0]
        return first.micro_grain if isinstance(first, MacroUnit) else 1

    @cached_property
    def micro_grain(self) -> int:
        """``tau_J``: micro updates spanned by one update of this unit."""
        return self.update_grain * self.constituent_micro_grain
