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

from pyphi import exceptions
from pyphi.data_structures.pyphi_float import PyPhiFloat
from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import UnitVerdict
from pyphi.macro.criteria import _as_unit
from pyphi.macro.criteria import canonical_units
from pyphi.macro.criteria import judge_candidate
from pyphi.macro.system import MacroSystem
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate

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


def _normalized_history(substrate, micro_history, required: int):
    """Validate and shape ``micro_history`` (oldest first).

    A bare state is accepted when ``required == 1``.
    """
    history = tuple(micro_history)
    if history and not isinstance(history[0], (tuple, list)):
        if required != 1:
            raise ValueError(
                f"micro_history must be a sequence of {required} universe "
                "states (oldest first); got a bare state"
            )
        history = (history,)
    history = tuple(tuple(s) for s in history)
    if len(history) != required:
        raise ValueError(
            f"micro_history must have {required} entries (the maximum "
            f"micro grain admitted); got {len(history)}"
        )
    n = substrate.size
    for state in history:
        if len(state) != n or any(v not in (0, 1) for v in state):
            raise ValueError(
                f"each history entry must be a binary universe state of "
                f"length {n}; got {state}"
            )
    return history


def _system_of(substrate, units, micro_history) -> MacroSystem | None:
    """The system of ``units`` over the full universe, or None.

    Returns None when the system's state is unreachable under its own
    TPM: such a system specifies no cause and cannot exist (phi_s = 0).
    """
    units = canonical_units(units)
    needed = max(unit.micro_grain for unit in units)
    window = micro_history[len(micro_history) - needed :]
    try:
        return MacroSystem.from_micro(substrate, units, window)
    except exceptions.StateUnreachableError:
        return None


def _phi(substrate, units, micro_history, memo):
    """Memoized ``(system, phi_s)`` of the system of ``units``."""
    system = _system_of(substrate, units, micro_history)
    if system is None:
        return None, None
    if system not in memo:
        memo[system] = PyPhiFloat(system.sia().phi)
    return system, memo[system]


def _as_constituent(unit: MacroUnit) -> MacroUnit | int:
    """A pool unit as a constituent: identity micro units become bare
    indices, so derived units compare equal to hand-built ones."""
    if (
        len(unit.constituents) == 1
        and not isinstance(unit.constituents[0], MacroUnit)
        and unit.micro_grain == 1
        and unit.mapping == (0, 1)
        and not unit.background_apportionment
    ):
        return unit.constituents[0]
    return unit


def _assemble_systems(pool, background_cap: int):
    """Nonempty unit sets with pairwise-disjoint stakes (Eq. 18).

    Yields tuples in depth-first inclusion order over ``pool``.
    """
    out: list[tuple[MacroUnit, ...]] = []

    def extend(start, partial, claimed, apportioned):
        for k in range(start, len(pool)):
            unit = pool[k]
            stake = set(unit.micro_constituents) | set(unit.background_apportionment)
            if claimed & stake:
                continue
            total = apportioned + len(unit.background_apportionment)
            if total > background_cap:
                continue
            current = (*partial, unit)
            out.append(current)
            extend(k + 1, current, claimed | stake, total)

    extend(0, (), set(), 0)
    return out


def _decompositions(footprint, pool, *, allow_singleton: bool):
    """Sets of pool units with disjoint footprints whose union is
    ``footprint``, all sharing one micro grain."""
    remaining_all = set(footprint)
    candidates = [
        unit for unit in pool if set(unit.micro_constituents) <= remaining_all
    ]
    out: list[tuple[MacroUnit, ...]] = []

    def extend(partial, remaining):
        if not remaining:
            if len(partial) == 1 and not allow_singleton:
                return
            if len({unit.micro_grain for unit in partial}) == 1:
                out.append(tuple(partial))
            return
        first = min(remaining)
        for unit in candidates:
            fp = set(unit.micro_constituents)
            if first in fp and fp <= remaining:
                extend((*partial, unit), remaining - fp)

    extend((), remaining_all)
    return out


def _apportionments(n, footprint, inherited, bounds: SearchBounds):
    """Candidate ``W^J`` sets for a footprint.

    Always contains the union of the constituents' apportionments
    (Eq. 12). Under ENUMERATE, extends it with subsets of the remaining
    background up to ``max_background`` total.
    """
    inherited = tuple(sorted(inherited))
    if bounds.apportionment == "NONE":
        return (inherited,)
    if len(inherited) > bounds.max_background:
        return ()
    available = sorted(set(range(n)) - set(footprint) - set(inherited))
    out = []
    for size in range(bounds.max_background - len(inherited) + 1):
        for extra in itertools.combinations(available, size):
            out.append(tuple(sorted((*inherited, *extra))))
    return tuple(out)


def _f(substrate, V, W, footprint, pool, micro_history, bounds, memo):
    """``f(U^J, W^J)``: evaluated competitor systems (Eq. 16)."""
    fp = set(footprint)
    allowed = set(W)
    members = [
        unit
        for unit in pool
        if set(unit.micro_constituents) < fp
        and set(unit.background_apportionment) <= allowed
    ]
    own = canonical_units(V)
    competitors = []
    for combo in _assemble_systems(members, bounds.max_background):
        if canonical_units(combo) == own:
            continue
        system, phi = _phi(substrate, combo, micro_history, memo)
        if system is None:
            continue
        competitors.append((system, phi))
    return competitors


def _variants(V, W, bounds: SearchBounds):
    """Mapped and grained unit variants of a valid decomposition.

    A single-constituent decomposition is pure grain raising, so its
    variants start at update grain 2 (a grain-1 wrap is the constituent
    relabeled).
    """
    constituents = tuple(_as_constituent(u) for u in canonical_units(V))
    min_grain = 2 if len(V) == 1 else 1
    out = []
    for update_grain in range(min_grain, bounds.max_update_grain + 1):
        for mapping in candidate_mappings(len(V), update_grain, bounds):
            out.append(MacroUnit(constituents, update_grain, mapping, W))
    return out


def _judge(substrate, V, W, footprint, micro_history, bounds, pool, memo):
    _, phi = _phi(substrate, V, micro_history, memo)
    competitors = _f(substrate, V, W, footprint, pool, micro_history, bounds, memo)
    return judge_candidate(0.0 if phi is None else phi, competitors)


def _trivial_verdict(phi) -> UnitVerdict:
    return UnitVerdict(
        valid=True,
        reason=Reason.VALID,
        phi=0.0 if phi is None else float(phi),
        witness=None,
        witness_phi=None,
        num_competitors=0,
    )


def _is_micro(unit: MacroUnit) -> bool:
    """Micro for gating purposes: one micro constituent at grain 1.

    Eqs. 15-16 gate macroing only; micro units are axiomatically valid.
    """
    return len(unit.micro_constituents) == 1 and unit.micro_grain == 1


def _unit_history_requirement(unit: MacroUnit, bounds: SearchBounds) -> int:
    return max(bounds.max_micro_grain, unit.constituent_micro_grain)


@dataclass(frozen=True)
class DecompositionVerdict:
    """A judged candidate decomposition ``(V^J, W^J)``."""

    constituents: tuple[MacroUnit | int, ...]
    background_apportionment: tuple[int, ...]
    verdict: UnitVerdict


def _derive_units(substrate, micro_history, bounds, memo, *, within=None, proper=False):
    """The intrinsic-unit recursion (paper p. 9), bounded by ``bounds``.

    Level 0 is the micro units. Each level derives candidate
    decompositions from the previous level's pool; the competitor set
    draws from the incrementally updated pool, with footprints
    processed smallest-first. Returns ``(pool, verdicts)``.
    """
    n = substrate.size
    indices = tuple(range(n)) if within is None else tuple(sorted(within))
    pool: list[MacroUnit] = [micro_unit(i) for i in indices]
    verdicts: list[DecompositionVerdict] = []
    for unit in pool:
        _, phi = _phi(substrate, (unit,), micro_history, memo)
        verdicts.append(
            DecompositionVerdict(
                constituents=(unit.constituents[0],),
                background_apportionment=(),
                verdict=_trivial_verdict(phi),
            )
        )
    seen: set = set()
    min_size = 1 if bounds.max_update_grain > 1 else 2
    for _level in range(bounds.max_depth):
        pool_prev = tuple(pool)
        emitted_any = False
        max_size = min(len(indices) - (1 if proper else 0), bounds.max_constituents)
        for size in range(min_size, max_size + 1):
            for footprint in itertools.combinations(indices, size):
                new_units: list[MacroUnit] = []
                decompositions = _decompositions(
                    footprint,
                    pool_prev,
                    allow_singleton=bounds.max_update_grain > 1,
                )
                for V in decompositions:
                    inherited = set().union(
                        *(set(u.background_apportionment) for u in V)
                    )
                    for W in _apportionments(n, footprint, inherited, bounds):
                        key = (canonical_units(V), W)
                        if key in seen:
                            continue
                        seen.add(key)
                        verdict = _judge(
                            substrate,
                            V,
                            W,
                            footprint,
                            micro_history,
                            bounds,
                            pool,
                            memo,
                        )
                        verdicts.append(
                            DecompositionVerdict(
                                constituents=tuple(
                                    _as_constituent(u) for u in canonical_units(V)
                                ),
                                background_apportionment=W,
                                verdict=verdict,
                            )
                        )
                        if verdict.valid:
                            new_units.extend(_variants(V, W, bounds))
                pool.extend(new_units)
                emitted_any = emitted_any or bool(new_units)
        if not emitted_any:
            break
    return tuple(pool), tuple(verdicts)


def _f_for_unit(substrate, unit, V, micro_history, bounds, memo):
    pool, _ = _derive_units(
        substrate,
        micro_history,
        bounds,
        memo,
        within=unit.micro_constituents,
        proper=True,
    )
    return _f(
        substrate,
        V,
        unit.background_apportionment,
        unit.micro_constituents,
        pool,
        micro_history,
        bounds,
        memo,
    )


def competing_systems(
    substrate: Substrate,
    unit: MacroUnit,
    micro_history,
    bounds: SearchBounds = SearchBounds(),
) -> tuple[MacroSystem, ...]:
    """``f(U^J, W^J)`` materialized within the unit's footprint (Eq. 16)."""
    history = _normalized_history(
        substrate, micro_history, _unit_history_requirement(unit, bounds)
    )
    if _is_micro(unit):
        return ()
    memo: dict[MacroSystem, PyPhiFloat] = {}
    V = canonical_units(_as_unit(c) for c in unit.constituents)
    return tuple(
        system for system, _ in _f_for_unit(substrate, unit, V, history, bounds, memo)
    )


def is_intrinsic_unit(
    substrate: Substrate,
    unit: MacroUnit,
    micro_history,
    bounds: SearchBounds = SearchBounds(),
) -> UnitVerdict:
    """Eqs. 15-16 for one candidate; micro units return VALID trivially.

    The unit's own mapping and update grain are ignored (Eq. 15 is
    mapping-independent); the recursion is run restricted to the unit's
    footprint to build ``f(U^J, W^J)``.
    """
    history = _normalized_history(
        substrate, micro_history, _unit_history_requirement(unit, bounds)
    )
    memo: dict[MacroSystem, PyPhiFloat] = {}
    if _is_micro(unit):
        _, phi = _phi(substrate, (unit,), history, memo)
        return _trivial_verdict(phi)
    V = canonical_units(_as_unit(c) for c in unit.constituents)
    _, phi = _phi(substrate, V, history, memo)
    competitors = _f_for_unit(substrate, unit, V, history, bounds, memo)
    return judge_candidate(0.0 if phi is None else phi, competitors)


@dataclass(frozen=True)
class IntrinsicUnitsResult:
    """The recursion's output: the valid-unit pool and every verdict.

    Attributes:
        units: All derived intrinsic units, micro units included, in
            derivation order (footprints smallest-first per level).
        verdicts: One :class:`DecompositionVerdict` per judged
            ``(V^J, W^J)`` candidate, micro units included.
    """

    units: tuple[MacroUnit, ...]
    verdicts: tuple[DecompositionVerdict, ...]

    def units_by_footprint(self) -> dict[tuple[int, ...], tuple[MacroUnit, ...]]:
        """The unit pool grouped by micro footprint."""
        grouped: dict[tuple[int, ...], list[MacroUnit]] = {}
        for unit in self.units:
            grouped.setdefault(unit.micro_constituents, []).append(unit)
        return {k: tuple(v) for k, v in grouped.items()}


def intrinsic_units(
    substrate: Substrate, micro_history, bounds: SearchBounds
) -> IntrinsicUnitsResult:
    """The recursion's fixed point: the valid-unit pool plus all verdicts."""
    history = _normalized_history(substrate, micro_history, bounds.max_micro_grain)
    memo: dict[MacroSystem, PyPhiFloat] = {}
    units, verdicts = _derive_units(substrate, history, bounds, memo)
    return IntrinsicUnitsResult(units=units, verdicts=verdicts)


def valid_systems(
    substrate: Substrate, micro_history, bounds: SearchBounds
) -> tuple[MacroSystem, ...]:
    """The bounded ``P(u)``: every Eq-18-compatible system of intrinsic
    units, evaluated over the full universe with everything else as
    background. Systems whose state is unreachable are dropped."""
    history = _normalized_history(substrate, micro_history, bounds.max_micro_grain)
    memo: dict[MacroSystem, PyPhiFloat] = {}
    units, _ = _derive_units(substrate, history, bounds, memo)
    systems = []
    for combo in _assemble_systems(list(units), bounds.max_background):
        system = _system_of(substrate, combo, history)
        if system is not None:
            systems.append(system)
    return tuple(systems)
