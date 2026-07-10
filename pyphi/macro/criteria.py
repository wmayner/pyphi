"""Intrinsic-unit criteria (Marshall et al. 2024, Eqs. 15-16).

A candidate macro unit J with direct constituents ``V^J`` exists as one
unit only if its constituent system -- the system of the elements of
``V^J`` over the full universe, with everything else as background --
is integrated (Eq. 15) and strictly more irreducible than every
competing system that can be built within the unit's footprint
(Eq. 16). Both criteria are properties of the pair ``(V^J, W^J)``: the
candidate's own mapping and update grain do not enter, so mapped and
grained variants of one decomposition share a verdict.

This module holds the pure criteria logic. The competitor set
``f(U^J, W^J)`` is materialized by :mod:`pyphi.macro.search`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from pyphi import exceptions
from pyphi import numerics
from pyphi.macro.system import MacroSystem
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate


class Reason(Enum):
    """Why a candidate unit is valid or invalid."""

    VALID = "VALID"
    NOT_INTEGRATED = "NOT_INTEGRATED"
    NOT_MAXIMAL = "NOT_MAXIMAL"
    TIED = "TIED"


@dataclass(frozen=True)
class UnitVerdict:
    """The outcome of checking Eqs. 15-16 for one candidate decomposition.

    Attributes
    ----------
    valid : bool
        Whether the candidate satisfies both criteria.
    reason : Reason
        ``VALID``, or which criterion failed: ``NOT_INTEGRATED``
        (Eq. 15), ``NOT_MAXIMAL`` or ``TIED`` (Eq. 16).
    phi : float
        φₛ(v^J), the constituent system's integrated information.
    witness : MacroSystem or None
        The competitor that beat or tied the candidate, if any.
    witness_phi : float or None
        The witness's φₛ.
    num_competitors : int
        Size of the competitor set ``f(U^J, W^J)``.
    """

    valid: bool
    reason: Reason
    phi: float
    witness: MacroSystem | None
    witness_phi: float | None
    num_competitors: int


def judge_candidate(
    phi: float, competitors: Iterable[tuple[MacroSystem, float]]
) -> UnitVerdict:
    """Eqs. 15-16 given φₛ(v^J) and the evaluated competitor set.

    All inequalities are strict at ``config.numerics.precision``; a
    candidate that ties its strongest competitor is invalid with reason
    ``TIED``.

    Parameters
    ----------
    phi : float
        The candidate's φₛ(v^J).
    competitors : Iterable[tuple[MacroSystem, float]]
        ``(system, φₛ)`` pairs for ``f(U^J, W^J)``.
    """
    competitors = tuple(competitors)
    if not numerics.is_positive(phi):
        return UnitVerdict(
            valid=False,
            reason=Reason.NOT_INTEGRATED,
            phi=phi,
            witness=None,
            witness_phi=None,
            num_competitors=len(competitors),
        )
    best_system: MacroSystem | None = None
    best_phi = float("-inf")
    for system, competitor_phi in competitors:
        if best_system is None or competitor_phi > best_phi:
            best_system = system
            best_phi = competitor_phi
    if best_system is not None:
        if numerics.eq(phi, best_phi):
            return UnitVerdict(
                valid=False,
                reason=Reason.TIED,
                phi=phi,
                witness=best_system,
                witness_phi=best_phi,
                num_competitors=len(competitors),
            )
        if best_phi > phi:
            return UnitVerdict(
                valid=False,
                reason=Reason.NOT_MAXIMAL,
                phi=phi,
                witness=best_system,
                witness_phi=best_phi,
                num_competitors=len(competitors),
            )
    return UnitVerdict(
        valid=True,
        reason=Reason.VALID,
        phi=phi,
        witness=None,
        witness_phi=None,
        num_competitors=len(competitors),
    )


def _as_unit(constituent: MacroUnit | int) -> MacroUnit:
    """A constituent as a unit: micro indices become identity units."""
    if isinstance(constituent, MacroUnit):
        return constituent
    return micro_unit(constituent)


def canonical_units(units: Iterable[MacroUnit]) -> tuple[MacroUnit, ...]:
    """The units of a system in canonical order.

    Sorting makes systems that differ only in unit order compare and
    hash equal, so memoized evaluations are shared.
    """
    return tuple(
        sorted(
            units,
            key=lambda unit: (
                unit.micro_constituents,
                unit.micro_grain,
                unit.mapping,
                unit.background_apportionment,
            ),
        )
    )


def constituent_system(
    substrate: Substrate,
    constituents: Iterable[MacroUnit | int],
    micro_history,
) -> MacroSystem:
    """The system of a unit's direct constituents (Eq. 15).

    Each element of ``V^J`` participates with its full definition: a
    micro index becomes an identity micro unit; a meso constituent
    keeps its mapping, grain, and apportionment. The system spans the
    full universe, with all remaining micro units as background.

    ``micro_history`` (oldest first; a bare state is accepted when the
    constituents have micro grain 1) may be longer than the
    constituents require; only the trailing window is used.
    """
    units = canonical_units(_as_unit(c) for c in constituents)
    history = tuple(micro_history)
    if history and not isinstance(history[0], (tuple, list)):
        history = (history,)
    history = tuple(tuple(s) for s in history)
    needed = max(unit.micro_grain for unit in units)
    if len(history) < needed:
        raise ValueError(
            f"micro_history must have at least {needed} entries for "
            f"these constituents; got {len(history)}"
        )
    return MacroSystem.from_micro(substrate, units, history[len(history) - needed :])


def unit_integration(
    substrate: Substrate,
    constituents: Iterable[MacroUnit | int],
    micro_history,
) -> float:
    """φₛ(v^J): the constituent system's integrated information (Eq. 15).

    A constituent system whose state is unreachable specifies no cause
    and cannot exist; its integration is zero.
    """
    try:
        system = constituent_system(substrate, constituents, micro_history)
    except exceptions.StateUnreachableError:
        return 0.0
    return float(system.sia().phi)
