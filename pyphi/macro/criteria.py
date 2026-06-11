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

from pyphi import utils
from pyphi.macro.system import MacroSystem


class Reason(Enum):
    """Why a candidate unit is valid or invalid."""

    VALID = "VALID"
    NOT_INTEGRATED = "NOT_INTEGRATED"
    NOT_MAXIMAL = "NOT_MAXIMAL"
    TIED = "TIED"


@dataclass(frozen=True)
class UnitVerdict:
    """The outcome of checking Eqs. 15-16 for one candidate decomposition.

    Attributes:
        valid: Whether the candidate satisfies both criteria.
        reason: ``VALID``, or which criterion failed: ``NOT_INTEGRATED``
            (Eq. 15), ``NOT_MAXIMAL`` or ``TIED`` (Eq. 16).
        phi: ``phi_s(v^J)``, the constituent system's integrated
            information.
        witness: The competitor that beat or tied the candidate, if any.
        witness_phi: The witness's ``phi_s``.
        num_competitors: Size of the competitor set ``f(U^J, W^J)``.
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
    """Eqs. 15-16 given ``phi_s(v^J)`` and the evaluated competitor set.

    All inequalities are strict at ``config.numerics.precision``; a
    candidate that ties its strongest competitor is invalid with reason
    ``TIED``.

    Args:
        phi: The candidate's ``phi_s(v^J)``.
        competitors: ``(system, phi_s)`` pairs for ``f(U^J, W^J)``.
    """
    competitors = tuple(competitors)
    if not utils.is_positive(phi):
        return UnitVerdict(
            valid=False,
            reason=Reason.NOT_INTEGRATED,
            phi=float(phi),
            witness=None,
            witness_phi=None,
            num_competitors=len(competitors),
        )
    best_system: MacroSystem | None = None
    best_phi = float("-inf")
    for system, competitor_phi in competitors:
        if best_system is None or float(competitor_phi) > best_phi:
            best_system = system
            best_phi = float(competitor_phi)
    if best_system is not None:
        if utils.eq(phi, best_phi):
            return UnitVerdict(
                valid=False,
                reason=Reason.TIED,
                phi=float(phi),
                witness=best_system,
                witness_phi=best_phi,
                num_competitors=len(competitors),
            )
        if best_phi > float(phi):
            return UnitVerdict(
                valid=False,
                reason=Reason.NOT_MAXIMAL,
                phi=float(phi),
                witness=best_system,
                witness_phi=best_phi,
                num_competitors=len(competitors),
            )
    return UnitVerdict(
        valid=True,
        reason=Reason.VALID,
        phi=float(phi),
        witness=None,
        witness_phi=None,
        num_competitors=len(competitors),
    )
