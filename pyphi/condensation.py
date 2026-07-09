# condensation.py
"""Condensation of candidate systems into complexes.

Implements the recursive exclusion cascade (Marshall, Albantakis, Tononi
2023, Algorithm A1; Albantakis et al. 2023, Exclusion): walk candidates in
descending φₛ tiers, accept each tier's overlap-clique winners, and drop
candidates that overlap an accepted complex. Ties within a clique escalate
to Composition (big Φ) per the S1 tie-resolution supplement; a clique whose
Φ also ties fails exclusion — its members are removed, but their units stay
available to lower-φ candidates in later tiers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _sia_node_indices(sia: Any) -> tuple[int, ...] | None:
    """Return the candidate-system node indices of a SIA, across formalisms.

    IIT 3.0 SIAs carry a ``System`` reference under ``.system``; IIT 4.0
    SIAs expose ``.node_indices`` directly.
    """
    system = getattr(sia, "system", None)
    if system is not None:
        return system.node_indices
    return getattr(sia, "node_indices", None)


def _exclusion_records(
    accepted: list[Any], sorted_sias: list[Any]
) -> dict[tuple[int, ...], tuple[Any, ...]]:
    """Map each accepted complex (by units) to the ExcludedCandidate records
    it excluded: every irreducible candidate that overlaps it and was not
    itself accepted.

    A candidate that overlaps several accepted complexes appears in each of
    their exclusion sets. Reads only values the cascade already computed.
    """
    from pyphi.models.complex import ExcludedCandidate

    accepted_indices = {tuple(_sia_node_indices(s) or ()) for s in accepted}
    records: dict[tuple[int, ...], tuple[Any, ...]] = {}
    for acc in accepted:
        acc_idx = tuple(_sia_node_indices(acc) or ())
        acc_set = set(acc_idx)
        recs = []
        for cand in sorted_sias:
            cand_idx = tuple(_sia_node_indices(cand) or ())
            if cand_idx == acc_idx or cand_idx in accepted_indices:
                continue
            if acc_set & set(cand_idx):
                recs.append(ExcludedCandidate(cand_idx, float(cand.phi)))
        records[acc_idx] = tuple(recs)
    return records


def _config_iit_version() -> str:
    from pyphi.conf import config as _config

    return _config.formalism.iit.version


def _accept(sia: Any, result: list[Any], covered: set[int]) -> None:
    """Add a SIA to the accepted-complex result list and mark its units as covered."""
    indices = _sia_node_indices(sia)
    if indices is None:
        return
    result.append(sia)
    covered.update(indices)


def _phi_groups(sorted_sias: list[Any]) -> Iterable[list[Any]]:
    """Yield contiguous groups of SIAs sharing the same φₛ value
    (precision-aware), assuming the input is sorted by ``.order_by()``
    descending."""
    from pyphi import utils as _utils

    i = 0
    while i < len(sorted_sias):
        tier_phi = float(sorted_sias[i].phi)
        j = i + 1
        while j < len(sorted_sias) and _utils.eq(float(sorted_sias[j].phi), tier_phi):
            j += 1
        yield sorted_sias[i:j]
        i = j


def _find_overlap_cliques(sias: list[Any]) -> list[list[Any]]:
    """Group SIAs into connected components by unit overlap."""
    n = len(sias)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    units = [set(_sia_node_indices(sia) or ()) for sia in sias]
    for i in range(n):
        for j in range(i + 1, n):
            if units[i] & units[j]:
                union(i, j)

    groups: dict[int, list[Any]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(sias[i])
    return list(groups.values())


def _big_phi_of_sia(sia: Any, substrate: Any, state: tuple[int, ...]) -> float:
    """Compute the structure-integrated information Φ of the SIA's
    candidate system. Builds the system from substrate + state + the
    SIA's units and invokes the active formalism's cause-effect-structure
    computation.
    """
    from pyphi.system import System

    indices = _sia_node_indices(sia)
    if indices is None:
        return 0.0
    system = System.from_substrate(substrate, state, indices)
    return float(system.ces().big_phi)


def _resolve_clique_by_big_phi(
    clique: list[Any], substrate: Any, state: tuple[int, ...]
) -> Any | None:
    """Pick the Φ-maximal candidate in an overlap clique via the
    substrate-exclusion cascade (Composition escalation). Returns ``None``
    when Φ ties — the exclusion postulate is violated for that
    clique and none of its candidates qualify as a complex.
    """
    from dataclasses import dataclass

    from pyphi import resolve_ties

    @dataclass(frozen=True)
    class _CandidateProxy:
        sia: Any
        big_phi: float

    proxies = [
        _CandidateProxy(sia=sia, big_phi=_big_phi_of_sia(sia, substrate, state))
        for sia in clique
    ]
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Composition")
    outcome = resolve_ties.resolve_complex_tie(proxies, context=ctx)
    if outcome.outcome == "RESOLVED" and outcome.resolved is not None:
        return outcome.resolved.sia
    return None


def _substrate_exclusion_cascade(
    sorted_sias: list[Any],
    substrate: Any,
    state: tuple[int, ...],
) -> list[Any]:
    """Walk SIAs in descending φₛ tiers, applying the S1
    substrate-exclusion cascade within each tier."""
    result: list[Any] = []
    covered: set[int] = set()

    for tier in _phi_groups(sorted_sias):
        # Within this tier, discard candidates whose units overlap any
        # already-accepted complex.
        survivors = [
            sia for sia in tier if not (set(_sia_node_indices(sia) or ()) & covered)
        ]
        if not survivors:
            continue
        for clique in _find_overlap_cliques(survivors):
            if len(clique) == 1:
                _accept(clique[0], result, covered)
                continue
            winner = _resolve_clique_by_big_phi(clique, substrate, state)
            if winner is not None:
                _accept(winner, result, covered)
    return result


def _resolve_clique_iit3(clique: list[Any]) -> Any | None:
    """Return the unique complex from an IIT 3.0 overlap clique, or None
    when the clique is indeterminate.

    Single-candidate cliques resolve trivially; multi-candidate cliques
    always flag ``UNRESOLVED_WITHIN_BUDGET`` because IIT 3.0 has no
    paper-canonical escalation level. The caller treats None as
    exclusion-postulate failure for the clique.
    """
    from pyphi import resolve_ties

    if len(clique) == 1:
        return clique[0]
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Exclusion")
    outcome = resolve_ties.resolve_iit3_complex_tie(clique, context=ctx)
    if outcome.outcome == "RESOLVED" and outcome.resolved is not None:
        return outcome.resolved
    return None


def _iit3_exclusion_cascade(
    sorted_sias: list[Any],
    substrate: Any,  # noqa: ARG001 — kept for parity with iit4 cascade signature
    state: Any,  # noqa: ARG001 — kept for parity with iit4 cascade signature
) -> list[Any]:
    """Walk SIAs in descending Φ tiers, applying the IIT 3.0
    cross-subsystem cascade within each overlap clique.

    Within a tier, drop candidates whose units overlap an already-
    accepted complex, then group survivors into overlap cliques.
    Each clique with one member is accepted directly; cliques with
    multiple members run through ``_resolve_clique_iit3`` and are
    skipped when indeterminate.
    """
    result: list[Any] = []
    covered: set[int] = set()
    for tier in _phi_groups(sorted_sias):
        survivors = [
            sia for sia in tier if not (set(_sia_node_indices(sia) or ()) & covered)
        ]
        if not survivors:
            continue
        for clique in _find_overlap_cliques(survivors):
            winner = _resolve_clique_iit3(clique)
            if winner is not None:
                _accept(winner, result, covered)
    return result
