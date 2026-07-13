# condensation.py
"""Condensation of candidate systems into complexes.

Implements the recursive exclusion cascade (Marshall et al. 2023,
Algorithm A1; Albantakis et al. 2023, Exclusion): walk candidates in
descending φₛ tiers, accept each tier's overlap-clique winners, and drop
candidates that overlap an accepted complex. Ties within a clique escalate
to Composition (big Φ) per the S1 tie-resolution supplement; a clique whose
Φ also ties fails exclusion — its members are removed, but their units stay
available to lower-φ candidates in later tiers.

Overlap is assessed in micro units: each candidate carries a ``footprint``
of micro indices, so candidate systems of micro units and of macro units
compete in the same cascade.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Candidate:
    """A candidate system as the exclusion cascade sees it.

    ``footprint`` holds the candidate's micro units — the currency in which
    overlap is assessed (a macro system's own analysis indexes macro units
    of a synthetic substrate, so its indices cannot be compared with other
    candidates' directly). The providers defer materialization: the SIA is
    needed only for accepted complexes, the system only when a tie escalates
    to Composition.

    Attributes
    ----------
    footprint : frozenset[int]
        The candidate's micro units.
    phi : float
        The candidate's φₛ.
    sia_provider : Callable
        Zero-argument callable returning the candidate's system
        irreducibility analysis.
    system_provider : Callable
        Zero-argument callable returning the candidate's system, used to
        compute Φ when a φₛ tie escalates to Composition.
    units : tuple or None
        The candidate's macro unit structure; ``None`` for a system of
        micro units.
    """

    footprint: frozenset[int]
    phi: float
    sia_provider: Callable[[], Any]
    system_provider: Callable[[], Any]
    units: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class CondensationOutcome:
    """The cascade's result.

    Attributes
    ----------
    accepted : tuple[Candidate, ...]
        The complexes, in descending-φₛ acceptance order.
    failed_cliques : tuple[tuple[Candidate, ...], ...]
        Cliques of overlapping candidates that tied at φₛ and still tied
        at Φ, failing the exclusion postulate; none of their members is a
        complex.
    """

    accepted: tuple[Candidate, ...]
    failed_cliques: tuple[tuple[Candidate, ...], ...]


def _config_iit_version() -> str:
    from pyphi.conf import config as _config

    return _config.formalism.iit.version


def _phi_tiers(candidates: Sequence[Candidate]):
    """Yield φₛ tiers in descending order.

    Tier membership is tolerant: a candidate joins the tier when its φₛ
    equals the tier head's up to ``config.numerics.precision``. Within a
    tier, candidates keep their input order, so the caller's dispatch
    order — not floating-point noise — breaks presentation ties.
    """
    from pyphi import numerics as _numerics

    # The sort only orders candidates; tolerant tier membership is decided by
    # numerics.eq in the loop below.
    # numerics: exact — ordering only; tier ties resolved by numerics.eq.
    indexed = sorted(enumerate(candidates), key=lambda pair: -pair[1].phi)
    i = 0
    while i < len(indexed):
        tier_phi = indexed[i][1].phi
        j = i + 1
        while j < len(indexed) and _numerics.eq(indexed[j][1].phi, tier_phi):
            j += 1
        tier = sorted(indexed[i:j], key=lambda pair: pair[0])
        yield [candidate for _, candidate in tier]
        i = j


def _find_overlap_cliques(candidates: list[Candidate]) -> list[list[Candidate]]:
    """Group candidates into connected components by footprint overlap."""
    n = len(candidates)
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

    for i in range(n):
        for j in range(i + 1, n):
            if candidates[i].footprint & candidates[j].footprint:
                union(i, j)

    groups: dict[int, list[Candidate]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(candidates[i])
    return list(groups.values())


def _fingerprint_key(system: Any):
    """Content digest used to deduplicate Φ computations; systems without
    one never deduplicate."""
    key = getattr(system, "_fingerprint", None)
    return key if key is not None else object()


def _resolve_phi_tied_group(
    survivors: list[Candidate],
) -> tuple[list[Candidate], list[tuple[Candidate, ...]]]:
    """Recursive exclusion among the φₛ-tied survivors of one tier.

    Per Marshall et al. 2023 (Algorithm A1, tied branch) and the Albantakis
    et al. 2023 S1 tie supplement: a tied candidate that overlaps no other
    tied candidate is a complex outright. Among candidates that do overlap
    another tied candidate, Φ (Composition) escalates the tie: the Φ-maximal
    candidates win — unless two Φ-maximal candidates overlap *each other*,
    in which case that clique fails the exclusion postulate and none of its
    members is a complex. Accepting a winner removes only the candidates
    overlapping it; everything else (including candidates that overlapped
    only excluded or failed rivals) re-enters the walk.

    Φ is computed once per distinct system content fingerprint: equal
    fingerprints denote byte-identical kernel computations, so their Φ
    values tie exactly, and a conflicted group whose members all share one
    fingerprint fails exclusion with no cause-effect-structure computation
    at all.
    """
    from pyphi import numerics as _numerics

    accepted: list[Candidate] = []
    failed: list[tuple[Candidate, ...]] = []
    remaining = list(survivors)
    while remaining:
        conflicted_ids = {
            id(c)
            for c in remaining
            if any(
                c.footprint & other.footprint for other in remaining if other is not c
            )
        }
        accepted.extend(c for c in remaining if id(c) not in conflicted_ids)
        remaining = [c for c in remaining if id(c) in conflicted_ids]
        if not remaining:
            break

        systems = [candidate.system_provider() for candidate in remaining]
        keys = [_fingerprint_key(system) for system in systems]
        if len(set(keys)) == 1:
            tier = list(remaining)
        else:
            big_phis: dict[Any, float] = {}
            for system, key in zip(systems, keys, strict=True):
                if key not in big_phis:
                    big_phis[key] = float(system.ces().big_phi)
            values = [big_phis[key] for key in keys]
            best = max(values)
            # numerics: tolerant — Φ-tier membership is a selection.
            tier = [
                c
                for c, value in zip(remaining, values, strict=True)
                if _numerics.eq(value, best)
            ]

        tier_ids = {id(c) for c in tier}
        conflicted_tier = [
            c
            for c in tier
            if any(c.footprint & other.footprint for other in tier if other is not c)
        ]
        conflicted_tier_ids = {id(c) for c in conflicted_tier}
        winners = [c for c in tier if id(c) not in conflicted_tier_ids]
        failed.extend(tuple(clique) for clique in _find_overlap_cliques(conflicted_tier))
        accepted.extend(winners)
        remaining = [
            c
            for c in remaining
            if id(c) not in tier_ids
            and not any(c.footprint & winner.footprint for winner in winners)
        ]
    return accepted, failed


def exclusion_cascade(candidates: Sequence[Candidate]) -> CondensationOutcome:
    """Condense candidates into complexes by the recursive exclusion cascade.

    ``candidates`` may arrive in any order; they are grouped into φₛ tiers
    (descending) with tolerant membership. Within each tier, candidates
    overlapping an accepted complex are dropped, and the φₛ-tied survivors
    are resolved recursively (see :func:`_resolve_phi_tied_group`): tied
    candidates with no overlap conflict are complexes; overlap conflicts
    escalate to Composition (Φ); Φ-tied *overlapping* candidates fail
    exclusion — their units stay available to lower-φₛ candidates in later
    tiers. Within-tier presentation order follows the input order.
    """
    accepted: list[Candidate] = []
    covered: set[int] = set()
    failed: list[tuple[Candidate, ...]] = []
    for tier in _phi_tiers(candidates):
        survivors = [c for c in tier if not (c.footprint & covered)]
        if not survivors:
            continue
        tier_accepted, tier_failed = _resolve_phi_tied_group(survivors)
        position = {id(c): i for i, c in enumerate(survivors)}
        tier_accepted.sort(key=lambda c: position[id(c)])
        accepted.extend(tier_accepted)
        failed.extend(tier_failed)
        for complex_candidate in tier_accepted:
            covered |= complex_candidate.footprint
    return CondensationOutcome(tuple(accepted), tuple(failed))


def _resolve_clique_iit3(clique: list[Candidate]) -> Candidate | None:
    """Return the unique complex from an IIT 3.0 overlap clique, or ``None``
    when the clique is indeterminate.

    Single-candidate cliques resolve trivially; multi-candidate cliques
    always flag ``UNRESOLVED_WITHIN_BUDGET`` because IIT 3.0 has no
    paper-canonical escalation level. The caller treats ``None`` as
    exclusion-postulate failure for the clique.
    """
    from pyphi import resolve_ties

    if len(clique) == 1:
        return clique[0]
    sias = [candidate.sia_provider() for candidate in clique]
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Exclusion")
    outcome = resolve_ties.resolve_iit3_complex_tie(sias, context=ctx)
    if outcome.outcome == "RESOLVED" and outcome.resolved is not None:
        for candidate, sia in zip(clique, sias, strict=True):
            if sia is outcome.resolved:
                return candidate
    return None


def iit3_exclusion_cascade(candidates: Sequence[Candidate]) -> CondensationOutcome:
    """Condense candidates under IIT 3.0: the recursive tier walk with no
    Composition escalation.

    ``candidates`` may arrive in any order; they are grouped into φₛ tiers
    (descending) with tolerant membership. Within a tier, candidates
    overlapping an accepted complex are dropped and survivors group into
    overlap cliques; a clique with one member is accepted directly, and a
    multi-member clique is indeterminate (IIT 3.0 has no paper-canonical
    system-level tie-break) — it fails exclusion and the walk continues.
    Within-tier presentation order follows the input order.
    """
    accepted: list[Candidate] = []
    covered: set[int] = set()
    failed: list[tuple[Candidate, ...]] = []
    for tier in _phi_tiers(candidates):
        survivors = [c for c in tier if not (c.footprint & covered)]
        if not survivors:
            continue
        for clique in _find_overlap_cliques(survivors):
            winner = _resolve_clique_iit3(clique)
            if winner is None:
                failed.append(tuple(clique))
                continue
            accepted.append(winner)
            covered |= winner.footprint
    return CondensationOutcome(tuple(accepted), tuple(failed))


def exclusion_records(
    accepted: Sequence[Candidate], candidates: Sequence[Candidate]
) -> dict[tuple[int, ...], tuple[Any, ...]]:
    """Map each accepted complex (by sorted footprint) to the
    ExcludedCandidate records it excluded: every candidate that overlaps it
    and was not itself accepted.

    A candidate that overlaps several accepted complexes appears in each of
    their exclusion sets. An excluded candidate may carry higher φₛ than a
    complex whose record it appears in, when it was carved away by a
    different overlapping complex. Accepted candidates are identified by
    object identity, not footprint, so a losing candidate that shares an
    accepted complex's exact footprint — a rival grain over the same micro
    units — is recorded. Reads only values the cascade already computed.
    """
    from pyphi.models.complex import ExcludedCandidate

    accepted_ids = {id(c) for c in accepted}
    records: dict[tuple[int, ...], tuple[Any, ...]] = {}
    for acc in accepted:
        recs = tuple(
            ExcludedCandidate(tuple(sorted(cand.footprint)), cand.phi, units=cand.units)
            for cand in candidates
            if id(cand) not in accepted_ids and acc.footprint & cand.footprint
        )
        records[tuple(sorted(acc.footprint))] = recs
    return records
