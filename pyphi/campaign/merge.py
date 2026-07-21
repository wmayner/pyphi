"""Tie-preserving merges of shard outputs.

Every merge re-runs the same :mod:`pyphi.resolve_ties` machinery the
single-machine code path uses, applied to the union of shard tie sets.
Exactness: the global extremum is attained inside some shard, so any
candidate within tolerance of the global extremum is within tolerance of
its own shard's extremum and therefore present in that shard's tie set —
the union loses nothing. Candidates are restored to global enumeration
order before resolution, so the selected representative is identical to a
full sweep's.
"""

from __future__ import annotations

from typing import Any

from pyphi import resolve_ties
from pyphi.direction import Direction
from pyphi.models.mice import MaximallyIrreducibleCause
from pyphi.models.mice import MaximallyIrreducibleEffect

__all__ = [
    "build_distinction",
    "merge_purview_rias",
    "merge_sia_strides",
    "merge_stride_rias",
]


def _pin_key(ria: Any) -> str:
    spec = ria.specified_state
    return repr(spec.state) if spec is not None else "none"


def _merged_partition_margin(winner: Any, shard_winners: list[Any]) -> Any:
    """Global margin from per-shard winners, or None when underivable.

    The global runner-up normalized φ is the smaller of: the best
    normalized φ among non-winning shards, and the winning shard's own
    runner-up (its winner's normalized φ plus its margin). Underivable
    (None) when any shard's margin is None — a short-circuited or
    single-candidate slice.
    """
    if any(getattr(s, "partition_margin", None) is None for s in shard_winners):
        return None
    if winner.normalized_phi is None:
        return None
    winner_nphi = float(winner.normalized_phi)
    rivals = []
    for shard in shard_winners:
        nphi = float(shard.normalized_phi)
        if nphi == winner_nphi:
            # The winning value's shard: its runner-up is the rival.
            rivals.append(nphi + float(shard.partition_margin))
        else:
            rivals.append(nphi)
    # numerics: exact — reported margin, not a selection.
    return max(0.0, min(rivals) - winner_nphi) if rivals else None


def merge_stride_rias(entries: list[tuple[Any, dict]]) -> Any:
    """Merge the stride winners of one (mechanism, direction, purview).

    ``entries`` pairs each stride's winning RIA with its aux record, whose
    ``tie_indices`` map each specified-state pin (by ``repr`` of its
    state) to the global enumeration indices of that pin's partition-tie
    members, in tie-set order.
    """
    per_pin: dict[str, list[tuple[int, Any]]] = {}
    shard_best: dict[str, list[Any]] = {}
    for ria, aux in entries:
        for pin in getattr(ria, "_state_ties", None) or (ria,):
            key = _pin_key(pin)
            ties = getattr(pin, "_partition_ties", None) or (pin,)
            indices = aux["tie_indices"][key]
            per_pin.setdefault(key, []).extend(zip(indices, ties, strict=True))
            shard_best.setdefault(key, []).append(pin)
    pin_winners = []
    for key, indexed in per_pin.items():
        indexed.sort(key=lambda pair: pair[0])
        candidates = [c for _, c in indexed]
        ties = tuple(resolve_ties.partitions(candidates))
        winner = ties[0]
        for tie in ties:
            tie.set_partition_ties(ties)
        winner.partition_margin = _merged_partition_margin(winner, shard_best[key])
        pin_winners.append(winner)
    state_ties = tuple(resolve_ties.states(pin_winners))
    for tie in state_ties:
        tie.set_state_ties(state_ties)
    return state_ties[0]


def merge_purview_rias(
    direction: Direction, rias: list, canonical_purviews: list
) -> Any:
    """Merge per-purview RIAs into the MICE (mirrors ``find_mice``'s tail)."""
    mice_cls = (
        MaximallyIrreducibleCause
        if direction == Direction.CAUSE
        else MaximallyIrreducibleEffect
    )
    order = {tuple(p): i for i, p in enumerate(canonical_purviews)}
    rias = sorted(rias, key=lambda ria: order[tuple(ria.purview)])
    all_mice = [mice_cls(ria) for ria in rias]
    ties = tuple(resolve_ties.purviews(all_mice))
    for tie in ties:
        tie.set_purview_ties(ties)
    winner = ties[0]
    others = [m for m in all_mice if m is not winner]
    if others:
        # numerics: exact — reported margin, not a selection.
        best_rival = max(float(m.phi) for m in others)
        winner.purview_margin = max(0.0, float(winner.phi) - best_rival)
    return winner


def build_distinction(mechanism: Any, mic: Any, mie: Any) -> Any:
    """Assemble a distinction from a merged MIC and MIE."""
    from pyphi.models import Concept

    return Concept(mechanism=tuple(mechanism), cause=mic, effect=mie)


def merge_sia_strides(entries: list[tuple[Any, dict]]) -> Any:
    """Merge SIA stride winners (union of tie sets, global order restored)."""
    indexed: list[tuple[int, Any]] = []
    for sia, aux in entries:
        ties = getattr(sia, "ties", None) or (sia,)
        indexed.extend(zip(aux["tie_indices"], ties, strict=True))
    indexed.sort(key=lambda pair: pair[0])
    candidates = [c for _, c in indexed]
    ties = tuple(resolve_ties.sias(candidates))
    winner = ties[0]
    for tie in ties:
        tie.set_ties(list(ties))
    return winner
