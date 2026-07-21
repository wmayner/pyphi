"""Tie-preserving merges of shard outputs.

Every merge re-runs the same :mod:`pyphi.resolve_ties` machinery the
single-machine code path uses, applied to per-shard winners (and, where
serialization carries them, their tie sets) restored to global enumeration
order. Exactness: the global extremum is attained inside some shard, and a
candidate that would win the full sweep is necessarily its own shard's
winner — strides preserve enumeration order within each shard — so
resolving over the ordered shard winners selects the identical
representative a full sweep selects.
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
    ``pin_winner_indices`` map each specified-state pin (by ``repr`` of
    its state) to the global enumeration index of that pin's winning
    partition.

    Per pin, the candidates are the per-shard pin winners restored to
    global enumeration order. This selects the identical winner a full
    sweep would: the full sweep's winner is the earliest cascade-tied
    candidate in enumeration order, strides preserve that order within
    each shard, so the full winner is necessarily its own shard's pin
    winner. The merged tie set contains the cascade-tied shard winners;
    the winning shard's own within-slice tie peers remain attached to the
    winner it contributed.
    """
    per_pin: dict[str, list[tuple[int, Any]]] = {}
    for ria, aux in entries:
        for pin in getattr(ria, "_state_ties", None) or (ria,):
            key = _pin_key(pin)
            per_pin.setdefault(key, []).append((aux["pin_winner_indices"][key], pin))
    pin_winners = []
    for indexed in per_pin.values():
        indexed.sort(key=lambda pair: pair[0])
        candidates = [c for _, c in indexed]
        ties = tuple(resolve_ties.partitions(candidates))
        winner = ties[0]
        for tie in ties:
            tie.set_partition_ties(ties)
        winner.partition_margin = _merged_partition_margin(winner, candidates)
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
