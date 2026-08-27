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
    """Merge the strides of one (mechanism, direction, purview).

    Each entry pairs one stride's **per-pin local minimum** — one RIA per
    specified-state pin, minimized over the stride's own partition slice
    (one single entry per stride for pin-less formalisms) — with its aux
    record, whose ``pin_key`` identifies the pin and whose
    ``pin_winner_index`` is the global enumeration index of the pin's
    locally winning partition.

    φ per pin is a minimum over partitions and pin selection is a maximum
    over pins, so the merge takes the cross-stride minimum per pin first
    — every stride reports every pin, so this is the true global minimum
    — and only then runs the state cascade over the per-pin global
    winners, exactly as the unsharded search does. Per pin, candidates
    are restored to global enumeration order so ties resolve to the
    identical representative a full sweep selects.
    """
    per_pin: dict[str, list[tuple[int, Any]]] = {}
    for pin, aux in entries:
        per_pin.setdefault(aux["pin_key"], []).append((aux["pin_winner_index"], pin))
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
        # numerics: exact — reported margin, not a selection.
        winner.purview_margin = max(0.0, float(winner.phi) - best_rival)
    return winner


def build_distinction(mechanism: Any, mic: Any, mie: Any) -> Any:
    """Assemble a distinction from a merged MIC and MIE."""
    from pyphi.models import Concept

    return Concept(mechanism=tuple(mechanism), cause=mic, effect=mie)


def _merge_sia_partition_candidates(entries: list[tuple[Any, dict]]) -> Any:
    """Cross-stride minimum over one partition sweep (one state pair)."""
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


def merge_sia_strides(entries: list[tuple[Any, dict]], system: Any = None) -> Any:
    """Merge SIA strides.

    A stride whose analysis short-circuited to a null result (e.g. the
    system lacks strong connectivity) never consulted its partition slice;
    every stride of the cell then carries the identical result, and any
    one of them is the merge.

    Entries carrying a ``pair_key`` aux field are per-(cause, effect)
    specified-state-pair local minima: φ_s per pair is a minimum over
    partitions and pair selection is a cascade over pairs, so the merge
    takes the cross-stride minimum per pair first — every stride reports
    every pair — and then runs the same pair-selection cascade the
    unsharded search runs (:func:`pyphi.formalism.iit4._resolve_pair_sias`,
    which needs ``system`` for Composition escalation and the canonical
    tie-break). Entries without ``pair_key`` are single-sweep results
    (pin-less formalisms, or a cell with an untied specified state) and
    merge as a plain cross-stride minimum with global order restored.
    """
    for sia, aux in entries:
        if aux.get("short_circuit"):
            return sia
    per_pair: dict[tuple, list[tuple[Any, dict]]] = {}
    unpaired: list[tuple[Any, dict]] = []
    for sia, aux in entries:
        if "pair_key" in aux:
            per_pair.setdefault(tuple(map(_as_key, aux["pair_key"])), []).append(
                (sia, aux)
            )
        else:
            unpaired.append((sia, aux))
    if not per_pair:
        return _merge_sia_partition_candidates(unpaired)
    if unpaired:
        raise ValueError(
            "cannot merge a mix of per-pair and single-sweep SIA stride entries"
        )
    if system is None:
        raise ValueError("merging per-pair SIA strides requires the system")
    from pyphi.formalism.iit4 import merge_pair_minima

    merged_pairs = {
        key: _merge_sia_partition_candidates(pair_entries)
        for key, pair_entries in per_pair.items()
    }
    return merge_pair_minima(system, merged_pairs)


def _as_key(value: Any) -> Any:
    """Normalize a serialized pair-key component (lists round-trip as
    tuples; ``None`` stays ``None``)."""
    return tuple(value) if isinstance(value, list) else value
