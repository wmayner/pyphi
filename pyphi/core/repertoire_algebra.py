"""Stateless repertoire computation over CandidateSystem.

Layer 2 of the kernel. Functions take a CandidateSystem as the first
argument; results are memoized via a per-instance decorator that purges
when the CandidateSystem is garbage-collected.

Numerical bodies are ports of the corresponding Subsystem methods in
pyphi/subsystem.py. Parity tests guard equivalence.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from functools import wraps
from typing import Any
from weakref import WeakValueDictionary

from pyphi import distribution as _dist
from pyphi.direction import Direction

# One cache dict per memoized function name.
_caches: dict[str, dict[tuple, Any]] = {}

# Live CandidateSystem references keyed by id, with finalizers that purge
# the corresponding cache entries on GC.
_observers: WeakValueDictionary[int, Any] = WeakValueDictionary()


def _evict(cs_id: int) -> None:
    """Purge cache entries whose first key element is ``cs_id``."""
    for fn_cache in _caches.values():
        for key in [k for k in fn_cache if k and k[0] == cs_id]:
            del fn_cache[key]


def _memoize(fn: Callable) -> Callable:
    """Memoize a function over CandidateSystem instances by ``id()``.

    Uses ``WeakValueDictionary`` + ``weakref.finalize`` so that cache
    entries are purged when the CandidateSystem is collected.
    """
    cache = _caches.setdefault(fn.__name__, {})

    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        cs_id = id(cs)
        key = (cs_id, args)
        if cs_id not in _observers:
            _observers[cs_id] = cs
            weakref.finalize(cs, _evict, cs_id)
        if key in cache:
            return cache[key]
        result = fn(cs, *args)  # raises propagate; key not added on raise
        cache[key] = result
        return result

    return wrapper


def cache_info() -> dict[str, dict[str, int]]:
    """Return per-function cache size."""
    return {name: {"size": len(c)} for name, c in _caches.items()}


def clear_caches(cs: Any | None = None) -> None:
    """Clear cache entries. If ``cs`` given, clear only that instance's entries."""
    if cs is None:
        for c in _caches.values():
            c.clear()
        return
    _evict(id(cs))


# =============================================================================
# Delegating ports (P7 Phase 4) — these forward to a transient legacy Subsystem
# constructed from the CandidateSystem's fields. Each function will be ported
# natively in Phase 8 once parity is locked in by tests.
# =============================================================================


def _legacy_subsystem(cs: Any) -> Any:
    """Construct a legacy Subsystem from a CandidateSystem (worktree only)."""
    from pyphi.subsystem import Subsystem

    return Subsystem(cs.network, cs.state, cs.node_indices, cut=cs.cut)


# ---- repertoire computation ----


@_memoize
def cause_repertoire(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Cause repertoire — IIT 4.0 Eq. 5 / Eq. 7."""
    return _legacy_subsystem(cs).cause_repertoire(mechanism, purview)


@_memoize
def effect_repertoire(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...]
) -> Any:
    """Effect repertoire — IIT 4.0 Eq. 5 / Eq. 7."""
    return _legacy_subsystem(cs).effect_repertoire(mechanism, purview)


@_memoize
def repertoire(
    cs: Any,
    direction: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
) -> Any:
    return _legacy_subsystem(cs).repertoire(direction, mechanism, purview)


def unconstrained_repertoire(
    cs: Any, direction: Direction, purview: tuple[int, ...]
) -> Any:
    return repertoire(cs, direction, (), purview)


def unconstrained_cause_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return cause_repertoire(cs, (), purview)


def unconstrained_effect_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return effect_repertoire(cs, (), purview)


def expand_repertoire(
    cs: Any,
    direction: Direction,
    repertoire_array: Any,
    *,
    new_purview: tuple[int, ...] | None = None,
) -> Any:
    if repertoire_array is None:
        return None
    purview = _dist.purview(repertoire_array)
    if purview is None:
        return None
    expanded_purview = cs.node_indices if new_purview is None else new_purview
    if not set(purview).issubset(expanded_purview):
        raise ValueError("Expanded purview must contain original purview.")
    non_purview_indices = tuple(set(expanded_purview) - set(purview))
    uc = unconstrained_repertoire(cs, direction, non_purview_indices)
    expanded = repertoire_array * uc
    return _dist.normalize(expanded)


def expand_cause_repertoire(
    cs: Any, repertoire_array: Any, *, new_purview: tuple[int, ...] | None = None
) -> Any:
    return expand_repertoire(
        cs, Direction.CAUSE, repertoire_array, new_purview=new_purview
    )


def expand_effect_repertoire(
    cs: Any, repertoire_array: Any, *, new_purview: tuple[int, ...] | None = None
) -> Any:
    return expand_repertoire(
        cs, Direction.EFFECT, repertoire_array, new_purview=new_purview
    )


@_memoize
def partitioned_repertoire(cs: Any, direction: Any, partition: Any) -> Any:
    return _legacy_subsystem(cs).partitioned_repertoire(direction, partition)


# ---- forward repertoires + probabilities ----


@_memoize
def forward_cause_repertoire(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: tuple[int, ...] | None = None,
) -> Any:
    return _legacy_subsystem(cs).forward_cause_repertoire(
        mechanism, purview, purview_state
    )


def forward_effect_repertoire(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    return _legacy_subsystem(cs).forward_effect_repertoire(mechanism, purview, **kwargs)


def forward_repertoire(
    cs: Any,
    direction: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> Any:
    return _legacy_subsystem(cs).forward_repertoire(
        direction, mechanism, purview, purview_state, **kwargs
    )


def unconstrained_forward_cause_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_forward_cause_repertoire(purview)


def unconstrained_forward_effect_repertoire(cs: Any, purview: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).unconstrained_forward_effect_repertoire(purview)


def unconstrained_forward_repertoire(
    cs: Any, direction: Any, purview: tuple[int, ...]
) -> Any:
    return _legacy_subsystem(cs).unconstrained_forward_repertoire(direction, purview)


def forward_cause_probability(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: Any,
    mechanism_state: Any | None = None,
) -> float:
    return _legacy_subsystem(cs).forward_cause_probability(
        mechanism, purview, purview_state, mechanism_state
    )


def forward_effect_probability(
    cs: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: Any,
) -> float:
    return _legacy_subsystem(cs).forward_effect_probability(
        mechanism, purview, purview_state
    )


def forward_probability(
    cs: Any,
    direction: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    purview_state: Any,
    **kwargs: Any,
) -> float:
    return _legacy_subsystem(cs).forward_probability(
        direction, mechanism, purview, purview_state, **kwargs
    )


# ---- info / phi ----


def cause_info(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    return _legacy_subsystem(cs).cause_info(mechanism, purview, **kwargs)


def effect_info(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    return _legacy_subsystem(cs).effect_info(mechanism, purview, **kwargs)


def cause_effect_info(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    return _legacy_subsystem(cs).cause_effect_info(mechanism, purview, **kwargs)


def intrinsic_information(
    cs: Any,
    direction: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    return _legacy_subsystem(cs).intrinsic_information(
        direction, mechanism, purview, **kwargs
    )


# ---- mechanism / system analysis ----


def evaluate_partition(
    cs: Any,
    direction: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    partition: Any,
    **kwargs: Any,
) -> Any:
    return _legacy_subsystem(cs).evaluate_partition(
        direction, mechanism, purview, partition, **kwargs
    )


def find_mip(
    cs: Any,
    direction: Any,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    **kwargs: Any,
) -> Any:
    return _legacy_subsystem(cs).find_mip(direction, mechanism, purview, **kwargs)


def cause_mip(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> Any:
    return _legacy_subsystem(cs).cause_mip(mechanism, purview, **kwargs)


def effect_mip(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> Any:
    return _legacy_subsystem(cs).effect_mip(mechanism, purview, **kwargs)


def phi_cause_mip(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    return _legacy_subsystem(cs).phi_cause_mip(mechanism, purview, **kwargs)


def phi_effect_mip(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    return _legacy_subsystem(cs).phi_effect_mip(mechanism, purview, **kwargs)


def phi(
    cs: Any, mechanism: tuple[int, ...], purview: tuple[int, ...], **kwargs: Any
) -> float:
    return _legacy_subsystem(cs).phi(mechanism, purview, **kwargs)


def find_mice(cs: Any, direction: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).find_mice(direction, mechanism, **kwargs)


def mic(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).mic(mechanism, **kwargs)


def mie(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).mie(mechanism, **kwargs)


def phi_max(cs: Any, mechanism: tuple[int, ...]) -> float:
    return _legacy_subsystem(cs).phi_max(mechanism)


def concept(cs: Any, mechanism: tuple[int, ...], **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).concept(mechanism, **kwargs)


def distinction(cs: Any, mechanism: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).distinction(mechanism)


def all_distinctions(cs: Any, **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).all_distinctions(**kwargs)


def sia(cs: Any, **kwargs: Any) -> Any:
    return _legacy_subsystem(cs).sia(**kwargs)


def potential_purviews(
    cs: Any, direction: Any, mechanism: tuple[int, ...], **kwargs: Any
) -> Any:
    return _legacy_subsystem(cs).potential_purviews(direction, mechanism, **kwargs)


def indices2nodes(cs: Any, indices: tuple[int, ...]) -> Any:
    return _legacy_subsystem(cs).indices2nodes(indices)
