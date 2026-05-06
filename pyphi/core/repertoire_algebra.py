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
