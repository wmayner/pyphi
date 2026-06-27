"""Content-addressed cache with refcounted, GC-driven eviction.

Entries are keyed on ``(fingerprint, args)``, where ``fingerprint`` is a
label-free content digest of a source object (a ``System`` or ``Substrate``).
Distinct objects that share a fingerprint share entries. An entry set is
evicted when the last live source object carrying its fingerprint is
garbage-collected, so prompt release is preserved while equivalent objects
reuse results.

Safe for concurrent use by worker threads: cached values are correct, eviction
is sound, and no operation raises under concurrent access. The ``hits`` and
``misses`` counters are best-effort under free-threaded Python (exact under the
GIL and under process isolation) — they are diagnostics, left out of the lock
to keep the hot path contention-free.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from typing import Any

from pyphi.cache import cache_utils
from pyphi.cache.policy import _DictCacheAdapter
from pyphi.cache.registry import register as _register_policy


class ContentCache:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hits = 0
        self.misses = 0
        self._cache: dict[tuple[bytes, tuple], Any] = {}
        self._live: dict[bytes, int] = {}
        self._observed: set[int] = set()
        # Guards the eviction and refcount bookkeeping only; the hot path
        # (get_or_compute) is lock-free. Reentrant because a weakref finalizer
        # (_on_death) can fire on the thread already holding the lock — e.g. a
        # cyclic GC triggered by an allocation inside the locked region.
        self._lock = threading.RLock()
        _register_policy(
            _DictCacheAdapter(
                name=name,
                backing=self._cache,
                stats=lambda: (self.hits, self.misses),
            )
        )

    @property
    def size(self) -> int:
        return len(self._cache)

    def observe(self, source: Any, fingerprint: bytes) -> None:
        """Register ``source`` as a live carrier of ``fingerprint``."""
        sid = id(source)
        if sid in self._observed:
            return
        with self._lock:
            if sid in self._observed:
                return
            self._observed.add(sid)
            self._live[fingerprint] = self._live.get(fingerprint, 0) + 1
            weakref.finalize(source, self._on_death, sid, fingerprint)

    def _on_death(self, sid: int, fingerprint: bytes) -> None:
        with self._lock:
            self._observed.discard(sid)
            remaining = self._live.get(fingerprint, 0) - 1
            if remaining <= 0:
                self._live.pop(fingerprint, None)
                self._evict_locked(fingerprint)
            else:
                self._live[fingerprint] = remaining

    def get_or_compute(
        self,
        fingerprint: bytes,
        args: tuple,
        compute: Callable[[], Any],
        *,
        store: bool = True,
    ) -> Any:
        key = (fingerprint, args)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        result = compute()  # raises propagate; key not added on raise
        if store and not cache_utils.memory_full():
            self._cache[key] = result
        return result

    def evict(self, fingerprint: bytes) -> None:
        with self._lock:
            self._evict_locked(fingerprint)

    def _evict_locked(self, fingerprint: bytes) -> None:
        # list(self._cache) is an atomic snapshot; iterate it (not the live
        # dict) and pop, so a concurrent lock-free insert cannot raise
        # "dictionary changed size during iteration".
        for key in list(self._cache):
            if key and key[0] == fingerprint:
                self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._live.clear()
            self._observed.clear()
            self.hits = 0
            self.misses = 0
