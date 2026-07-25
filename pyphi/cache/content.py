"""Content-addressed cache with refcounted eviction and a byte-weighted bound.

Entries are keyed on ``(fingerprint, args)``, where ``fingerprint`` is a
label-free content digest of a source object (a ``System`` or ``Substrate``).
Distinct objects that share a fingerprint share entries. An entry set is
evicted when the last live source object carrying its fingerprint is
garbage-collected, so prompt release is preserved while equivalent objects
reuse results.

A second eviction path bounds occupancy while a source object stays alive: the
entries are held in a :class:`~pyphi.cache.cache_utils.ByteBoundedStore`, which
holds its byte weight steady once resident memory reaches the cache ceiling and
admits new entries by evicting least recently used ones. Occupancy is measured
in bytes rather than entries because cached repertoires vary in size by orders
of magnitude across purview orders.

Safe for concurrent use by worker threads: cached values are correct, eviction
is sound, and no operation raises under concurrent access. The ``hits`` and
``misses`` counters are best-effort under free-threaded Python (exact under the
GIL and under process isolation) — they are diagnostics, left out of the lock
to keep the hot path contention-free. The tracked byte weight carries the same
caveat: a hit reinserting its entry concurrently with a fingerprint eviction
can leave the weight off by that entry, which shifts where the bound falls
without affecting any cached value. Admission and eviction themselves are
locked.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from typing import Any

from pyphi.cache.cache_utils import _MISSING
from pyphi.cache.cache_utils import ByteBoundedStore
from pyphi.cache.policy import _DictCacheAdapter
from pyphi.cache.registry import register as _register_policy


class ContentCache:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hits = 0
        self.misses = 0
        self._store = ByteBoundedStore()
        self._cache = self._store.data
        self._live: dict[bytes, int] = {}
        self._observed: set[int] = set()
        # Finalize handles by source id, so clear() can detach them; a
        # finalizer left armed across clear() would fire alongside the one
        # registered by a later re-observation and over-decrement the
        # refcount, evicting entries that still have live carriers.
        self._finalizers: dict[int, weakref.finalize] = {}
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
                weigh=lambda: (self._store.nbytes, self._store.evictions),
            )
        )

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def nbytes(self) -> int:
        """Estimated bytes held by this cache's entries."""
        return self._store.nbytes

    @property
    def evictions(self) -> int:
        """Entries discarded to stay within the byte bound."""
        return self._store.evictions

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
            self._finalizers[sid] = weakref.finalize(
                source, self._on_death, sid, fingerprint
            )

    def _on_death(self, sid: int, fingerprint: bytes) -> None:
        with self._lock:
            self._observed.discard(sid)
            self._finalizers.pop(sid, None)
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
        """Return the cached value for ``(fingerprint, args)``, else compute it.

        On a miss, ``compute`` is called and its result returned. An exception
        from ``compute`` propagates and no entry is added. The result is stored
        when ``store`` is true, evicting least recently used entries to stay
        within the cache's byte bound once resident memory has reached the
        configured ceiling.

        A hit reinserts its entry, which moves it to the recent end of the
        backing dict's iteration order.

        The hit path is lock-free; the ``hits`` and ``misses`` counters it
        updates are best-effort under free-threaded Python.
        """
        key = (fingerprint, args)
        value = self._cache.pop(key, _MISSING)
        if value is not _MISSING:
            self._cache[key] = value
            self.hits += 1
            return value
        self.misses += 1
        result = compute()  # raises propagate; key not added on raise
        if store:
            self._admit(key, result)
        return result

    def _admit(self, key: tuple[bytes, tuple], value: Any) -> None:
        with self._lock:
            self._store.admit(key, value)

    def evict(self, fingerprint: bytes) -> None:
        with self._lock:
            self._evict_locked(fingerprint)

    def _evict_locked(self, fingerprint: bytes) -> None:
        # list(self._cache) is an atomic snapshot; iterate it (not the live
        # dict) and pop, so a concurrent lock-free hit cannot raise
        # "dictionary changed size during iteration".
        for key in list(self._cache):
            if key and key[0] == fingerprint:
                self._store.discard(key)

    def clear(self) -> None:
        with self._lock:
            for handle in self._finalizers.values():
                handle.detach()
            self._finalizers.clear()
            self._store.clear()
            self._live.clear()
            self._observed.clear()
            self.hits = 0
            self.misses = 0
