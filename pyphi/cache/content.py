"""Content-addressed cache with refcounted, GC-driven eviction.

Entries are keyed on ``(fingerprint, args)``, where ``fingerprint`` is a
label-free content digest of a source object (a ``System`` or ``Substrate``).
Distinct objects that share a fingerprint share entries. An entry set is
evicted when the last live source object carrying its fingerprint is
garbage-collected, so prompt release is preserved while equivalent objects
reuse results. NOT thread-safe (matching the kernel cache it replaces).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

from pyphi.cache.cache_utils import memory_full
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
        self._observed.add(sid)
        self._live[fingerprint] = self._live.get(fingerprint, 0) + 1
        weakref.finalize(source, self._on_death, sid, fingerprint)

    def _on_death(self, sid: int, fingerprint: bytes) -> None:
        self._observed.discard(sid)
        remaining = self._live.get(fingerprint, 0) - 1
        if remaining <= 0:
            self._live.pop(fingerprint, None)
            self.evict(fingerprint)
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
        if store and not memory_full():
            self._cache[key] = result
        return result

    def evict(self, fingerprint: bytes) -> None:
        for key in [k for k in self._cache if k and k[0] == fingerprint]:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()
        self._live.clear()
        self._observed.clear()
        self.hits = 0
        self.misses = 0
