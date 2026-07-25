"""Cache policy Protocol and adapters.

A ``CachePolicy`` is the uniform observability and control surface across all
of PyPhi's cache flavors: it declares only ``name``, ``info()``, and
``clear()``. It does not include ``get`` / ``put`` / ``key``, because those
have different signatures across flavors (the kernel keys on ``id(cs)``,
module-level caches on ``_make_key``, instance-level caches on custom keys).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from .cache_utils import _CacheInfo


@runtime_checkable
class CachePolicy(Protocol):
    """Uniform observability + control surface for caches."""

    name: str

    def info(self) -> _CacheInfo: ...
    def clear(self) -> None: ...


@dataclass
class _DictCacheAdapter:
    """Adapter wrapping a backing dict with externally-tracked hit/miss counts.

    Used by the module-level ``@cache(...)`` decorator and by ``ContentCache``
    instances. The ``stats`` callable returns ``(hits, misses)`` so the
    adapter doesn't need to mutate them — the wrapper closure that updates
    the counts owns them. The optional ``weigh`` callable returns
    ``(nbytes, evictions)`` for caches that track occupancy in bytes; caches
    that do not report zero for both.
    """

    name: str
    backing: dict[Any, Any]
    stats: Callable[[], tuple[int, int]]
    weigh: Callable[[], tuple[int, int]] | None = None

    def info(self) -> _CacheInfo:
        hits, misses = self.stats()
        nbytes, evictions = self.weigh() if self.weigh else (0, 0)
        return _CacheInfo(hits, misses, len(self.backing), nbytes, evictions)

    def clear(self) -> None:
        self.backing.clear()
