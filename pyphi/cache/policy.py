"""Cache policy Protocol and adapters.

A CachePolicy is the uniform observability + control surface across all
of PyPhi's cache flavors. The Protocol intentionally does NOT include
``get`` / ``put`` / ``key`` — those have legitimately different
signatures across flavors (kernel uses ``id(cs)``, module-level uses
``_make_key``, instance-level uses custom keys). Forcing a uniform
get/put would re-introduce complexity that doesn't pay off.
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

    Used by the module-level ``@cache(...)`` decorator and by ``DictCache``
    instances. The ``stats`` callable returns ``(hits, misses)`` so the
    adapter doesn't need to mutate them — the wrapper closure that updates
    the counts owns them.
    """

    name: str
    backing: dict[Any, Any]
    stats: Callable[[], tuple[int, int]]

    def info(self) -> _CacheInfo:
        hits, misses = self.stats()
        return _CacheInfo(hits, misses, len(self.backing))

    def clear(self) -> None:
        self.backing.clear()
