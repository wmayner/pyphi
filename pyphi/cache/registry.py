"""Process-local registry of cache policies.

Every PyPhi cache (kernel, module-level, instance-level) registers a
``CachePolicy`` adapter here at construction time. The registry exposes
a uniform ``info()`` / ``clear_all()`` / ``clear(name)`` surface,
re-exported from :mod:`pyphi.cache`.

This registry is process-local — caches in PyPhi are not shared across
processes. PyPhi assumes process-isolated parallelism (Ray-based); see
the threading note in :mod:`pyphi.cache`.
"""

from __future__ import annotations

from .cache_utils import _CacheInfo
from .policy import CachePolicy

_registry: dict[str, CachePolicy] = {}


def register(policy: CachePolicy) -> None:
    """Register a cache policy. Replaces silently on duplicate name."""
    _registry[policy.name] = policy


def unregister(name: str) -> None:
    """Remove a registration. ``KeyError`` if name unknown."""
    del _registry[name]


def info() -> dict[str, _CacheInfo]:
    """Return per-cache stats for every registered policy."""
    return {name: policy.info() for name, policy in _registry.items()}


def clear_all() -> None:
    """Clear every registered cache."""
    for policy in _registry.values():
        policy.clear()


def clear(name: str) -> None:
    """Clear one named cache. ``KeyError`` if unknown."""
    _registry[name].clear()
