"""Disk-backed content-addressed store for top-level results.

Persists serialized results to one file per key under
``DISK_CACHE_LOCATION``. Keys are opaque hex strings built by the key
module functions; values are ``serialize``-encoded results. A truncated or
unreadable file decodes to ``None`` (a silent miss), never an exception
reaching the caller; staleness across code or config changes is handled by
the cache key (its code-version component), not an in-file tag.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from pyphi import constants
from pyphi import serialize
from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.registry import register as _register_policy
from pyphi.provenance import _git_info

log = logging.getLogger(__name__)


def _decode_or_none(data: bytes, node_labels: Any = None) -> Any | None:
    """Deserialize a stored result; ``None`` on any error (a cache miss).

    Staleness across code or config changes is handled entirely by the cache
    key (it folds in a code-version component), so there is no in-file version
    tag; this only tolerates a corrupt/truncated file. ``node_labels``
    replaces the stored label frame, so a hit is decoded in the requesting
    system's labels.
    """
    try:
        return serialize.loads(data, format="msgpack", node_labels=node_labels)
    except Exception:  # any decode failure is a cache miss, not an error
        return None


class DiskCache:
    """A content-addressed file store satisfying the CachePolicy surface.

    Marked ``persistent``, so registry-wide :func:`pyphi.cache.clear_all` —
    whose purpose is recovering memory — leaves the durable files alone.
    Clear explicitly with :meth:`clear` or ``pyphi.cache.clear(name)``.
    """

    persistent = True

    def __init__(self, name: str, subdir: str) -> None:
        self.name = name
        self._subdir = subdir
        self.hits = 0
        self.misses = 0
        _register_policy(self)

    @property
    def _dir(self) -> Path:
        return constants.DISK_CACHE_LOCATION / self._subdir

    def get(self, key: str) -> bytes | None:
        try:
            data = (self._dir / key).read_bytes()
        except OSError:
            self.misses += 1
            return None
        self.hits += 1
        return data

    def put(self, key: str, data: bytes) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        # The temp name is unique per call, not just per process: two threads
        # writing the same key must not collide on the temp path.
        tmp = self._dir / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        tmp.write_bytes(data)
        tmp.replace(self._dir / key)

    def clear(self) -> None:
        if self._dir.exists():
            for path in self._dir.iterdir():
                if path.is_file():
                    path.unlink()

    @property
    def size(self) -> int:
        if not self._dir.exists():
            return 0
        return sum(1 for path in self._dir.iterdir() if path.is_file())

    def info(self) -> _CacheInfo:
        return _CacheInfo(self.hits, self.misses, self.size)


def _config_digest(snapshot: Any) -> bytes:
    """Digest every formalism and numerics configuration field.

    Complete by construction: any field added to the formalism or numerics
    layers automatically enters the key. Infrastructure settings are excluded
    because they must not affect result values. Over-keying costs only cache
    misses; under-keying silently returns a result computed under a different
    configuration.
    """
    return repr(
        (
            dataclasses.asdict(snapshot.formalism),
            dataclasses.asdict(snapshot.numerics),
        )
    ).encode()


def result_cache_key(system: Any, kind: str, snapshot: Any) -> str | None:
    """Hex cache key, or ``None`` (do not cache) when the git tree is dirty.

    The code-version component is the git sha in a checkout, else the released
    pyphi version, so a code change that alters results changes the key.
    """
    sha, dirty = _git_info()
    if dirty:
        return None
    if sha is not None:
        version = f"git:{sha}"
    else:
        version = f"v:{importlib.metadata.version('pyphi')}"
    h = hashlib.blake2b(digest_size=32)
    h.update(system._fingerprint)
    h.update(kind.encode())
    h.update(_config_digest(snapshot))
    h.update(version.encode())
    return h.hexdigest()


_RESULT_DISK_CACHE = DiskCache("disk.results", "results")


def maybe_disk_cached(system: Any, kind: str, user_kwargs: dict, compute: Any) -> Any:
    """Return a disk-cached result for ``compute()`` when it is safe to.

    Bypasses (just calls ``compute()``) when the cache is disabled, when the
    caller passed result-affecting kwargs the key cannot capture, or when the
    git tree is dirty (``result_cache_key`` returns ``None``). A hit is
    decoded with the requesting system's node labels (the key is label-free,
    so an equivalent system with different labels may have produced the
    entry).
    """
    from pyphi.conf import config

    if user_kwargs or not config.infrastructure.disk_cache_results:
        return compute()
    key = result_cache_key(system, kind, config.snapshot())
    if key is None:
        return compute()
    hit = _RESULT_DISK_CACHE.get(key)
    if hit is not None:
        result = _decode_or_none(hit, node_labels=system.node_labels)
        if result is not None:
            return result
    result = compute()
    try:
        _RESULT_DISK_CACHE.put(key, serialize.dumps(result, format="msgpack"))
    except Exception:
        # The write is best-effort: never let a serialization or storage
        # failure destroy a freshly computed result.
        log.warning(
            "disk result cache write failed; returning the result uncached",
            exc_info=True,
        )
    return result
