"""Disk-backed content-addressed store for top-level results.

Persists serialized results to one file per key under
``DISK_CACHE_LOCATION``. Keys are opaque hex strings built by the key
module functions; values are ``serialize``-encoded results. A truncated or
unreadable file decodes to ``None`` (a silent miss), never an exception
reaching the caller; staleness across code or config changes is handled by
the cache key (its code-version component), not an in-file tag.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any

from pyphi import constants
from pyphi import serialize
from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.registry import register as _register_policy
from pyphi.provenance import _git_info


def _decode_or_none(data: bytes) -> Any | None:
    """Deserialize a stored result; ``None`` on any error (a cache miss).

    Staleness across code or config changes is handled entirely by the cache
    key (it folds in a code-version component), so there is no in-file version
    tag; this only tolerates a corrupt/truncated file.
    """
    try:
        return serialize.loads(data, format="msgpack")
    except Exception:  # any decode failure is a cache miss, not an error
        return None


class DiskCache:
    """A content-addressed file store satisfying the CachePolicy surface."""

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
        tmp = self._dir / f".{key}.{os.getpid()}.tmp"
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
    """Digest only the configuration fields that change a result value."""
    iit = snapshot.formalism.iit
    fields = (
        iit.version,
        iit.mechanism_phi_measure,
        iit.system_phi_measure,
        iit.specification_measure,
        iit.ces_measure,
        iit.mechanism_partition_scheme,
        iit.system_partition_scheme,
        snapshot.numerics.precision,
    )
    return repr(fields).encode()


def _code_version() -> str:
    """The running pyphi code's identity: git sha in a checkout, else version."""
    sha, _dirty = _git_info()
    if sha is not None:
        return f"git:{sha}"
    return f"v:{importlib.metadata.version('pyphi')}"


def result_cache_key(system: Any, kind: str, snapshot: Any) -> str | None:
    """Hex cache key, or ``None`` (do not cache) when the git tree is dirty."""
    _sha, dirty = _git_info()
    if dirty:
        return None
    h = hashlib.blake2b(digest_size=32)
    h.update(system._fingerprint)
    h.update(kind.encode())
    h.update(_config_digest(snapshot))
    h.update(_code_version().encode())
    return h.hexdigest()


_RESULT_DISK_CACHE = DiskCache("disk.results", "results")


def maybe_disk_cached(system: Any, kind: str, user_kwargs: dict, compute: Any) -> Any:
    """Return a disk-cached result for ``compute()`` when it is safe to.

    Bypasses (just calls ``compute()``) when the cache is disabled, when the
    caller passed result-affecting kwargs the key cannot capture, or when the
    git tree is dirty (``result_cache_key`` returns ``None``).
    """
    from pyphi.conf import config

    if user_kwargs or not config.infrastructure.disk_cache_results:
        return compute()
    key = result_cache_key(system, kind, config.snapshot())
    if key is None:
        return compute()
    hit = _RESULT_DISK_CACHE.get(key)
    if hit is not None:
        result = _decode_or_none(hit)
        if result is not None:
            return result
    result = compute()
    _RESULT_DISK_CACHE.put(key, serialize.dumps(result, format="msgpack"))
    return result
