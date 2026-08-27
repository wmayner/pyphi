# cache/cache_utils.py
"""Common utilities for caching."""

import os
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from pyphi.conf import config

_CacheInfo = namedtuple(
    "CacheInfo",
    ["hits", "misses", "currsize", "nbytes", "evictions"],
    defaults=(0, 0),
)


@lru_cache(maxsize=1)
def _process_handle(pid: int) -> psutil.Process:
    """The psutil handle for ``pid``, kept for reuse.

    Constructing the handle costs ten times as much as reading resident
    memory from an existing one, and :func:`memory_full` is called on every
    cache miss. Holding only the newest handle means a forked child replaces
    its parent's rather than reusing it.
    """
    return psutil.Process(pid)


def _cgroup_memory_limit(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    self_cgroup: Path = Path("/proc/self/cgroup"),
) -> int | None:
    """Resident-memory limit of this process's cgroup, or ``None`` if unconfined.

    Reads the memory limit at the process's own group and at every ancestor
    group up to the hierarchy root — ``memory.max`` under cgroup v2,
    ``memory.limit_in_bytes`` under v1 — and returns the smallest one found.
    An ancestor's limit binds every group below it, and scheduler-managed
    layouts (Slurm, systemd slices) commonly place the limit on a parent
    rather than on the process's own leaf group. A literal ``max``, or a value
    at or above total physical memory, means no limit at that level — cgroup
    v1 represents "unlimited" as a number near the word size rather than as a
    sentinel.

    Parameters
    ----------
    cgroup_root : pathlib.Path, optional
        Mount point of the cgroup hierarchy.
    self_cgroup : pathlib.Path, optional
        The file naming this process's groups.

    Returns
    -------
    int or None
        The limit in bytes, or ``None`` when the process may use the whole
        machine or the limit cannot be read.
    """
    candidates: list[Path] = []

    def walk_to_root(base: Path, relative: str, filename: str) -> None:
        """Collect ``filename`` at the group and at every ancestor group."""
        node = base / relative if relative else base
        while True:
            candidates.append(node / filename)
            if node in (base, node.parent):
                break
            node = node.parent

    try:
        for line in self_cgroup.read_text().splitlines():
            hierarchy, _, remainder = line.partition(":")
            controllers, _, relative = remainder.partition(":")
            relative = relative.strip().lstrip("/")
            if hierarchy == "0" and not controllers:
                walk_to_root(cgroup_root, relative, "memory.max")
            elif "memory" in controllers.split(","):
                walk_to_root(cgroup_root / "memory", relative, "memory.limit_in_bytes")
    except OSError:
        pass
    # The hierarchy root as seen from inside a container's cgroup namespace,
    # reachable even when the process's own groups cannot be read.
    candidates.append(cgroup_root / "memory.max")
    candidates.append(cgroup_root / "memory" / "memory.limit_in_bytes")

    total = psutil.virtual_memory().total
    limits = []
    for path in candidates:
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue  # "max" under v2
        if 0 < value < total:
            limits.append(value)
    return min(limits) if limits else None


@lru_cache(maxsize=1)
def memory_limit_bytes() -> int:
    """Resident memory this process may use.

    The process's cgroup allowance when it is confined to less than the
    machine — a scheduler-managed job, a container, a cgroup — and total
    physical memory otherwise. Cached, since the allowance is fixed when the
    process starts.
    """
    limit = _cgroup_memory_limit()
    return limit if limit is not None else psutil.virtual_memory().total


def memory_full():
    """Check if the memory is too full for further caching.

    Measures resident memory against ``memory_ceiling_bytes`` when that
    is set, and otherwise against ``memory_ceiling_percentage`` of the
    memory this process may use (see :func:`memory_limit_bytes`) — which is the
    machine's total only when the process is free to use all of it.
    """
    current_process = _process_handle(os.getpid())
    budget = config.infrastructure.memory_ceiling_bytes
    if budget is None:
        budget = (
            memory_limit_bytes() * config.infrastructure.memory_ceiling_percentage // 100
        )
    return current_process.memory_info().rss > budget


_ENTRY_OVERHEAD_BYTES = 512
"""Non-payload cost of one entry: its key, the value object, and a dict slot.

One constant rather than a per-entry measurement, since sizing each key exactly
would cost more than the admission it informs. Its accuracy governs which
entries are evicted rather than how many: a store's budget is latched from its
own tracked weight, so a uniform error cancels from the total and changes only
the weight of a large array relative to a small one.
"""

_ARRAY_DIM_BYTES = 16
"""Per-dimension cost of an ndarray's shape and strides, one intp each."""

_RECHECK_INTERVAL = 4096
"""Admissions between ceiling re-checks once a store is bounded.

Re-checking lets a bound that was latched during a transient spike be lifted,
and keeps the cost of the check off all but one admission in this many.
"""

_MISSING = object()


def entry_weight(value: Any) -> int:
    """Estimated bytes one cached value occupies, including its key and slot.

    An ndarray view keeps its whole underlying buffer alive, and the cache may
    be that buffer's only owner, so a view is charged the buffer of the array
    it derives from. A base that is itself also cached is then charged twice;
    overcounting a shared buffer only evicts sooner, where undercounting lets
    the bound exceed real memory. A sequence — the combinatorial index tables
    are lists of tuples — is charged per element, since its cost is the
    elements rather than any single buffer.
    """
    if isinstance(value, np.ndarray):
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        return _ENTRY_OVERHEAD_BYTES + _ARRAY_DIM_BYTES * value.ndim + owner.nbytes
    if isinstance(value, list | tuple):
        return _ENTRY_OVERHEAD_BYTES + sum(map(_element_weight, value))
    return _ENTRY_OVERHEAD_BYTES


_SEQUENCE_HEADER_BYTES = 56
"""Object header of a tuple or list, before its element pointers."""

_POINTER_BYTES = 8

_SCALAR_BYTES = 32
"""A small int or similar leaf, counted once rather than by identity.

Small integers are interned, so charging each occurrence overstates a table of
index tuples. The overstatement is uniform across entries, which is what the
bound compares.
"""

_MAX_WEIGHT_DEPTH = 4


def _element_weight(element: Any, depth: int = 0) -> int:
    """Bytes one element of a cached sequence occupies.

    Recurses through nested sequences to a fixed depth, since the index tables
    nest two or three levels and a bound that stopped at the first would
    undercount them by the width of every inner tuple.
    """
    if depth < _MAX_WEIGHT_DEPTH and isinstance(element, tuple | list):
        return (
            _SEQUENCE_HEADER_BYTES
            + _POINTER_BYTES * len(element)
            + sum(_element_weight(x, depth + 1) for x in element)
        )
    return _SCALAR_BYTES


class ByteBoundedStore:
    """A dict that holds its byte weight steady once memory reaches the ceiling.

    Insertion order is the recency order, so the least recently used entry is
    the first one iteration yields; a caller reinserts an entry on a hit to
    move it to the recent end. Until resident memory reaches the cache ceiling
    (see :func:`memory_full`) the store grows freely. From then on it admits an
    entry by evicting least recently used ones, and refuses one too large to
    fit an empty store rather than flushing everything to hold it.

    Eviction holds occupancy steady; it does not reduce resident memory, since
    freeing a Python object returns its memory to the process allocator for
    reuse rather than to the operating system.

    Not internally synchronized. A caller sharing a store across threads holds
    its own lock across :meth:`admit` and :meth:`discard`; lock-free hits that
    pop and reinsert entries in ``data`` directly are tolerated, and the
    eviction loop never raises because of them.
    """

    def __init__(self) -> None:
        self.data: dict[Any, Any] = {}
        self.evictions = 0
        self._weight = 0
        self._budget: int | None = None
        self._admissions = 0

    @property
    def nbytes(self) -> int:
        """Estimated bytes held by this store's entries."""
        return self._weight

    def admit(self, key: Any, value: Any) -> None:
        """Store an entry, evicting least recently used ones to make room."""
        weight = entry_weight(value)
        self._admissions += 1
        # Consult the ceiling while unbounded, periodically once bounded, and
        # whenever the bound is about to refuse an entry outright, so a bound
        # latched during a transient spike cannot persist.
        if (
            self._budget is None
            or self._admissions % _RECHECK_INTERVAL == 0
            or weight > self._budget
        ):
            if memory_full():
                # Hold occupancy here and trade old entries for new ones.
                if self._budget is None:
                    self._budget = self._weight
            else:
                # Room again, whether because the spike that set the bound has
                # passed or because memory was released elsewhere.
                self._budget = None
        if self._budget is not None:
            if weight > self._budget:
                # Does not fit even in an empty store: refuse up front rather
                # than draining the working set first for an entry that could
                # never be held.
                return
            while self.data and self._weight + weight > self._budget:
                # A lock-free hit on another thread may pop and reinsert an
                # entry at any point, so the iterator can find the dict
                # changed under it and a chosen key can vanish before the
                # pop; retry in both cases rather than raise. Each retry
                # re-reads the loop condition, and every successful pop
                # reduces the weight, so the loop terminates.
                try:
                    oldest = next(iter(self.data))
                except (RuntimeError, StopIteration):
                    continue
                evicted = self.data.pop(oldest, _MISSING)
                if evicted is _MISSING:
                    continue
                self._weight -= entry_weight(evicted)
                self.evictions += 1
            if self._weight + weight > self._budget:
                # Still no room: lock-free hits on other threads reinserted
                # entries while the loop ran.
                return
        self.discard(key)
        self.data[key] = value
        self._weight += weight

    def discard(self, key: Any) -> None:
        """Remove an entry if present, crediting back its weight."""
        previous = self.data.pop(key, _MISSING)
        if previous is not _MISSING:
            self._weight -= entry_weight(previous)

    def clear(self) -> None:
        self.data.clear()
        self.evictions = 0
        self._weight = 0
        self._budget = None
        self._admissions = 0


class _HashedSeq(list):
    """This class guarantees that ``hash()`` will be called no more than once
    per element.  This is important because the ``lru_cache()`` will hash the
    key multiple times on a cache miss.
    """

    __slots__ = ("hashvalue",)

    def __init__(self, tup, hash=hash):
        super().__init__()
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):  # type: ignore[override]
        return self.hashvalue


def _make_key(
    args,
    kwds,
    typed,
    kwd_mark=(object(),),
    fasttypes=None,
    sorted=sorted,
    tuple=tuple,
    type=type,
    len=len,
):
    """Make a cache key from optionally typed positional and keyword arguments.

    The key is constructed in a way that is flat as possible rather than as a
    nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache its
    hash value, then that argument is returned without a wrapper.  This saves
    space and improves lookup speed.
    """
    if fasttypes is None:
        fasttypes = {int, str, frozenset, type(None)}
    key = args
    sorted_items = None
    if kwds:
        sorted_items = sorted(kwds.items())
        key += kwd_mark
        for item in sorted_items:
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            assert sorted_items is not None  # Type narrowing: kwds is truthy
            key += tuple(type(v) for k, v in sorted_items)
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)
