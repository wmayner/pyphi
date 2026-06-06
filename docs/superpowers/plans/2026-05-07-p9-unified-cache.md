# P9 — Unified Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every PyPhi cache (kernel `_memoize`, module-level `@cache(...)`, instance-level `DictCache`) a uniform observability + control surface (`pyphi.cache.info()` / `pyphi.cache.clear_all()`), bound the kernel cache by `MAXIMUM_CACHE_MEMORY_PERCENTAGE`, document the single-threaded-per-process assumption, and remove dead Redis/joblib scaffolding.

**Architecture:** Tiny `CachePolicy` Protocol (`name`, `info()`, `clear()`) + a process-local registry. Each cache flavor keeps its own decorator/lifecycle but registers an adapter at construction time. No unified `get/put` surface — that's where lifecycles legitimately differ.

**Tech Stack:** Python 3.12+, existing `pyphi/cache/` package, existing `pyphi/core/repertoire_algebra.py` kernel, `psutil` (already a dep) for memory check.

**Spec:** `docs/superpowers/specs/2026-05-07-p9-unified-cache-design.md`.

**Branch:** `feature/p9-unified-cache` (already cut off `feature/p7-kernel-rewrite` tip in worktree `../pyphi-p7-kernel-rewrite`).

---

## File Structure

```
pyphi/cache/
├── __init__.py           # MODIFY: re-export registry + policy types; remove joblib_memory; add module docstring
├── policy.py             # CREATE: CachePolicy Protocol + concrete adapters
├── registry.py           # CREATE: process-local registry (register/info/clear/clear_all)
├── cache_utils.py        # UNCHANGED: keep _make_key, memory_full
└── redis.py              # DELETE

pyphi/core/
└── repertoire_algebra.py # MODIFY: wrap _caches values in _KernelCache; add memory bound; mirror threading docstring

pyphi/network.py          # MODIFY: pass instance name to PurviewCache
pyphi/jsonify.py          # MODIFY: name _ObjectCache for registration
pyphi/conf.py             # MODIFY: remove REDIS_CACHE + REDIS_CONFIG Options
pyphi/conf.pyi            # MODIFY: remove REDIS_CACHE + REDIS_CONFIG fields

docs/conf.py              # MODIFY: remove |MICECache| Sphinx alias

test/
├── test_cache_registry.py        # CREATE: registry contract tests
├── test_cache_policy.py          # CREATE: Protocol + adapter tests
├── test_core_repertoire_algebra.py  # EXTEND: memory-bound test
└── test_cache_integration.py     # CREATE: end-to-end registry walk after exercising suite

changelog.d/p9-unified-cache.refactor.md  # CREATE: user-facing changelog
```

---

## Phase 0: Baseline

### Task 0.1: Confirm worktree + branch state

**Files:** none (verification only)

- [ ] **Step 1: Confirm worktree path and branch**

Run: `cd ../pyphi-p7-kernel-rewrite && git status && git rev-parse --abbrev-ref HEAD`
Expected: clean working tree (modulo the untracked `filename` artifact); branch `feature/p9-unified-cache`

- [ ] **Step 2: Confirm baseline tests pass on current branch tip**

Run fast lane in foreground: `uv run pytest -x test/test_models.py test/test_partition.py test/test_subsystem_surface.py test/test_golden_regression.py test/test_invariants.py test/test_core_repertoire_algebra.py test/test_models_layering.py -q`
Expected: all green.

If any failure occurs that isn't from this branch's own work, STOP and investigate before proceeding.

---

## Phase 1: CachePolicy Protocol + Registry

### Task 1.1: Define the Protocol (test-first)

**Files:**
- Create: `test/test_cache_policy.py`
- Create: `pyphi/cache/policy.py`

- [ ] **Step 1: Write failing test for the Protocol shape**

Create `test/test_cache_policy.py`:

```python
"""Tests for the CachePolicy Protocol and adapters."""

from __future__ import annotations

from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.policy import CachePolicy
from pyphi.cache.policy import _DictCacheAdapter


def test_dict_cache_adapter_is_a_cache_policy():
    """Concrete adapters must structurally satisfy the Protocol."""
    backing: dict[tuple, int] = {"a": 1, "b": 2}
    adapter = _DictCacheAdapter(name="test.adapter", backing=backing, stats=lambda: (3, 4))
    # Static check: adapter is a CachePolicy
    p: CachePolicy = adapter
    assert p.name == "test.adapter"
    info = p.info()
    assert isinstance(info, _CacheInfo)
    assert info.hits == 3
    assert info.misses == 4
    assert info.currsize == 2


def test_dict_cache_adapter_clear_clears_backing():
    backing: dict[tuple, int] = {"a": 1}
    adapter = _DictCacheAdapter(name="test.adapter", backing=backing, stats=lambda: (0, 0))
    adapter.clear()
    assert backing == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_cache_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.cache.policy'`

- [ ] **Step 3: Implement Protocol + DictCacheAdapter**

Create `pyphi/cache/policy.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_cache_policy.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add test/test_cache_policy.py pyphi/cache/policy.py
git commit -m "$(cat <<'EOF'
P9: add CachePolicy Protocol + dict adapter

Protocol surface is intentionally narrow (name, info, clear) — get/put
signatures legitimately differ across cache flavors and forcing a
uniform interface re-introduces the complexity P7 stripped from the
kernel. Concrete adapters wrap backing storage; tracking of hit/miss
counts stays with the existing wrapper closures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.2: Registry (test-first)

**Files:**
- Create: `test/test_cache_registry.py`
- Create: `pyphi/cache/registry.py`

- [ ] **Step 1: Write failing test for registry contract**

Create `test/test_cache_registry.py`:

```python
"""Tests for the process-local cache registry."""

from __future__ import annotations

import pytest

from pyphi.cache import registry as reg
from pyphi.cache.cache_utils import _CacheInfo
from pyphi.cache.policy import _DictCacheAdapter


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Snapshot the registry around each test."""
    snapshot = dict(reg._registry)
    reg._registry.clear()
    yield
    reg._registry.clear()
    reg._registry.update(snapshot)


def _make_adapter(name: str, contents: dict | None = None, stats=(0, 0)):
    backing = contents if contents is not None else {}
    return _DictCacheAdapter(name=name, backing=backing, stats=lambda: stats)


def test_register_and_info_roundtrip():
    adapter = _make_adapter("test.x", {"k": "v"}, stats=(1, 2))
    reg.register(adapter)
    info = reg.info()
    assert "test.x" in info
    assert info["test.x"] == _CacheInfo(1, 2, 1)


def test_clear_one_clears_only_that_cache():
    a = _make_adapter("test.a", {"k1": 1})
    b = _make_adapter("test.b", {"k2": 2})
    reg.register(a)
    reg.register(b)
    reg.clear("test.a")
    assert a.backing == {}
    assert b.backing == {"k2": 2}


def test_clear_all_clears_every_registered_cache():
    a = _make_adapter("test.a", {"k": 1})
    b = _make_adapter("test.b", {"k": 2})
    reg.register(a)
    reg.register(b)
    reg.clear_all()
    assert a.backing == {}
    assert b.backing == {}


def test_unregister_removes_entry():
    a = _make_adapter("test.a")
    reg.register(a)
    reg.unregister("test.a")
    assert "test.a" not in reg.info()


def test_duplicate_registration_replaces_silently():
    """Module reloads / fixture re-registration should not error."""
    a1 = _make_adapter("test.a", {"k1": 1})
    a2 = _make_adapter("test.a", {"k2": 2})
    reg.register(a1)
    reg.register(a2)
    assert reg.info()["test.a"].currsize == 1  # a2 wins
    assert a1 not in reg._registry.values()


def test_clear_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        reg.clear("test.nonexistent")
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_cache_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.cache.registry'`

- [ ] **Step 3: Implement registry**

Create `pyphi/cache/registry.py`:

```python
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
    """Remove a registration. KeyError if name unknown."""
    del _registry[name]


def info() -> dict[str, _CacheInfo]:
    """Return per-cache stats for every registered policy."""
    return {name: policy.info() for name, policy in _registry.items()}


def clear_all() -> None:
    """Clear every registered cache."""
    for policy in _registry.values():
        policy.clear()


def clear(name: str) -> None:
    """Clear one named cache. KeyError if unknown."""
    _registry[name].clear()
```

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_cache_registry.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add test/test_cache_registry.py pyphi/cache/registry.py
git commit -m "$(cat <<'EOF'
P9: add process-local cache registry

register / unregister / info / clear / clear_all over a dict of
CachePolicy adapters. Duplicate registration replaces silently;
unknown names raise KeyError on clear/unregister.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.3: Re-export from `pyphi.cache`

**Files:**
- Modify: `pyphi/cache/__init__.py`

- [ ] **Step 1: Write failing test for the public surface**

Append to `test/test_cache_registry.py`:

```python
def test_pyphi_cache_re_exports_registry_surface():
    """Top-level pyphi.cache exposes info / clear_all / clear / register."""
    from pyphi import cache

    assert callable(cache.info)
    assert callable(cache.clear_all)
    assert callable(cache.clear)
    assert callable(cache.register)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_cache_registry.py::test_pyphi_cache_re_exports_registry_surface -v`
Expected: FAIL with `AttributeError: module 'pyphi.cache' has no attribute 'info'`

- [ ] **Step 3: Add re-exports to `pyphi/cache/__init__.py`**

At the top of `pyphi/cache/__init__.py` (after existing imports), add:

```python
from .registry import clear as clear  # noqa: PLC0414
from .registry import clear_all as clear_all  # noqa: PLC0414
from .registry import info as info  # noqa: PLC0414
from .registry import register as register  # noqa: PLC0414
from .registry import unregister as unregister  # noqa: PLC0414
```

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_cache_registry.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/cache/__init__.py test/test_cache_registry.py
git commit -m "$(cat <<'EOF'
P9: re-export registry surface from pyphi.cache

Users get pyphi.cache.info() / clear_all() / clear(name) without
reaching into pyphi.cache.registry.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Kernel `_memoize` integration + memory bound

### Task 2.1: Wrap kernel `_caches` in registered adapter

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Modify: `test/test_core_repertoire_algebra.py`

- [ ] **Step 1: Write failing test for registry visibility**

Append to `test/test_core_repertoire_algebra.py`:

```python
def test_kernel_caches_appear_in_registry():
    """Each kernel-memoized function registers a policy under kernel.<name>."""
    from pyphi import cache as cache_module
    from pyphi.cache import registry as reg

    # Trigger at least one kernel function to ensure decoration ran.
    from pyphi.core import repertoire_algebra as ra
    assert "kernel._single_node_cause_repertoire" in reg._registry, (
        f"expected kernel adapter in registry, got: {list(reg._registry.keys())}"
    )

    info = cache_module.info()
    assert "kernel._single_node_cause_repertoire" in info


def test_kernel_clear_via_registry_clears_kernel_cache():
    """pyphi.cache.clear('kernel.<name>') empties that kernel cache."""
    from pyphi import cache as cache_module
    from pyphi.core import repertoire_algebra as ra

    # Force at least one cache hit on the test fixture.
    cs = _basic_candidate_system()  # existing helper in this file
    ra._single_node_cause_repertoire(cs, 0, frozenset({0, 1}))
    name = "kernel._single_node_cause_repertoire"
    assert cache_module.info()[name].currsize >= 1

    cache_module.clear(name)
    assert cache_module.info()[name].currsize == 0
```

(If `_basic_candidate_system` doesn't exist in the test file, replace with whatever the existing fixture pattern is — there is one because `test_core_repertoire_algebra.py` already exercises kernel functions. Check the file before writing.)

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_core_repertoire_algebra.py::test_kernel_caches_appear_in_registry -v`
Expected: FAIL — registry doesn't have `kernel.*` entries.

- [ ] **Step 3: Refactor kernel cache structure**

In `pyphi/core/repertoire_algebra.py`:

Change the module-level `_caches` from `dict[str, dict[tuple, Any]]` to a wrapper that registers itself:

```python
# One adapter per memoized function name.
_caches: dict[str, dict[tuple, Any]] = {}
_kernel_stats: dict[str, list[int]] = {}  # [hits, misses] per function name
```

(Keeping `_caches` as `dict[str, dict[tuple, Any]]` so `_evict` still works; adding parallel `_kernel_stats` for the adapter's `stats` callable.)

In `_memoize`, after creating the cache dict, register an adapter:

```python
def _memoize(fn: Callable) -> Callable:
    cache = _caches.setdefault(fn.__name__, {})
    stats = _kernel_stats.setdefault(fn.__name__, [0, 0])  # [hits, misses]

    # Register adapter for this kernel cache (idempotent; replaces on reload).
    from pyphi.cache.policy import _DictCacheAdapter
    from pyphi.cache.registry import register as _register_policy
    _register_policy(
        _DictCacheAdapter(
            name=f"kernel.{fn.__name__}",
            backing=cache,
            stats=lambda s=stats: (s[0], s[1]),
        )
    )

    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        cs_id = id(cs)
        key = (cs_id, args)
        if cs_id not in _observers:
            _observers[cs_id] = cs
            weakref.finalize(cs, _evict, cs_id)
        if key in cache:
            stats[0] += 1
            return cache[key]
        stats[1] += 1
        result = fn(cs, *args)
        cache[key] = result
        return result

    return wrapper
```

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_core_repertoire_algebra.py -v`
Expected: all PASS, including the two new tests.

- [ ] **Step 5: Run golden + property suites to verify no regression**

Run in foreground: `uv run pytest -x test/test_golden_regression.py test/test_invariants.py test/test_core_repertoire_algebra.py -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "$(cat <<'EOF'
P9: register kernel _memoize caches in the cache registry

Each memoized function exposes a CachePolicy adapter under
'kernel.<fn_name>'. Hit/miss counters added; existing weakref-finalizer
eviction unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.2: Memory bound on kernel cache

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py`
- Modify: `test/test_core_repertoire_algebra.py`

- [ ] **Step 1: Write failing test for memory bound**

Append to `test/test_core_repertoire_algebra.py`:

```python
def test_kernel_cache_respects_memory_full(monkeypatch):
    """When memory_full() returns True, kernel cache stops adding entries."""
    from pyphi import cache as cache_module
    from pyphi.cache import cache_utils
    from pyphi.core import repertoire_algebra as ra

    cache_module.clear_all()  # baseline
    cs = _basic_candidate_system()

    # Force memory_full to return True.
    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)

    # Two distinct args; both should compute but neither should be cached.
    ra._single_node_cause_repertoire(cs, 0, frozenset({0, 1}))
    ra._single_node_cause_repertoire(cs, 0, frozenset({1, 2}))

    info = cache_module.info()["kernel._single_node_cause_repertoire"]
    assert info.currsize == 0, (
        f"expected 0 cached entries when memory full, got {info.currsize}"
    )
    # Misses still count.
    assert info.misses >= 2
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_core_repertoire_algebra.py::test_kernel_cache_respects_memory_full -v`
Expected: FAIL — kernel doesn't check `memory_full()` yet.

- [ ] **Step 3: Add memory bound to wrapper**

In `pyphi/core/repertoire_algebra.py`, update the `_memoize` wrapper body:

```python
    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        from pyphi.cache.cache_utils import memory_full

        cs_id = id(cs)
        key = (cs_id, args)
        if cs_id not in _observers:
            _observers[cs_id] = cs
            weakref.finalize(cs, _evict, cs_id)
        if key in cache:
            stats[0] += 1
            return cache[key]
        stats[1] += 1
        result = fn(cs, *args)
        if not memory_full():
            cache[key] = result
        return result
```

(Import is local-in-function to avoid circular import at module load. `memory_full()` returns False quickly via psutil.)

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_core_repertoire_algebra.py::test_kernel_cache_respects_memory_full -v`
Expected: PASS.

- [ ] **Step 5: Run full test suite to confirm no regression**

Run in foreground: `uv run pytest -x test/test_core_repertoire_algebra.py test/test_golden_regression.py -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/test_core_repertoire_algebra.py
git commit -m "$(cat <<'EOF'
P9: bound kernel _memoize by MAXIMUM_CACHE_MEMORY_PERCENTAGE

Closes the unbounded-growth hole in long-running notebook sessions.
Per-miss psutil.memory_percent() check (~10us); if profiling shows it
matters in tight loops, amortize to every Nth miss.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Module-level `@cache(...)` integration

### Task 3.1: Register adapters from the legacy `cache()` decorator

**Files:**
- Modify: `pyphi/cache/__init__.py`
- Modify: `test/test_cache_registry.py`

- [ ] **Step 1: Write failing test that combinatorial caches show up in registry**

Append to `test/test_cache_registry.py`:

```python
def test_module_level_cache_decorator_registers_adapter():
    """A function decorated with @cache(...) registers a policy."""
    # Trigger import to ensure decoration ran.
    from pyphi import combinatorics  # noqa: F401
    from pyphi import cache as cache_module

    info = cache_module.info()
    # combinations_with_nonempty_intersection_by_order is one such function.
    expected_name = (
        "pyphi.combinatorics.combinations_with_nonempty_intersection_by_order"
    )
    assert expected_name in info, (
        f"expected {expected_name} in registry, got keys: {list(info.keys())}"
    )
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_cache_registry.py::test_module_level_cache_decorator_registers_adapter -v`
Expected: FAIL — module-level decorator doesn't register yet.

- [ ] **Step 3: Modify `cache()` decorator to register an adapter**

In `pyphi/cache/__init__.py`, modify the `decorating_function` inside `cache()`:

```python
    def decorating_function(user_function, hits=0, misses=0):
        # ... existing wrapper definitions unchanged ...

        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]

        # Register a CachePolicy adapter under the qualified function name.
        from .policy import _DictCacheAdapter
        from .registry import register as _register_policy

        _register_policy(
            _DictCacheAdapter(
                name=f"{user_function.__module__}.{user_function.__qualname__}",
                backing=cache,
                # ``cache_info`` returns a _CacheInfo; we want (hits, misses).
                stats=lambda: (cache_info().hits, cache_info().misses),
            )
        )

        return update_wrapper(wrapper, user_function)
```

(The `stats` closure goes through `cache_info()` so we don't need to chase the `nonlocal` counters — they're in the wrapper's enclosing scope, and `cache_info` already reads them correctly.)

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_cache_registry.py -v`
Expected: all PASS, including new test.

- [ ] **Step 5: Run full fast lane to confirm no regression**

Run in foreground: `uv run pytest -x test/test_partition.py test/test_combinatorics.py test/test_distribution.py test/test_golden_regression.py -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add pyphi/cache/__init__.py test/test_cache_registry.py
git commit -m "$(cat <<'EOF'
P9: module-level @cache(...) decorator registers a CachePolicy

Combinatorial caches in partition.py / distribution.py /
combinatorics.py now appear in pyphi.cache.info() under
'<module>.<qualname>' names.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Instance-level `DictCache` integration

### Task 4.1: `DictCache` accepts a name and registers when given one

**Files:**
- Modify: `pyphi/cache/__init__.py` (the `DictCache` class)
- Modify: `test/test_cache_registry.py`

- [ ] **Step 1: Write failing test for named DictCache**

Append to `test/test_cache_registry.py`:

```python
def test_dict_cache_with_name_registers():
    from pyphi import cache as cache_module
    from pyphi.cache import DictCache

    DictCache(name="test.dict_cache.named")
    assert "test.dict_cache.named" in cache_module.info()


def test_dict_cache_without_name_does_not_register():
    """Default behavior: anonymous DictCache instances stay out of the registry."""
    from pyphi import cache as cache_module
    from pyphi.cache import DictCache

    before = set(cache_module.info().keys())
    DictCache()  # no name
    after = set(cache_module.info().keys())
    assert before == after
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_cache_registry.py -v -k dict_cache`
Expected: FAIL — `DictCache.__init__` doesn't accept `name`.

- [ ] **Step 3: Add `name` parameter to `DictCache`**

In `pyphi/cache/__init__.py`, modify `DictCache`:

```python
class DictCache:
    """A generic dictionary-based cache.

    If ``name`` is provided, the cache registers itself with the cache
    registry on construction. Anonymous instances stay out of the
    registry — useful for short-lived helpers in tests.
    """

    def __init__(self, name: str | None = None):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.name = name
        if name is not None:
            from .policy import _DictCacheAdapter
            from .registry import register as _register_policy

            _register_policy(
                _DictCacheAdapter(
                    name=name,
                    backing=self.cache,
                    stats=lambda: (self.hits, self.misses),
                )
            )

    # ... rest of class unchanged ...
```

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_cache_registry.py -v -k dict_cache`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/cache/__init__.py test/test_cache_registry.py
git commit -m "$(cat <<'EOF'
P9: DictCache opt-in registry registration via name parameter

Anonymous DictCache instances stay out of the registry; production
sites pass an explicit name in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.2: Wire `Network.purview_cache` with per-instance name

**Files:**
- Modify: `pyphi/network.py`
- Modify: `test/test_cache_registry.py`

- [ ] **Step 1: Write failing test for per-instance Network registration**

Append to `test/test_cache_registry.py`:

```python
def test_multiple_networks_register_independent_purview_caches():
    """Two Network instances appear under distinct registry names."""
    from pyphi import cache as cache_module
    from pyphi import examples

    n1 = examples.basic_network()
    n2 = examples.basic_network()

    info = cache_module.info()
    n1_keys = [k for k in info if k.startswith(f"network.{id(n1)}.")]
    n2_keys = [k for k in info if k.startswith(f"network.{id(n2)}.")]

    assert n1_keys, "n1 purview cache not registered"
    assert n2_keys, "n2 purview cache not registered"
    assert n1_keys != n2_keys, "two networks collided on the same registry name"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_cache_registry.py -v -k multiple_networks`
Expected: FAIL — Network doesn't pass a name.

- [ ] **Step 3: Update `pyphi/network.py:Network.__init__`**

Find the line `self.purview_cache = purview_cache or cache.PurviewCache()` (around line 96).

Change to:

```python
        self.purview_cache = purview_cache or cache.PurviewCache(
            name=f"network.{id(self)}.purview_cache",
        )
```

Also: `PurviewCache` extends `DictCache` so the `name` parameter is inherited. No `PurviewCache` change needed unless its `__init__` overrides — verify by reading `pyphi/cache/__init__.py:170` (`class PurviewCache(DictCache)`); if it has no `__init__`, no change needed.

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_cache_registry.py -v -k multiple_networks`
Expected: PASS.

- [ ] **Step 5: Run network tests to confirm no regression**

Run: `uv run pytest -x test/test_compute_network.py test/test_network.py -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add pyphi/network.py test/test_cache_registry.py
git commit -m "$(cat <<'EOF'
P9: Network.purview_cache registers under a per-instance name

Multiple Network instances coexist in real workflows (param sweeps,
comparisons) — per-instance names give correct stats. Tradeoff:
pyphi.cache.info() is longer in multi-network sessions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4.3: Wire `_ObjectCache` in `jsonify.py`

**Files:**
- Modify: `pyphi/jsonify.py`
- Modify: `test/test_cache_registry.py`

- [ ] **Step 1: Write failing test for jsonify object cache registration**

Append to `test/test_cache_registry.py`:

```python
def test_jsonify_object_cache_registered():
    """jsonify._ObjectCache registers under a stable singleton name."""
    from pyphi import cache as cache_module
    from pyphi import jsonify  # noqa: F401  # ensure import

    # Construct one to trigger registration if it's per-instance, or rely
    # on import-time singleton registration.
    info = cache_module.info()
    assert any(k.startswith("jsonify.") for k in info), (
        f"no jsonify cache registered; keys: {list(info.keys())}"
    )
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_cache_registry.py -v -k jsonify`
Expected: FAIL — `_ObjectCache` doesn't register.

- [ ] **Step 3: Locate and update `_ObjectCache`**

Read `pyphi/jsonify.py` around line 316. Identify where `_ObjectCache` is defined and where it's instantiated. Two cases:

- If `_ObjectCache` extends `DictCache`, instantiate with a name: `_ObjectCache(name="jsonify.object_cache")`.
- If `_ObjectCache` is its own class with its own dict, either subclass DictCache (preferred) or add the same registration pattern manually.

Per-instance vs singleton: `_ObjectCache` instances are constructed inside `to_json` calls. They're transient. We have two choices:

(a) Don't register transient `_ObjectCache` instances — they're scoped to a single JSON encode. Stats wouldn't be meaningful across calls.
(b) Register with a per-call name like `f"jsonify.object_cache.{id(self)}"`.

Choose (a) — transient caches don't belong in process-level stats. Add a docstring noting this. The test above asserts ANY `jsonify.` key appears; revise to assert that no transient registration leaks instead.

Revise the test (replace the previous `test_jsonify_object_cache_registered`):

```python
def test_jsonify_object_caches_do_not_leak_into_registry():
    """Transient _ObjectCache instances inside to_json should NOT register —
    they're per-call scratch caches, not process-level state."""
    from pyphi import cache as cache_module
    from pyphi import examples
    from pyphi import jsonify

    before = set(cache_module.info().keys())

    # Trigger a jsonify path that uses _ObjectCache.
    network = examples.basic_network()
    jsonify.dumps(network)

    after = set(cache_module.info().keys())
    leaked = {k for k in (after - before) if k.startswith("jsonify.")}
    assert not leaked, f"transient jsonify caches leaked into registry: {leaked}"
```

This test PASSES as soon as `_ObjectCache` does NOT register itself (i.e., the default behavior of an anonymous `DictCache`). Verify this is the case by reading `_ObjectCache` and confirming it either is a plain `DictCache` (with no name) or subclasses `DictCache` and doesn't pass a name.

If `_ObjectCache` has its own `__init__` that doesn't call `DictCache.__init__(name=...)`, this is already true — no change needed.

- [ ] **Step 4: Run test to verify passes**

Run: `uv run pytest test/test_cache_registry.py -v -k jsonify`
Expected: PASS (no `jsonify.` key in registry).

- [ ] **Step 5: Commit**

```bash
git add test/test_cache_registry.py
git commit -m "$(cat <<'EOF'
P9: pin jsonify _ObjectCache as transient (not registered)

Per-call scratch cache for object-graph deduplication during JSON
encoding. Process-level stats would be misleading for a transient
cache; explicitly assert the non-registration in a regression test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Threading documentation

### Task 5.1: Document the single-threaded-per-process assumption

**Files:**
- Modify: `pyphi/cache/__init__.py`
- Modify: `pyphi/core/repertoire_algebra.py`

- [ ] **Step 1: Add module docstring to `pyphi/cache/__init__.py`**

Replace the existing `"""Memoization and caching utilities."""` line with:

```python
"""Memoization and caching utilities.

Threading
---------
Caches in PyPhi are NOT thread-safe. PyPhi assumes process-isolated
parallelism (Ray-based), where each worker has its own interpreter and
its own copy of every cache. Do not share cache instances across
threads.

If a future parallelism model uses shared memory (free-threaded Python,
asyncio with shared state, etc.), this module will need locks.

Public surface
--------------
- ``info()``: dict of name -> _CacheInfo across every registered cache.
- ``clear_all()``: clear every registered cache.
- ``clear(name)``: clear one cache by name.
- ``register(policy)``: register a CachePolicy adapter.

See :mod:`pyphi.cache.policy` for the CachePolicy Protocol and
:mod:`pyphi.cache.registry` for the registry implementation.
"""
```

- [ ] **Step 2: Mirror the warning in the kernel module**

In `pyphi/core/repertoire_algebra.py`, replace the existing module docstring:

```python
"""Stateless repertoire computation over CandidateSystem.

Layer 2 of the kernel. Functions take a CandidateSystem as the first
argument; results are memoized via a per-instance decorator that purges
when the CandidateSystem is garbage-collected.

Numerical bodies are ports of the corresponding Subsystem methods in
pyphi/subsystem.py. Parity tests guard equivalence.

Threading
---------
The kernel cache is NOT thread-safe — see :mod:`pyphi.cache` for the
process-isolated parallelism assumption.
"""
```

- [ ] **Step 3: Run pyright to confirm clean**

Run: `uv run pyright pyphi/cache pyphi/core/repertoire_algebra.py`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add pyphi/cache/__init__.py pyphi/core/repertoire_algebra.py
git commit -m "$(cat <<'EOF'
P9: document single-threaded-per-process cache assumption

Caches are not thread-safe; PyPhi relies on Ray's process-isolated
parallelism. Future shared-memory models will need locks. Spelled out
in pyphi.cache and pyphi.core.repertoire_algebra docstrings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: Dead-code removal

### Task 6.1: Delete `pyphi/cache/redis.py`

**Files:**
- Delete: `pyphi/cache/redis.py`
- Modify: `pyphi/cache/__init__.py` (no remaining redis imports — verify)

- [ ] **Step 1: Verify no live references**

Run: `cd ../pyphi-p7-kernel-rewrite && rg -n "from .*\.cache import.*[Rr]edis|pyphi\.cache\.redis|RedisCache" --type py 2>/dev/null`
Expected: zero matches inside `pyphi/`. Tests OK to ignore — they don't exist.

- [ ] **Step 2: Delete the file**

```bash
rm pyphi/cache/redis.py
```

- [ ] **Step 3: Run import smoke test**

Run: `uv run python -c "import pyphi; import pyphi.cache; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add -A pyphi/cache/redis.py
git commit -m "$(cat <<'EOF'
P9: remove pyphi/cache/redis.py — dead scaffolding

RedisCache class was defined but never instantiated anywhere in
pyphi/, test/, or docs/. The MICECache wrapper that historically tied
it to MICE caching no longer exists. A future distributed-cache feature
is more cleanly built fresh against the CachePolicy Protocol than
on top of half-built scaffolding.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.2: Remove `joblib_memory`

**Files:**
- Modify: `pyphi/cache/__init__.py`

- [ ] **Step 1: Remove the joblib_memory line and imports**

In `pyphi/cache/__init__.py`:

- Delete the line `import joblib`
- Delete the line `from pyphi import constants` (if `constants` becomes unused — verify)
- Delete the line:
  ```python
  joblib_memory = joblib.Memory(location=constants.DISK_CACHE_LOCATION, verbose=0)
  ```

- [ ] **Step 2: Verify pyright + smoke test**

Run: `uv run pyright pyphi/cache && uv run python -c "from pyphi import cache; print('ok')"`
Expected: 0 errors, `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyphi/cache/__init__.py
git commit -m "$(cat <<'EOF'
P9: remove unused joblib_memory disk cache instance

Defined but with zero call sites in pyphi/ or test/. Users who relied
on it for ad-hoc disk caching can construct their own joblib.Memory
in two lines.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.3: Remove `REDIS_CACHE` and `REDIS_CONFIG` from config

**Files:**
- Modify: `pyphi/conf.py`
- Modify: `pyphi/conf.pyi`
- Modify: `pyphi_config.yml` (if these keys appear there)

- [ ] **Step 1: Locate and remove from `pyphi/conf.py`**

Read `pyphi/conf.py` lines 660-680 to confirm the `REDIS_CACHE = Option(...)` and `REDIS_CONFIG = Option(...)` blocks and the docstring before them. Delete both `Option` declarations including their docstrings.

- [ ] **Step 2: Remove from `pyphi/conf.pyi`**

Read `pyphi/conf.pyi` lines 80-100 to confirm. Delete:

```
REDIS_CACHE: bool
REDIS_CONFIG: dict[str, Any]
```

- [ ] **Step 3: Check `pyphi_config.yml`**

Run: `grep -n "REDIS" pyphi_config.yml`
If matches: delete those lines.
If empty: skip.

- [ ] **Step 4: Run config tests + pyright**

Run: `uv run pytest -x test/test_config.py -q && uv run pyright pyphi/conf.py pyphi/conf.pyi`
Expected: green tests, 0 pyright errors.

- [ ] **Step 5: Commit**

```bash
git add pyphi/conf.py pyphi/conf.pyi pyphi_config.yml
git commit -m "$(cat <<'EOF'
P9: remove REDIS_CACHE and REDIS_CONFIG dead config keys

Neither was read by any live code path. Per the no-back-compat dictum,
hard removal — users with these keys in pyphi_config.yml can delete
them; YAML loaders will warn on unknown keys, not error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.4: Remove the stale `MICECache` Sphinx alias

**Files:**
- Modify: `docs/conf.py`

- [ ] **Step 1: Remove the alias**

Read `docs/conf.py` line 302. Delete the line:

```
.. |MICECache| replace:: :class:`~pyphi.cache.MICECache`
```

(Search for any uses of `|MICECache|` elsewhere first — `rg -n "MICECache" docs/ --type rst` and `--type py`. If any, replace with descriptive text.)

- [ ] **Step 2: Verify docs build**

If a docs build is straightforward in this repo: `make -C docs html` and confirm no warnings about `|MICECache|`. If docs aren't built routinely, skip — the substitution is gone, build will be clean by construction.

- [ ] **Step 3: Commit**

```bash
git add docs/conf.py
git commit -m "$(cat <<'EOF'
P9: remove stale |MICECache| Sphinx alias

The MICECache class it referenced no longer exists.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7: Acceptance gates

### Task 7.1: Architectural integration test

**Files:**
- Create: `test/test_cache_integration.py`

- [ ] **Step 1: Write integration test that exercises a full SIA path and checks registry coverage**

Create `test/test_cache_integration.py`:

```python
"""Integration test: after running an IIT 4.0 SIA computation, confirm
that the kernel + combinatorial caches all show up in the registry."""

from __future__ import annotations

import pytest

from pyphi import cache
from pyphi import examples
from pyphi.formalism.queries import sia


@pytest.fixture(autouse=True)
def _clear_caches():
    cache.clear_all()
    yield
    cache.clear_all()


def test_sia_run_populates_kernel_and_combinatorial_caches():
    """End-to-end: a SIA run touches both kernel and combinatorial caches."""
    network = examples.basic_network()
    subsystem = examples.basic_subsystem()
    _ = sia(subsystem)

    info = cache.info()

    kernel_keys = [k for k in info if k.startswith("kernel.")]
    combinatorial_keys = [
        k for k in info
        if k.startswith("pyphi.partition.")
        or k.startswith("pyphi.combinatorics.")
        or k.startswith("pyphi.distribution.")
    ]
    network_keys = [k for k in info if k.startswith(f"network.{id(network)}.")]

    assert kernel_keys, f"no kernel cache entries; got: {sorted(info)}"
    assert combinatorial_keys, f"no combinatorial cache entries; got: {sorted(info)}"
    assert network_keys, f"no network purview cache for this network; got: {sorted(info)}"

    # All three types have non-zero size after a SIA run.
    assert any(info[k].currsize > 0 for k in kernel_keys)
```

- [ ] **Step 2: Run test**

Run: `uv run pytest test/test_cache_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_cache_integration.py
git commit -m "$(cat <<'EOF'
P9: integration test pinning end-to-end registry coverage

After running a full IIT 4.0 SIA, confirm kernel, combinatorial, and
network caches are all visible through pyphi.cache.info().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7.2: Full acceptance suite

- [ ] **Step 1: Golden 17/17**

Run in foreground: `uv run pytest test/test_golden_regression.py -v`
Expected: 17 passed.

- [ ] **Step 2: Hypothesis fast lane**

Run in foreground: `uv run pytest test/test_invariants_hypothesis.py -v`
Expected: green (≥21 properties).

- [ ] **Step 3: Models layering**

Run: `uv run pytest test/test_models_layering.py test/test_core_layering.py -v`
Expected: green.

- [ ] **Step 4: Pyright clean**

Run: `uv run pyright pyphi/cache pyphi/core/repertoire_algebra.py pyphi/network.py`
Expected: 0 errors.

- [ ] **Step 5: Ruff clean**

Run: `uv run ruff check pyphi/cache pyphi/core/repertoire_algebra.py`
Expected: 0 errors.

### Task 7.3: Performance check

- [ ] **Step 1: Time the golden suite before and after (rough sanity check)**

Run: `time uv run pytest test/test_golden_regression.py -q`
Expected: total wall time within 5% of pre-P9 baseline (P8 baseline was 15:04 for golden 17/17 — anything under 16:00 is acceptable).

If regression > 5%: profile the `memory_full()` per-miss cost. If that's the culprit, amortize to every Nth miss and rerun.

### Task 7.4: Changelog fragment

**Files:**
- Create: `changelog.d/p9-unified-cache.refactor.md`

- [ ] **Step 1: Write the fragment**

```markdown
The ``pyphi.cache`` module gained a uniform observability + control surface: ``pyphi.cache.info()`` returns per-cache statistics across the kernel ``_memoize`` caches, the module-level ``@cache(...)`` caches in ``pyphi.partition`` / ``pyphi.combinatorics`` / ``pyphi.distribution``, and the per-``Network`` purview caches; ``pyphi.cache.clear_all()`` and ``pyphi.cache.clear(name)`` clear them. The kernel cache now respects ``MAXIMUM_CACHE_MEMORY_PERCENTAGE`` (previously unbounded). Caches are not thread-safe — PyPhi assumes process-isolated parallelism (Ray-based); future shared-memory parallelism will need locks.

The unused ``RedisCache`` class, ``joblib_memory`` disk-cache instance, ``REDIS_CACHE`` and ``REDIS_CONFIG`` config keys, and the stale ``|MICECache|`` Sphinx alias have been removed. None had live call sites. Distributed caching is no longer supported; if you relied on the (never-wired) Redis scaffolding, you'll need to wait for a future feature that integrates with the new ``CachePolicy`` Protocol.
```

- [ ] **Step 2: Commit**

```bash
git add changelog.d/p9-unified-cache.refactor.md
git commit -m "$(cat <<'EOF'
P9: changelog fragment

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (run before declaring P9 complete)

- [ ] All acceptance gates green (golden 17/17, Hypothesis fast lane, layering tests, pyright, ruff)
- [ ] `pyphi.cache.info()` returns at least one entry for each flavor (kernel, combinatorial, network)
- [ ] `pyphi.cache.clear_all()` empties everything (verified by integration test)
- [ ] No call site references `RedisCache`, `joblib_memory`, `REDIS_CACHE`, `REDIS_CONFIG`, or `MICECache`
- [ ] Threading docstring present in `pyphi/cache/__init__.py` and `pyphi/core/repertoire_algebra.py`
- [ ] Changelog fragment in `changelog.d/`
- [ ] Self-review of every commit message: no roadmap-stage references inside docstrings/comments/changelog (P9 in commit subjects is fine — that's git metadata, not source)

## What's deferred

- Locks / thread-safety: P11 (parallelization redesign) territory.
- Re-introducing distributed cache: future task tracked in roadmap deferred-items registry.
- Removing legacy `_repertoire_cache` instance caches in `pyphi/subsystem.py`: bundled into P14 deletion.
- Per-call-site memory bounds (LRU): YAGNI.
