# P9 — Unified Cache: Design

**Status:** spec
**Date:** 2026-05-07
**Branch:** `feature/p9-unified-cache` (off `feature/p7-kernel-rewrite` tip)

## Motivation

PyPhi's caching is fragmented across four flavors that each solve a slightly different problem:

| Flavor | Used by | Backing | Eviction |
|---|---|---|---|
| Module-level `@cache(cache={}, maxmem=None)` (`pyphi/cache/__init__.py`) | `combinatorics.py`, `distribution.py`, `partition.py` | dict | manual |
| Instance-level `@cache.method("name")` + `DictCache` | `network.py` (`PurviewCache`), `jsonify.py` (`_ObjectCache`) | dict on `self` | per-instance |
| Kernel `@_memoize` (`pyphi/core/repertoire_algebra.py`) | every kernel repertoire/partition function | module dict keyed by `id(cs)` | weakref finalizer when CandidateSystem GC'd |
| `joblib.Memory` (`pyphi/cache/__init__.py:joblib_memory`) | nothing live (configured but unused inside the codebase) | disk | manual |

Plus dead code: `RedisCache`, `REDIS_CACHE` config, `REDIS_CONFIG` config — defined but with zero live call sites in `pyphi/`, `test/`, or `docs/`. The Sphinx alias `|MICECache|` in `docs/conf.py:302` references a class that doesn't exist.

The fragmentation has three concrete costs:

1. **No global view.** There's no `pyphi.cache.info()` that walks every cache and reports sizes / hit rates. Each cache reports its own stats through ad-hoc surfaces (`Subsystem._repertoire_cache.info()`, the wrapper's `cache_info()` attribute, etc.).
2. **No global clear.** Same problem: `pyphi.cache.clear_all()` doesn't exist. The kernel's `_memoize` has its own `clear_caches(cs=None)`; legacy `DictCache` has its own `clear()`; module-level `@cache(...)` exposes `cache_clear()` on each wrapper.
3. **Inconsistent memory bounds.** Legacy `cache()` honors `MAXIMUM_CACHE_MEMORY_PERCENTAGE` via `psutil`; kernel `_memoize` is unbounded (a long notebook session can grow `_caches` indefinitely); `DictCache` is unbounded.

Threading is also a latent concern (no cache uses locks; reads-then-writes are not atomic), but in current PyPhi this is moot — Ray-based parallelism is process-isolated, so each worker has its own caches and there is no shared mutation. We will document the assumption rather than introduce locks.

## Goals

1. **One observability surface.** `pyphi.cache.info()` returns a dict from cache name to `_CacheInfo(hits, misses, size)`. Walks every registered cache.
2. **One control surface.** `pyphi.cache.clear_all()` clears every registered cache. `pyphi.cache.clear(name)` clears one.
3. **Memory bound on the kernel cache.** The kernel `_memoize` honors `MAXIMUM_CACHE_MEMORY_PERCENTAGE` via the same `memory_full()` check the legacy `cache()` already uses.
4. **Document the single-threaded-per-process assumption.** Add a module-level docstring and a `CACHING.md` (or section in `pyphi/cache/__init__.py` docstring) stating that caches are not thread-safe and that PyPhi assumes process-isolated parallelism.
5. **Delete dead Redis code.** Remove `RedisCache`, `REDIS_CACHE`, `REDIS_CONFIG`, the conftest Redis fixture, the `CACHING.rst` Redis section, the benchmark Redis mode, and the stale `MICECache` Sphinx alias. The Redis machinery as it stands is half-built scaffolding (no decorator integration, no MICECache wrapper, no tests asserting it works); a future distributed-cache feature is more cleanly built fresh against whatever the abstraction looks like at that time. See the roadmap deferred-items registry for the future re-introduction note.

  Note: `joblib_memory` is RETAINED — `pyphi/metrics/distribution.py:_compute_hamming_matrix` uses `@joblib_memory.cache` for disk-persisted Hamming matrix caching (large matrices, infrequent recomputation). Initial spec drafted on a partial audit; corrected before Phase 6 execution.

## Non-goals

- **No unified decorator.** The four flavors solve different problems; forcing one decorator to span all four would re-introduce the complexity P7 deliberately stripped from the kernel. Each call site keeps the decorator that fits its lifecycle (module-level / instance-level / weakref-tracked).
- **No locks.** Threading safety is deferred until PyPhi adopts a parallelism model where it actually matters. We document the assumption.
- **No backend pluggability for caches that don't need it.** The combinatorial caches and the kernel cache stay in-memory only. There's no live consumer that needs Redis or disk for these.
- **No touching legacy `pyphi/subsystem.py` cache code.** That file is doomed in P14. The instance caches there (`_single_node_repertoire_cache`, etc.) will go with it.
- **No new config options.** The existing `MAXIMUM_CACHE_MEMORY_PERCENTAGE` and `CACHE_REPERTOIRES` / `CACHE_POTENTIAL_PURVIEWS` knobs cover what we need.

## Architecture

### `CachePolicy` protocol

A small structural Protocol in `pyphi/cache/policy.py`:

```python
class CachePolicy(Protocol):
    """A cache backend with a uniform stats / clear surface."""

    name: str

    def info(self) -> _CacheInfo: ...
    def clear(self) -> None: ...
```

The protocol intentionally does NOT include `get` / `put` / `key` — those live on each concrete policy because their signatures differ (the kernel `_memoize` uses `id(cs)` as the first key element; the legacy `cache()` uses `_make_key`; `DictCache` uses a custom `key()` method). Forcing a uniform `get/put` API across them would create exactly the kind of god-class-decorator we're avoiding.

### Registry

`pyphi/cache/registry.py`:

```python
_registry: dict[str, CachePolicy] = {}

def register(policy: CachePolicy) -> None: ...
def unregister(name: str) -> None: ...
def info() -> dict[str, _CacheInfo]: ...
def clear_all() -> None: ...
def clear(name: str) -> None: ...
```

Re-exported as `pyphi.cache.info()`, `pyphi.cache.clear_all()`, `pyphi.cache.clear(name)`.

### How each flavor registers

1. **Module-level `@cache(...)` decorator:** the decorator wraps the user function and ALSO registers a policy adapter under the function's qualified name (`f"{fn.__module__}.{fn.__qualname__}"`). The adapter exposes `info()` / `clear()` over the existing dict.
2. **`DictCache` instances:** `DictCache.__init__` takes an optional `name`; if provided, registers itself. The known long-lived instance cache (`Network.purview_cache`) registers under `f"network.{id(network)}.purview_cache"`. The transient `_ObjectCache` (constructed per `jsonify.loads()` call) does NOT register — process-level stats are misleading for per-call scratch caches. We default `name=None` for `DictCache` to avoid surprising users who construct ad-hoc instances; production sites opt in by passing an explicit name.
3. **Kernel `_memoize`:** the existing `_caches: dict[str, dict]` becomes `_caches: dict[str, _KernelCache]` where `_KernelCache` wraps the inner dict and exposes `info()` / `clear()`. Each function's cache registers under `f"kernel.{fn.__name__}"`.
4. **joblib disk cache:** if not removed in goal 5, register a thin adapter over `joblib_memory.store_backend.cache_size()` and `joblib_memory.clear()`.

### Memory bound on `_memoize`

Borrow `cache_utils.memory_full()`. In the kernel wrapper, before inserting into `cache`, check `memory_full()`; if yes, skip insertion (still return the computed result). This matches the existing legacy behavior.

The kernel's `_caches` is module-level, so we don't need a per-cache `full` flag; one global `memory_full()` call per miss is fine. For very tight loops this is a measurable cost (psutil call ~10μs); we can amortize by checking only every Nth miss if profiling shows it, but YAGNI for the initial cut.

### Threading documentation

Module-level docstring in `pyphi/cache/__init__.py`:

> Caches in PyPhi are not thread-safe. PyPhi assumes process-isolated
> parallelism (Ray-based), where each worker has its own interpreter and
> its own copy of every cache. Do not share cache instances across
> threads. If a future parallelism model uses shared memory, this module
> will need locks.

Mirror the warning at the top of `pyphi/core/repertoire_algebra.py` (kernel module).

## Decisions

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Unified decorator or unified observability? | Observability + control only | Decorators encode lifecycle which differs genuinely |
| 2 | Add memory bound to kernel `_memoize`? | Yes, via `memory_full()` | Closes the unbounded-growth hole |
| 3 | Add locks for thread safety? | No, document the assumption | Process-isolated parallelism makes locks unnecessary today |
| 4 | Delete `RedisCache` / `REDIS_CACHE` / `REDIS_CONFIG`? | Yes | Zero live call sites; per the no-back-compat memory |
| 5 | Delete `joblib_memory`? | No — has a live consumer | `pyphi/metrics/distribution.py:_compute_hamming_matrix` uses `@joblib_memory.cache` for disk-persisted Hamming matrix caching. Audit in initial spec missed this; corrected before Phase 6. |
| 6 | Register `_ObjectCache` instances? | No — they're transient (constructed per `jsonify.loads()` call) | Process-level stats are misleading for per-call scratch caches; pin the non-registration in a regression test instead |
| 7 | Register `Network.purview_cache` instances individually? | Yes — `f"network.{id(network)}.purview_cache"` | Multiple Networks coexist in real workflows (param sweeps, comparisons); per-instance names give correct stats. Tradeoff: longer `info()` output in multi-network sessions |
| 8 | Move `pyphi/cache/cache_utils.py:_make_key` somewhere? | No, leave it | Used internally by `cache()`; not worth moving |
| 9 | Keep `MAXIMUM_CACHE_MEMORY_PERCENTAGE` config? | Yes | Single knob covers all cache types; no reason to multiply knobs |
| 10 | New config `KERNEL_CACHE_ENABLED`? | No | Symmetric with not having `COMBINATORIAL_CACHE_ENABLED`; an escape hatch via `clear_all()` covers debugging |

## Risks

| Risk | Mitigation |
|---|---|
| `memory_full()` per-miss `psutil` cost regresses kernel performance | Benchmark golden fixture suite (acceptance gate). If ≥5% regression, amortize the check (every 16th miss) — but file an issue, don't silently change behavior |
| Registry name collisions across calls cause stats to overwrite | Use `f"{module}.{qualname}"` for module-level registrations; explicit names for instance caches; raise on duplicate registration in tests; warn at runtime |
| Removing `joblib_memory` breaks a user's out-of-tree script | Document removal in changelog; users can construct their own `joblib.Memory(location=...)` in two lines |
| Removing `REDIS_CACHE` / `REDIS_CONFIG` breaks user `pyphi_config.yml` files | Per the no-back-compat memory: hard removal, document in changelog. Users can delete the keys |
| Future need for distributed cache becomes harder without the scaffolding | The scaffolding doesn't actually save much — the connection setup is trivial; the integration with the cache abstraction is the real work, and that's done against whatever exists at that time. Tracked in the roadmap deferred-items registry |
| Adding `register=True` default to `DictCache` surprises subclassers | Default to `register=False`; only the two production sites opt in |

## Acceptance criteria

1. `pyphi.cache.info()` returns a dict covering: every `@cache(...)` decorated function, the kernel `_memoize` caches, the two production `DictCache` instances. All values are `_CacheInfo` namedtuples.
2. `pyphi.cache.clear_all()` clears every registered cache; `pyphi.cache.info()` afterward shows all `currsize == 0`.
3. Kernel `_memoize` honors `MAXIMUM_CACHE_MEMORY_PERCENTAGE` — verified by a unit test that monkeypatches `memory_full()` to return `True` and confirms the kernel cache stops growing.
4. Threading assumption documented in `pyphi/cache/__init__.py` and `pyphi/core/repertoire_algebra.py`.
5. `RedisCache`, `REDIS_CACHE`, `REDIS_CONFIG`, the `MICECache` Sphinx alias, the conftest Redis fixture, the `CACHING.rst` Redis section, and the benchmark Redis cache mode are removed. `joblib_memory` is retained (has a live consumer). Ruff + pyright clean.
6. Golden 17/17 pass unchanged. Hypothesis fast lane green. No performance regression > 5% on the golden suite (measured via existing harness).
7. Changelog fragment in `changelog.d/p9-unified-cache.refactor.md` describes the new public surface and the dead-code removals.

## Test strategy

- **Unit tests for the registry** (`test/test_cache_registry.py` new): register, unregister, info, clear, clear_all, duplicate-registration detection.
- **Unit test for kernel memory bound** (extends `test/test_core_repertoire_algebra.py`): monkeypatch `memory_full()`, exercise a kernel function, assert cache stays bounded.
- **Architectural test**: `pyphi.cache.info()` lists ≥ N entries (where N is the count of registered call sites we know about). Soft assertion guards against accidental decoupling.
- **Golden regression**: 17/17 unchanged.
- **Changelog**: single `.refactor.md` fragment.

## What does NOT happen in P9 (deferred)

- **Locks / thread safety.** Deferred to a future parallelism redesign (P11 may surface this; if so, it lands there).
- **Per-call-site memory bounds** (per-cache `maxsize`, LRU eviction). Not requested; YAGNI.
- **Distributed / cross-process caching.** Removed today; future re-introduction tracked in the roadmap deferred-items registry. When it returns, the integration target is the `CachePolicy` Protocol introduced here, not a half-built scaffold.
- **Reorganizing `pyphi/combinatorics.py` cache call sites.** P15 cleanup, separate concern.
- **Touching `pyphi/subsystem.py` cache decorators.** Doomed in P14.
- **Removing `CACHE_REPERTOIRES` / `CACHE_POTENTIAL_PURVIEWS` config keys.** They're still meaningful knobs for the live caches.
