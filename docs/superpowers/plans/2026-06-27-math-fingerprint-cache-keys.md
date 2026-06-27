# Math-fingerprint cache keys — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repertoire kernel cache and the potential-purview cache reuse results across mathematically-equivalent `System`/`Substrate` objects (including label-distinct and re-constructed ones) by keying on a label-free content digest instead of object identity.

**Architecture:** Add a reusable `ContentCache` (content-addressed dict + refcounted GC-time eviction) in `pyphi/cache/content.py`. Add `blake2b-256` fingerprints: `Substrate._cm_fingerprint` (connectivity only) and `Substrate._math_fingerprint` / `System._math_fingerprint` (full label-free math identity). Re-key the kernel `_memoize` cache on the `System` fingerprint and the purview cache on the `cm` fingerprint.

**Tech Stack:** Python 3.13, numpy, `hashlib.blake2b`, `weakref.finalize`, pytest + Hypothesis, uv.

## Global Constraints

- **No φ/α value may change.** Every golden stays byte-identical; verify with `uv run --all-extras pytest` (no path argument). Never `--no-verify`.
- **Digest:** `hashlib.blake2b(digest_size=32)` (256-bit). Fingerprints exclude `node_labels` and `state_space` label strings.
- **Eviction is a strict superset of today's:** prompt release when the last carrier of a fingerprint is GC'd, plus cross-object reuse while equivalents are alive. The `memory_full()` insert gate, `cache_potential_purviews` gate, `clear_caches()`, and `clear_system_caches_after_computing_sia` all keep working.
- **Commit trailer on every commit:**
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- **Stage only your own files** (concurrent instances commit to `main`); never `git add -A`. Ask before any `git push`.

---

## File structure

- `pyphi/cache/content.py` (new) — `ContentCache` class; the only home for the refcount + finalize logic.
- `pyphi/cache/__init__.py` — re-export `ContentCache`.
- `pyphi/substrate.py` — `Substrate._cm_fingerprint`, `Substrate._math_fingerprint`; migrate `potential_purviews` to a module-level `ContentCache`.
- `pyphi/system.py` — `System._math_fingerprint`.
- `pyphi/core/repertoire_algebra.py` — re-key `_memoize` onto `ContentCache`; rewrite `_evict`/`clear_caches`/`cache_info`.
- `test/cache/test_content_cache.py` (new) — `ContentCache` unit + eviction tests.
- `test/test_fingerprint.py` (new) — fingerprint correctness (unit + Hypothesis).
- `test/core/` — cross-object reuse tests for the kernel cache.

---

## Task 1: `ContentCache` primitive

**Files:**
- Create: `pyphi/cache/content.py`
- Modify: `pyphi/cache/__init__.py` (re-export)
- Test: `test/cache/test_content_cache.py`

**Interfaces:**
- Produces: `class ContentCache` with
  - `__init__(self, name: str)`
  - `observe(self, source: Any, fingerprint: bytes) -> None`
  - `get_or_compute(self, fingerprint: bytes, args: tuple, compute: Callable[[], Any], *, store: bool = True) -> Any`
  - `evict(self, fingerprint: bytes) -> None`
  - `clear(self) -> None`
  - `size` property returning `int`
  - attributes `hits: int`, `misses: int`

- [ ] **Step 1: Write the failing tests**

Create `test/cache/test_content_cache.py`:

```python
import gc

from pyphi.cache.content import ContentCache


def test_reuse_across_distinct_objects_with_same_fingerprint():
    cache = ContentCache("test.reuse")
    calls = []

    def compute():
        calls.append(1)
        return "value"

    a, b = object(), object()  # distinct objects, shared fingerprint
    fp = b"fp-shared"
    cache.observe(a, fp)
    cache.observe(b, fp)
    r1 = cache.get_or_compute(fp, (1,), compute)
    r2 = cache.get_or_compute(fp, (1,), compute)
    assert r1 == r2 == "value"
    assert len(calls) == 1  # computed once, reused
    assert cache.hits == 1 and cache.misses == 1


def test_refcount_eviction_only_when_last_carrier_dies():
    cache = ContentCache("test.evict")
    fp = b"fp-x"

    class Box:
        pass

    a, b = Box(), Box()
    cache.observe(a, fp)
    cache.observe(b, fp)
    cache.get_or_compute(fp, (), lambda: 42)
    assert cache.size == 1

    del a
    gc.collect()
    assert cache.size == 1  # b still alive -> entry survives

    del b
    gc.collect()
    assert cache.size == 0  # last carrier gone -> purged


def test_store_false_returns_value_without_caching():
    cache = ContentCache("test.nostore")
    fp = b"fp-y"
    r = cache.get_or_compute(fp, (), lambda: 7, store=False)
    assert r == 7
    assert cache.size == 0


def test_evict_purges_only_that_fingerprint():
    cache = ContentCache("test.evict-one")
    cache.get_or_compute(b"fp-a", (), lambda: 1)
    cache.get_or_compute(b"fp-b", (), lambda: 2)
    cache.evict(b"fp-a")
    assert cache.size == 1
    assert cache.get_or_compute(b"fp-b", (), lambda: 99) == 2  # still cached
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/cache/test_content_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.cache.content'`.

- [ ] **Step 3: Implement `ContentCache`**

Create `pyphi/cache/content.py`:

```python
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
from typing import Any, Callable

from pyphi.cache.cache_utils import memory_full
from pyphi.cache.policy import _DictCacheAdapter
from pyphi.cache.registry import register as _register_policy


class ContentCache:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hits = 0
        self.misses = 0
        self._cache: dict[tuple, Any] = {}
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
```

Add to `pyphi/cache/__init__.py` near the other re-exports (after the `from .registry import ...` block at the end):

```python
from .content import ContentCache as ContentCache  # noqa: E402
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/cache/test_content_cache.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/cache/content.py pyphi/cache/__init__.py test/cache/test_content_cache.py
git commit   # "Add ContentCache: content-addressed cache with refcounted eviction"
```

---

## Task 2: Substrate fingerprints

**Files:**
- Modify: `pyphi/substrate.py` (add two `cached_property` digests; add `from functools import cached_property` and `import hashlib` if absent)
- Test: `test/test_fingerprint.py`

**Interfaces:**
- Consumes: `Substrate._factored_tpm` (`FactoredTPM` with `.alphabet_sizes: tuple[int,...]`, `.n_nodes: int`, `.factor(i) -> np.ndarray`), `Substrate._cm` (`np.ndarray`).
- Produces: `Substrate._cm_fingerprint -> bytes` (32), `Substrate._math_fingerprint -> bytes` (32).

- [ ] **Step 1: Write the failing tests**

Create `test/test_fingerprint.py`:

```python
import numpy as np

from pyphi import Substrate, System, examples


def test_cm_fingerprint_ignores_tpm_weights():
    # Same connectivity, different TPM weights -> equal cm fingerprint.
    cm = np.array([[1, 1], [1, 1]])
    tpm_a = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    tpm_b = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]])
    a = Substrate(tpm_a, cm)
    b = Substrate(tpm_b, cm)
    assert a._cm_fingerprint == b._cm_fingerprint
    assert a._math_fingerprint != b._math_fingerprint  # TPM differs


def test_cm_fingerprint_separates_topologies():
    tpm = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    a = Substrate(tpm, np.array([[1, 1], [0, 1]]))
    b = Substrate(tpm, np.array([[1, 1], [1, 1]]))
    assert a._cm_fingerprint != b._cm_fingerprint


def test_substrate_fingerprint_ignores_labels():
    s = examples.basic_substrate()
    relabeled = Substrate(
        s.tpm, s.cm, node_labels=("X", "Y", "Z")
    )
    assert s._math_fingerprint == relabeled._math_fingerprint
    assert s._cm_fingerprint == relabeled._cm_fingerprint


def test_fingerprint_is_32_bytes_and_deterministic():
    s = examples.basic_substrate()
    assert len(s._math_fingerprint) == 32
    assert len(s._cm_fingerprint) == 32
    assert s._math_fingerprint == examples.basic_substrate()._math_fingerprint
```

(If `examples.basic_substrate()` lacks `.tpm`, use `s.to_joint()` for the TPM arg, or build the relabeled substrate from the same `factored_tpm`. Confirm the constructor signature with `uv run python -c "from pyphi import Substrate; help(Substrate.__init__)"` first.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_fingerprint.py -v`
Expected: FAIL — `AttributeError: 'Substrate' object has no attribute '_cm_fingerprint'`.

- [ ] **Step 3: Implement the fingerprints**

In `pyphi/substrate.py`, ensure these imports are present at the top:

```python
import hashlib
from functools import cached_property
```

Add to the `Substrate` class (near the `cm` / `factored_tpm` properties):

```python
@cached_property
def _cm_fingerprint(self) -> bytes:
    """blake2b-256 digest of the connectivity matrix (label-free).

    The exact dependency of ``potential_purviews``, which reads only ``cm``.
    """
    cm = np.ascontiguousarray(self._cm).astype(np.int8, copy=False)
    h = hashlib.blake2b(digest_size=32)
    h.update(repr(cm.shape).encode())
    h.update(cm.tobytes())
    return h.digest()

@cached_property
def _math_fingerprint(self) -> bytes:
    """blake2b-256 digest of the full label-free substrate math identity.

    Covers exactly what ``Substrate.__eq__`` compares: alphabet sizes, the
    factor array bytes (``+ 0.0`` to fold ``-0.0`` like ``FactoredTPM.__hash__``),
    and the connectivity. Excludes ``node_labels`` / ``state_space`` labels.
    """
    ftpm = self._factored_tpm
    h = hashlib.blake2b(digest_size=32)
    h.update(repr(ftpm.alphabet_sizes).encode())
    for i in range(ftpm.n_nodes):
        h.update((ftpm.factor(i) + 0.0).tobytes())
    h.update(self._cm_fingerprint)
    return h.digest()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_fingerprint.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/substrate.py test/test_fingerprint.py
git commit   # "Add label-free Substrate cm/math fingerprints (blake2b-256)"
```

---

## Task 3: System fingerprint

**Files:**
- Modify: `pyphi/system.py` (add `import hashlib`; add one `cached_property`)
- Test: `test/test_fingerprint.py` (extend)

**Interfaces:**
- Consumes: `Substrate._math_fingerprint` (Task 2); `System.substrate`, `System.state`, `System.node_indices`, `System.external_indices`, `System.partition` (with `.indices` and `.removed_edges()`).
- Produces: `System._math_fingerprint -> bytes` (32).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_fingerprint.py`:

```python
def test_system_fingerprint_ignores_labels_but_tracks_state_and_cut():
    base = examples.basic_substrate()
    relabeled = Substrate(base.tpm, base.cm, node_labels=("X", "Y", "Z"))
    s1 = System(base, (0, 0, 0))
    s2 = System(relabeled, (0, 0, 0))
    assert s1._math_fingerprint == s2._math_fingerprint  # label-free

    s_other_state = System(base, (1, 0, 0))
    assert s1._math_fingerprint != s_other_state._math_fingerprint

    cut = s1.partition  # NullCut; build a real cut for the contrast
    from pyphi.models.partitions import DirectedBipartition
    s_cut = System(base, (0, 0, 0), partition=DirectedBipartition((0,), (1, 2)))
    assert s1._math_fingerprint != s_cut._math_fingerprint


def test_equivalent_systems_share_fingerprint_and_phi():
    s1 = examples.basic_system()
    s2 = examples.basic_system()  # re-constructed, distinct object
    assert s1._math_fingerprint == s2._math_fingerprint
    assert s1.sia().phi == s2.sia().phi
```

(Confirm `System(substrate, state, partition=...)` is the real constructor signature; the spec records `System = (Substrate, state, node_subset, partition)`. Adjust the `System(...)` calls if `node_subset` is a required positional.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_fingerprint.py -v -k system`
Expected: FAIL — `AttributeError: 'System' object has no attribute '_math_fingerprint'`.

- [ ] **Step 3: Implement**

In `pyphi/system.py` add `import hashlib` at the top, and add to the `System` class:

```python
@cached_property
def _math_fingerprint(self) -> bytes:
    """blake2b-256 digest of the label-free system math identity.

    Serializes exactly the components ``System.__eq__`` compares: the
    substrate math fingerprint, the (index-coerced) state, the node and
    external indices, and the partition's mathematical content
    (``indices`` + ``removed_edges()``).
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(self.substrate._math_fingerprint)
    h.update(repr(tuple(self.state)).encode())
    h.update(repr(tuple(self.node_indices)).encode())
    h.update(repr(tuple(self.external_indices)).encode())
    h.update(repr(tuple(sorted(self.partition.indices))).encode())
    h.update(repr(sorted(self.partition.removed_edges())).encode())
    return h.digest()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_fingerprint.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add pyphi/system.py test/test_fingerprint.py
git commit   # "Add label-free System math fingerprint (blake2b-256)"
```

---

## Task 4: Re-key the kernel cache onto `ContentCache`

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py` (rewrite `_memoize`, `_evict`, `clear_caches`, `cache_info`; remove `_observers`)
- Test: `test/core/test_kernel_cache_reuse.py`

**Interfaces:**
- Consumes: `ContentCache` (Task 1), `System._math_fingerprint` (Task 3).
- Produces: `_memoize` keyed on the System fingerprint; `clear_caches(cs=None)` and `cache_info()` unchanged in signature.

- [ ] **Step 1: Write the failing test**

Create `test/core/test_kernel_cache_reuse.py`:

```python
from pyphi import examples
from pyphi.core import repertoire_algebra as ra


def test_distinct_equivalent_systems_share_kernel_cache():
    ra.clear_caches()
    s1 = examples.basic_system()
    s2 = examples.basic_system()  # distinct object, same math
    phi1 = s1.sia().phi
    sizes_after_first = sum(c["size"] for c in ra.cache_info().values())
    assert sizes_after_first > 0
    phi2 = s2.sia().phi  # should hit the cache populated by s1
    assert phi1 == phi2
    # No new kernel entries created for the equivalent second system.
    sizes_after_second = sum(c["size"] for c in ra.cache_info().values())
    assert sizes_after_second == sizes_after_first
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/core/test_kernel_cache_reuse.py -v`
Expected: FAIL — `sizes_after_second` exceeds `sizes_after_first` (id-keyed cache double-stores).

- [ ] **Step 3: Rewrite the kernel cache**

In `pyphi/core/repertoire_algebra.py`, replace the `_caches` / `_kernel_stats` / `_observers` / `_evict` / `_memoize` / `cache_info` / `clear_caches` block (lines ~39–113) with:

```python
from pyphi.cache.content import ContentCache

# One ContentCache per memoized function name.
_kernel_caches: dict[str, ContentCache] = {}


def _memoize(fn: Callable) -> Callable:
    """Memoize a function over System instances by content fingerprint.

    Distinct-but-equivalent Systems (re-constructed, or label-distinct) share
    entries; a fingerprint's entries are evicted when its last live carrier is
    garbage-collected. Inserts stop when ``memory_full()`` reports memory above
    ``maximum_cache_memory_percentage`` — already-computed values are still
    returned, just not cached.
    """
    cache = ContentCache(f"kernel.{fn.__name__}")
    _kernel_caches[fn.__name__] = cache

    @wraps(fn)
    def wrapper(cs: Any, *args: Any) -> Any:
        fp = cs._math_fingerprint
        cache.observe(cs, fp)
        return cache.get_or_compute(fp, args, lambda: fn(cs, *args))

    return wrapper


def cache_info() -> dict[str, dict[str, int]]:
    """Return per-function cache size."""
    return {name: {"size": c.size} for name, c in _kernel_caches.items()}


def clear_caches(cs: Any | None = None) -> None:
    """Clear cache entries. If ``cs`` given, clear only that instance's entries."""
    if cs is None:
        for c in _kernel_caches.values():
            c.clear()
        return
    fp = cs._math_fingerprint
    for c in _kernel_caches.values():
        c.evict(fp)
```

Delete the now-unused imports/objects: the module-level `_caches`, `_kernel_stats`, `_observers` (`WeakValueDictionary`), the old `_evict`, and the `_DictCacheAdapter` / `register` imports inside the old `_memoize` (they now live in `ContentCache`). Keep the `from functools import wraps` import. Remove the `WeakValueDictionary` import if nothing else uses it (grep first).

- [ ] **Step 4: Run to verify pass + no regression**

Run: `uv run pytest test/core/test_kernel_cache_reuse.py test/cache/test_content_cache.py -v`
Expected: PASS.

Then the fast lane:
Run: `uv run pytest test/core/ test/test_subsystem_surface.py -q`
Expected: PASS (no regressions in the kernel's consumers).

- [ ] **Step 5: Commit**

```bash
git add pyphi/core/repertoire_algebra.py test/core/test_kernel_cache_reuse.py
git commit   # "Key the repertoire kernel cache on System fingerprint"
```

---

## Task 5: Migrate the purview cache to `ContentCache`

**Files:**
- Modify: `pyphi/substrate.py` (rewrite `potential_purviews`; it no longer uses `@cache.method`)
- Test: `test/test_fingerprint.py` (extend) or `test/test_substrate.py`

**Interfaces:**
- Consumes: `ContentCache` (Task 1), `Substrate._cm_fingerprint` (Task 2), `irreducible_purviews`, `utils.powerset`.
- Produces: `Substrate.potential_purviews(direction, mechanism)` content-cached on the cm fingerprint, gated by `config.infrastructure.cache_potential_purviews`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_fingerprint.py`:

```python
def test_potential_purviews_shared_across_same_cm_substrates():
    import numpy as np
    from pyphi import Substrate
    from pyphi.direction import Direction
    from pyphi.substrate import _PURVIEW_CACHE

    _PURVIEW_CACHE.clear()
    cm = np.array([[1, 1], [1, 1]])
    tpm_a = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    tpm_b = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.5], [0.6, 0.3]])
    a = Substrate(tpm_a, cm)
    b = Substrate(tpm_b, cm)  # same topology, different weights

    pa = a.potential_purviews(Direction.CAUSE, (0,))
    size_after_a = _PURVIEW_CACHE.size
    pb = b.potential_purviews(Direction.CAUSE, (0,))  # should hit a's entry
    assert pa == pb
    assert _PURVIEW_CACHE.size == size_after_a  # no new entry for b
```

(Confirm `Direction` import path with `uv run python -c "from pyphi.direction import Direction"`; adjust if it lives elsewhere.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_fingerprint.py -v -k potential_purviews`
Expected: FAIL — `ImportError: cannot import name '_PURVIEW_CACHE'`.

- [ ] **Step 3: Migrate `potential_purviews`**

In `pyphi/substrate.py`, add a module-level cache near the top (after imports):

```python
_PURVIEW_CACHE = ContentCache("substrate.potential_purviews")
```

(Add `from pyphi.cache.content import ContentCache` to the imports.)

Replace the decorated method:

```python
@cache.method("purview_cache")
def potential_purviews(
    self, direction: Direction, mechanism: Mechanism
) -> list[Purview]:
    ...
    all_purviews = utils.powerset(self._node_indices)
    return irreducible_purviews(self.cm, direction, mechanism, all_purviews)
```

with:

```python
def potential_purviews(
    self, direction: Direction, mechanism: Mechanism
) -> list[Purview]:
    """All purviews which are not clearly reducible for mechanism.

    Depends only on connectivity, so the result is shared across all
    substrates with the same ``cm`` (keyed on ``_cm_fingerprint``).
    """
    def compute() -> list[Purview]:
        all_purviews = utils.powerset(self._node_indices)
        return irreducible_purviews(self.cm, direction, mechanism, all_purviews)

    fp = self._cm_fingerprint
    _PURVIEW_CACHE.observe(self, fp)
    return _PURVIEW_CACHE.get_or_compute(
        fp,
        (direction, mechanism),
        compute,
        store=config.infrastructure.cache_potential_purviews,
    )
```

Remove the now-unused per-instance `purview_cache` attribute assignments (`substrate.py:182,290`) and the `PurviewCache` import if nothing else references it (grep `purview_cache` and `PurviewCache` first; the `cache.method("purview_cache")` decorator was the only consumer).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_fingerprint.py -v`
Expected: PASS (all, including the new purview test).

Run: `uv run pytest test/test_substrate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/substrate.py test/test_fingerprint.py
git commit   # "Key potential_purviews on the cm fingerprint (cross-substrate reuse)"
```

---

## Task 6: Correctness invariants + full verification

**Files:**
- Test: `test/test_fingerprint.py` (add Hypothesis invariants)
- Possibly modify: `test/data/perf/call_counts.json` (regenerate if hit-rate shift changes pinned counts)

**Interfaces:**
- Consumes: everything above.
- Produces: the property invariants gating fingerprint soundness; a green full suite.

- [ ] **Step 1: Write the Hypothesis invariants**

Append to `test/test_fingerprint.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def small_substrates(draw):
    import numpy as np
    n = draw(st.integers(min_value=2, max_value=3))
    tpm = np.array(
        draw(
            st.lists(
                st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n),
                min_size=2 ** n, max_size=2 ** n,
            )
        )
    )
    cm = np.array(
        draw(
            st.lists(
                st.lists(st.integers(0, 1), min_size=n, max_size=n),
                min_size=n, max_size=n,
            )
        )
    )
    return tpm, cm


@settings(max_examples=100, deadline=None)
@given(small_substrates(), st.permutations(["A", "B", "C", "D"]))
def test_relabeling_collides_and_agrees(sub, perm):
    import numpy as np
    from pyphi import Substrate

    tpm, cm = sub
    n = cm.shape[0]
    labels = tuple(perm[:n])
    base = Substrate(tpm, cm)
    relabeled = Substrate(tpm, cm, node_labels=labels)
    # Relabeling never changes the math fingerprint.
    assert base._math_fingerprint == relabeled._math_fingerprint


@settings(max_examples=100, deadline=None)
@given(small_substrates(), small_substrates())
def test_math_difference_separates(a, b):
    import numpy as np
    from pyphi import Substrate

    sa = Substrate(*a)
    sb = Substrate(*b)
    if sa == sb:
        assert sa._math_fingerprint == sb._math_fingerprint
    else:
        assert sa._math_fingerprint != sb._math_fingerprint
```

- [ ] **Step 2: Run the invariants**

Run: `uv run pytest test/test_fingerprint.py -v`
Expected: PASS. If `test_math_difference_separates` ever fails, a math-distinct pair collided — investigate the fingerprint serialization before proceeding (do not weaken the assert).

- [ ] **Step 3: Full golden suite (byte-identical φ)**

Run: `uv run --all-extras pytest` (no path argument)
Expected: PASS, every φ/α unchanged. This is the load-bearing check — the cache must not alter any value. The doctest sweep over `pyphi/` runs here too.

- [ ] **Step 4: Perf-counter gate**

Run: `uv run pytest test/integration/test_perf_counters.py -v`
Expected: PASS. The cProfile call counts may legitimately shift (the point of P9.5 is fewer recomputations on repeated/equivalent systems). If a pin changes:
- Confirm the change is a *reduction* (or an expected structural change), not a correctness regression.
- Regenerate: `uv run python scripts/gen_perf_counts.py` and review the diff to `test/data/perf/call_counts.json` like a golden before staging.

- [ ] **Step 5: Commit**

```bash
git add test/test_fingerprint.py
# include test/data/perf/call_counts.json only if it legitimately changed
git commit   # "Add fingerprint soundness invariants for the content cache"
```

---

## Task 7: ROADMAP update

**Files:**
- Modify: `ROADMAP.md` (P9.5 dashboard row line ~43; the P9.5 detail at lines ~1248–1280)

- [ ] **Step 1: Flip the dashboard row**

Change the P9.5 row (line ~43) from `⬜ open` to `✅ landed`, with a one-line summary: content-addressed `ContentCache` keyed on a label-free `blake2b-256` fingerprint; kernel cache on the System fingerprint, purview cache on the cm fingerprint; refcounted eviction preserves prompt release; unblocks N4. Add a matching "landed" note in the detail section, and confirm no other ROADMAP prose still calls P9.5 upcoming.

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit   # "Mark P9.5 (math-fingerprint cache keys) landed"
```

---

## Notes for the implementer

- **`cache_repertoires` gate:** the kernel functions named `*repertoire*` are skipped from caching by `cache.method` only where that decorator is used; the kernel `_memoize` does not currently consult `cache_repertoires`. Do not add a new gate — preserve existing behavior. (If a kernel test depends on `cache_repertoires=False` disabling the kernel cache, surface it; it is out of scope here.)
- **`tobytes()` is native-endian.** Fine for this in-process cache; N4 (cross-machine) will need an endian-normalized serialization, which is N4's concern, not this task's.
- **Confirm constructor signatures** (`Substrate.__init__`, `System.__init__`, `Direction` import path, `examples.basic_substrate().tpm`) with a quick `uv run python -c "..."` before relying on the exact calls in the test snippets — they are written from the spec, not copy-verified.
- **Fast lane while iterating:** `uv run pytest test/cache/ test/core/ test/test_fingerprint.py -q`. The full `uv run --all-extras pytest` (Task 6 Step 3) is the only complete verification — run it with no path argument at least once.
