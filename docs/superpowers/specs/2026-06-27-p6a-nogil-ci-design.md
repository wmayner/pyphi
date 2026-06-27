# P6a — Free-threaded CI lane + concurrent cache safety — Design

**Status:** proposed
**Date:** 2026-06-27
**Roadmap item:** P6a (Wave 4) — the remaining two pieces: the no-GIL CI lane and the cache counter-race cleanup.

## Goal

Add a continuous-integration lane that runs the PyPhi test suite on a
free-threaded Python interpreter, and make PyPhi's shared in-memory caches
sound when worker threads execute concurrently under that interpreter.

## Background

The lazy-import and module-level-globals half of P6a landed in 2026-05
(commit `51062319`): `import pyphi` no longer eagerly loads any C extension
that re-enables the GIL. Two pieces were deferred:

1. A continuous-integration lane on a free-threaded interpreter.
2. Cleanup of a data race on the cache hit/miss counters.

Both deferrals were predicated on facts that have since changed, so this
design re-grounds them against the current code, verified empirically by
building and running the suite on a free-threaded interpreter.

### Empirical findings (verified 2026-06-27)

- **The interpreter target is Python 3.14t, not 3.13t.** `msgspec` is now a
  core runtime dependency (via the `pyphi.serialize` package). It ships
  free-threaded wheels for `cp314t` only; on `cp313t` it has no wheel and its
  source build fails to compile. The full dependency set (all four extras)
  installs cleanly on 3.14t with no source builds. The regular `test.yml`
  matrix already covers 3.14 in its standard (GIL-enabled) form.
- **The suite already passes free-threaded.** With the GIL disabled
  (`PYTHON_GIL=0`) on 3.14t, the full suite reports **2786 passed, 283
  skipped** — identical to the standard run. `import pyphi` keeps
  `sys._is_gil_enabled()` at `False`. No test needs an `xfail`. (The original
  plan to `xfail` failing tests "until P6b lands" is obsolete: P6b landed and
  removed the last C extension on the relations path.)
- **The `dev` dependency group does not install free-threaded.** It pulls in
  the graphify tooling, whose `tree-sitter-php` grammar has no free-threaded
  wheel. The lane therefore installs the runtime plus extras plus the test
  tools directly, not via the `dev` group.
- **The shared caches crash under the thread scheduler.** `parallel_backend`
  resolves to `LocalThreadScheduler` (a `ThreadPoolExecutor`) on free-threaded
  runtimes. Its worker threads share the parent process's module-level caches:
  the per-kernel-function `ContentCache` instances in
  `core/repertoire_algebra.py`, the `_PURVIEW_CACHE` in `substrate.py`, the
  `@cache`-decorated dict caches, and `joblib_memory`. Forcing the thread
  backend on a small System-Integrated-Analysis computation produces the
  **correct** integrated-information value (equal to the sequential result),
  but raises, repeatedly:

  ```
  RuntimeError: dictionary changed size during iteration
    pyphi/cache/content.py, in ContentCache.evict
      for key in [k for k in self._cache if k and k[0] == fingerprint]:
  ```

  The exception is raised inside a `weakref.finalize` callback, so the
  interpreter currently swallows it ("Exception ignored while calling weakref
  callback") and the computation still returns the right answer. The sequential
  full-suite run never triggers it because it never starts the thread pool.
  The bug is real: eviction iterates the live cache dict while another thread
  mutates it, so the eviction silently fails (the entries it meant to remove
  leak) and the output is polluted with tracebacks.

So the second P6a piece is not a cosmetic counter cleanup — it is making the
content cache thread-safe enough that the thread scheduler does not crash or
leak. The counters themselves are the least important part.

## What actually races, and why most of it is already safe

Under free-threaded CPython, individual dict operations (`d[k] = v`,
`d.get(k)`, `d.pop(k)`) are atomic: each holds the dict's internal lock for its
duration. Three things in the cache layer are therefore *not* a problem:

1. **Concurrent reads and writes of the cache dict.** Atomic per operation.
2. **Two threads missing the same key and both computing it.** The cached
   values are deterministic (the fingerprint is a content digest of the exact
   mathematical inputs), so both compute the same value and the last write
   wins. The only cost is redundant work.
3. **The hit/miss counters** (`self.hits += 1`). A read-modify-write on a
   shared integer can lose updates, so the counts can drift low. These counters
   are diagnostics, exposed through `cache_info()` / `pyphi.cache.info()`, and
   are never read by any computation.

Two things *are* a problem and need fixing:

4. **`ContentCache.evict` iterating the live dict.** The list comprehension
   `[k for k in self._cache if ...]` re-enters Python between elements, so a
   concurrent insert raises `RuntimeError: dictionary changed size during
   iteration`. This is the reproduced crash above.
5. **The refcount bookkeeping** (`_live: dict[bytes, int]` and `_observed:
   set[int]`). `observe` does a compound read-modify-write ("if this source is
   new, add it and increment the fingerprint's count"), and `_on_death` does
   the matching decrement-and-maybe-evict. These spans are not atomic across
   the two structures, so concurrent calls can double-count, decrement to an
   inconsistent value, or evict a fingerprint whose count is being raised by
   another thread (premature eviction → redundant recompute; or a leak).

## Design

Two independent deliverables. They share the free-threaded interpreter but
touch disjoint files, so they can land as separate commits.

### Deliverable 1 — Thread-safe `ContentCache`

Make `ContentCache` safe for concurrent use by the thread scheduler, keeping
the hot path (cache hit, and the get/compute/store sequence) lock-free.

Add one `threading.Lock` to the cache. It guards the eviction-and-bookkeeping
critical sections only — all of which run at a frequency far below the cache
hit rate:

- **`observe`** keeps its lock-free fast path: `if sid in self._observed:
  return` (an atomic set membership test) handles the common case where the
  source object has already been registered. Only the first-observe of a given
  source object takes the lock, re-checks membership under it (double-checked
  locking), then mutates `_observed` and `_live` and installs the finalizer.
- **`_on_death`** (the `weakref.finalize` callback) takes the lock for its
  whole body: decrement the count, and when it reaches zero, drop the
  fingerprint and evict its entries.
- **`evict`** takes the lock, collects the keys to remove with a single atomic
  call (`list(self._cache)`), then removes them with `self._cache.pop(key,
  None)`. No comprehension over the live dict; no `del` that can race a
  concurrent insert. This is the line that crashes today.
- **`clear`** takes the lock and resets every structure.

`get_or_compute` is **not** locked. The dict get, the store
(`self._cache[key] = result`), and the counter increments stay lock-free: the
dict operations are atomic, the values are deterministic, and the counters are
diagnostic. A concurrent `evict` (holding the lock, popping keys) and a
lock-free store are both atomic per dict operation and never iterate the live
dict, so they cannot raise.

The hit/miss counters are documented as best-effort under free-threading:
exact under the GIL and under process isolation (the normal execution model),
approximate when multiple threads share the cache. They are diagnostics, and
locking them would tax the hottest path in the library — the repertoire kernel,
called on the order of millions of times — to make a number that nothing
computes on exact. Per-thread counters summed on read would make them exact
without contention, but the counts are not load-bearing, so that is out of
scope.

The module docstring of `pyphi/cache/__init__.py` currently states that caches
assume process isolation and are not thread-safe. Update it: the content cache
is now safe for concurrent thread access (correct values, sound eviction, no
crash); only the hit/miss counters are approximate under free-threading.

The same hit/miss-counter race exists in the `@cache` decorator and in
`DictCache`. Those caches are not shared by the thread scheduler today (they
are per-object or guard non-thread paths), so they are left as-is; the counter
documentation in the module docstring covers them.

### Deliverable 2 — Free-threaded CI lane

Add a job to `.github/workflows/test.yml` that runs the suite on Python 3.14t
with the GIL disabled, as a blocking required check (the suite is green, so a
regression that re-enables the GIL or breaks free-threaded compatibility should
fail CI like any other matrix entry).

The job:

- Installs the free-threaded interpreter (`uv python install 3.14t`).
- Installs PyPhi with its extras and the test tools directly, bypassing the
  `dev` group (which cannot install free-threaded):
  `uv pip install -e ".[visualize,caching,emd,xarray]" pytest hypothesis`
  into a `3.14t` environment.
- Runs `PYTHON_GIL=0 ... pytest` with the repository's default options (no path
  argument, so the `pyphi/` doctests and the slow Hypothesis and
  paper-reproduction tests all run, matching the standard lane).
- Asserts the GIL stayed disabled: a tiny step (or a session-scoped test) that
  fails if `sys._is_gil_enabled()` is `True`, so a future dependency that
  re-enables the GIL is caught rather than silently accepted.

This is a single-OS lane (Linux) to start; the free-threaded wheel coverage is
the binding constraint and Linux has the broadest coverage. macOS and Windows
free-threaded lanes are a later addition if wheel coverage warrants.

### Concurrency regression test

A test that guards the fix, in `test/cache/test_content_cache_threadsafe.py`.
It runs in the standard lane as well as the free-threaded one: its assertions
hold on any interpreter, and the fix must pass both. The eviction crash
reproduces reliably under free-threading, where worker threads run Python
concurrently; under the GIL it can still surface, because the GIL is released
between bytecodes and a `weakref` finalizer on another thread can mutate the
cache mid-comprehension, but that is timing-dependent. The test is written to
maximize concurrent insert-versus-evict pressure so it exercises the path
hard on both interpreters rather than relying on a guaranteed crash under the
GIL.

The test drives many threads through a shared `ContentCache`:

- Each thread repeatedly `observe`s short-lived carrier objects (so finalizers
  fire and `evict` runs concurrently with inserts) and calls `get_or_compute`
  on overlapping fingerprints.
- Assert no exception escapes (today's run raises `RuntimeError` inside the
  finalizers).
- Assert every value returned by `get_or_compute` equals the value its
  `compute` callable would produce — concurrent access never returns a wrong
  value.
- Assert that after all carriers are released and garbage is collected, the
  cache is empty and `_live` is empty — eviction is sound, nothing leaks.

A second, end-to-end test forces the thread backend on a small computation and
asserts the integrated-information value equals the sequential result, mirroring
the existing parallel-equals-sequential guard (`test_parallel_equals_sequential.py`)
but for the thread scheduler. This pins the behavior the manual probe verified:
correct value, no escaping crash.

## Files

- `pyphi/cache/content.py` — add the lock and the four guarded methods; make
  `evict` use an atomic snapshot.
- `pyphi/cache/__init__.py` — update the module docstring (threading section).
- `test/cache/test_content_cache_threadsafe.py` — new concurrency regression
  test (runs in the standard lane).
- `test/parallel/test_parallel_equals_sequential.py` — add a thread-backend
  equivalence case (or a sibling test module under `test/parallel/`).
- `.github/workflows/test.yml` — add the 3.14t free-threaded job.

## Out of scope

- Making the hit/miss counters exact under free-threading (per-thread
  counters). They are diagnostics; nothing computes on them.
- Thread-safety work on the `@cache` decorator and `DictCache`, which the
  thread scheduler does not share today.
- macOS and Windows free-threaded CI lanes.
- The `joblib_memory` on-disk cache, which serializes through the filesystem
  and is not in the in-memory race surface.

## Verification

- `pyphi/`-touching change, so the complete check is `uv run --all-extras
  pytest` with no path argument on the standard interpreter (must stay 2786
  passed / 283 skipped), plus the new concurrency test passing there.
- The free-threaded lane green on 3.14t with `PYTHON_GIL=0`.
- The manual thread-backend probe that currently prints `RuntimeError`
  tracebacks runs clean after the fix.
- Pre-commit (ruff + pyright) passes on the changed files.

## Roadmap bookkeeping

On landing, flip the P6a dashboard row in `ROADMAP.md` from partial to landed
and update its Wave 4 summary, recording the 3.14t target (not 3.13t) and the
content-cache thread-safety fix, in the same change.
