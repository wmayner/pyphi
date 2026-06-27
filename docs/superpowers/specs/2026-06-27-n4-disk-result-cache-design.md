# N4 — Disk-backed result cache — Design

**Status:** proposed
**Date:** 2026-06-27
**Roadmap item:** N4 (Wave 6, pulled into 2.0). Disk-persistent cache of top-level results keyed on the P9.5 content fingerprint, so notebook re-runs and paper reproductions skip recomputation.

## Goal

Persist the expensive top-level results (system-integration analyses and
cause-effect structures) to disk, keyed on the mathematical identity of the
input rather than object identity, so that re-running the same computation in a
later session loads the stored result instead of recomputing it. The cache is
**opt-in** and never returns a result computed by different code or different
result-affecting configuration.

## Background and constraints

The pieces this builds on already exist:

- **P9.5 content fingerprint** — `System._fingerprint` is a label-free
  `blake2b-256` digest of a system's mathematical identity. Two
  mathematically-equivalent systems (e.g. relabeled) share a fingerprint, and a
  real mathematical difference separates them. This is the cache key's core.
- **P15 serializer** — `pyphi.serialize.dumps(obj, format="msgpack")` /
  `loads(data, format="msgpack")` round-trips the top-level result types
  (verified: a basic-system SIA round-trips to ~7.5 KB with φ preserved). This
  is the on-disk value codec. `save`/`load` already do transparent gzip on
  `.gz` paths.
- **N8 provenance** — `pyphi.provenance.Provenance` already computes the pyphi
  version (`importlib.metadata.version`), the git commit sha
  (`git rev-parse HEAD`), and a dirty-tree flag (`git status --porcelain`).
  This supplies the cache key's code-version component without new machinery.
- **P10 config snapshot** — every top-level result carries a `config`
  (`ConfigSnapshot`) sibling field recording the configuration it was computed
  under. This supplies the key's config component.
- **Existing disk cache** — `joblib_memory` (`joblib.Memory`) caches the
  Hamming distance matrices. It stays as-is: its `int → ndarray` functions are
  correctly arg-hashed and version-insensitive. **joblib is not droppable** —
  `joblib.externals.loky` is the default process-pool parallel backend
  (`local_process.py`), so the dependency remains regardless. N4 is a separate,
  fingerprint-keyed, serialize-backed store, not a `joblib.Memory` instance.

The constraint unique to a *disk* cache, which the in-memory `ContentCache`
never faced: a persisted result outlives the process and the code that produced
it. Keying on mathematical identity alone is therefore unsafe — a formalism bug
fix would silently serve a stale, wrong result. The key must additionally fold
in the result-affecting configuration and a code-version component.

## The cache key

```
key = blake2b-256( system._fingerprint
                  + _config_digest(result.config)
                  + _code_version() )
```

- **`system._fingerprint`** — the P9.5 label-free system identity (bytes).
- **`_config_digest(snapshot)`** — a digest of only the **result-affecting**
  configuration fields: the formalism version, the mechanism / system / CES
  measures, the mechanism and system partition schemes, and `precision`.
  Infrastructure fields (parallel workers, progress bars, cache toggles) are
  excluded so that changing them does not cause spurious misses. The digest is
  computed from a `ConfigSnapshot` of the configuration in effect for this
  computation (`pyphi.conf._global.snapshot()`), which is exactly the snapshot
  the result will carry in its own `config` field — so the read-path key (built
  before the result exists) and the write-path key agree.
- **`_code_version()`** — `git_sha` when pyphi is running from a git checkout,
  otherwise the released version string from `importlib.metadata`. Both come
  from the same logic N8's `Provenance` already uses.

**Dirty-tree guard.** When the working tree is dirty (`git_dirty` true), the
sha no longer identifies the running code, so the cache disables itself
entirely — no read, no write — for that process. A released install (no git)
is never "dirty" and caches normally on its version string.

This makes the key self-adjusting: a released install gets cross-session hits
keyed on its version; a development checkout gets sha-exact safety and simply
skips the cache while the tree is dirty (active editing, when a stale result is
least wanted).

## Components

### `pyphi/cache/disk.py` — `DiskCache`

A minimal content-addressed file store. One file per key.

- `__init__(self, name, subdir)` — registers with the cache registry (so
  `pyphi.cache.info()` / `clear(name)` see it); resolves its directory under
  `constants.DISK_CACHE_LOCATION / subdir`.
- `get(self, key: str) -> bytes | None` — read the file named `key` if present;
  return its bytes, else `None`. A read error (missing, truncated) returns
  `None`.
- `put(self, key: str, data: bytes) -> None` — write atomically: write to a
  temp file in the same directory, then `os.replace` onto the final name (so a
  concurrent reader never sees a partial file, and concurrent writers of the
  same key are last-writer-wins on identical content).
- `clear(self) -> None` — remove the store's files.
- `size` / `info()` — count and total bytes, for the registry surface.

The directory is created lazily on first `put` (so merely importing pyphi, or
running with the cache off, creates nothing).

### Stored record format

Each file holds a small self-describing record so a future codec change cannot
mis-decode an old file:

```
record = serialize.dumps(
    {"v": _CACHE_FORMAT_VERSION, "result": <the result object>},
    format="msgpack",
)  # gzip-compressed
```

On load, a mismatched `v`, an unknown format, or any deserialize error is
treated as a **miss** (the entry is ignored and recomputed), never an
exception that reaches the caller.

### Key builder — `pyphi/cache/disk.py`

- `result_cache_key(system, config_snapshot) -> str | None` — returns the hex
  key, or `None` to signal "do not cache" (dirty tree). Pure; no I/O.
- `_config_digest(snapshot) -> bytes` — digests the curated result-affecting
  field subset.
- `_code_version() -> str` — sha-or-version, reusing the provenance logic.

### Hook — `pyphi/formalism/queries.py`

`sia(cs, ...)` and `ces(cs, ...)` are the single chokepoint both IIT 3.0 and
4.0 dispatch through. A thin wrapper there:

```
key = None
if config.infrastructure.disk_cache_results:
    key = result_cache_key(cs, _global.snapshot())   # None if dirty tree
    if key is not None:
        hit = _RESULT_DISK_CACHE.get(key)
        if hit is not None:
            result = _decode(hit)            # None on any decode error (miss)
            if result is not None:
                return result
result = <compute as today>
if key is not None:
    _RESULT_DISK_CACHE.put(key, _encode(result))
return result
```

The key is built from the *current* config snapshot on the miss path (the
result's own `config` field is used for the digest on store, and they are the
same configuration, so reads and writes agree).

## Config surface

- `config.infrastructure.disk_cache_results: bool = False` — the opt-in switch.
- Storage lives under the existing `constants.DISK_CACHE_LOCATION`
  (`__pyphi_cache__/`) in a `results/` subdirectory, so the joblib Hamming
  cache and the result cache are siblings under one root.
- `pyphi.cache.clear("disk.results")` clears it through the existing registry;
  a convenience `pyphi.cache.clear_disk()` may wrap that.

## Scope

**In scope (v1):** the two expensive top-level results — `sia` and `ces` —
across IIT 3.0 and 4.0, via the one `queries` chokepoint.

**Out of scope (noted follow-ons):**
- The actual-causation `account` result (a different entry point); a clean
  extension once v1 is proven.
- Any eviction / size-bound / LRU policy. v1 is unbounded with an explicit
  `clear` and a documented location. A size or age cap is a follow-on if disk
  growth becomes a real problem — not speculated on now.
- Caching mechanism-level results (MICE, distinctions); the in-memory
  `ContentCache` already covers the kernel, and these are cheap relative to the
  top-level results.

## Correctness and testing

The disk path is correctness-critical (it can return a stored φ instead of
computing one), so the gate is exact parity against recomputation.

- **Exact parity:** a disk hit equals recomputation, for `sia` and `ces` under
  both IIT 3.0 and 4.0 (`result_from_disk == result_recomputed`).
- **Key sensitivity:** a different result-affecting config yields a different
  key (no false hit); a relabeled but mathematically-equivalent system yields
  the *same* key (genuine reuse — the P9.5 property carried to disk).
- **Dirty-tree guard:** with the tree marked dirty, nothing is written and
  nothing is read.
- **Corruption / format drift:** a truncated file and a wrong-`v` record are
  each treated as a miss with no exception.
- **Opt-in:** with `disk_cache_results=False` (default), no directory or file
  is created and the compute path is byte-identical to today.
- **Full suite:** `uv run --all-extras pytest` (no path argument) stays green;
  the default-off cache changes no existing result.

## Files

- `pyphi/cache/disk.py` — new: `DiskCache`, the key builder, the config digest,
  the code-version helper, the record codec.
- `pyphi/cache/__init__.py` — export `DiskCache` / `clear_disk`.
- `pyphi/formalism/queries.py` — the `sia` / `ces` hook + the module-level
  `_RESULT_DISK_CACHE`.
- `pyphi/conf/infrastructure.py` — the `disk_cache_results` field.
- `pyphi/constants.py` — the `results/` subdir constant (or derived in
  `disk.py`).
- `test/cache/test_disk_cache.py` — the cases above.
- `changelog.d/disk-result-cache.feature.md`.

## Roadmap bookkeeping

On landing, flip the N4 dashboard row to landed with a one-line summary, in the
same change.
