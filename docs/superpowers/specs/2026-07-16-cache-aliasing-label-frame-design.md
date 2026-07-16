# Cache/Aliasing Safety and the Serialization Label Frame — Design

**Date:** 2026-07-16
**Status:** Approved design, pending implementation
**Scope:** Wave 2 sub-unit B of the whole-library review: four confirmed
cache/aliasing findings, plus the serialization label normalization that
resolves the disk-cache finding at its root.

---

## Problem

Four confirmed findings share one theme: cached or hashed values are exposed
through writable aliases, or served under a key that does not capture
everything the stored value carries.

1. **Kernel cache returns shared writable arrays.** `_memoize`
   (`pyphi/core/repertoire_algebra.py`) returns the array stored in
   `ContentCache` by reference, with no copy and no read-only flag. The public
   `cause_repertoire()` hands that array straight to the caller, so a caller
   mutation silently poisons every later φ computation on that system and on
   every fingerprint-equivalent system. `max_entropy_distribution`
   (`pyphi/distribution.py`), cached by the module-level `@cache()` decorator
   and reachable from the same public path (empty mechanism), has the
   identical hazard.

2. **Disk result-cache label collision.** `result_cache_key`
   (`pyphi/cache/disk.py`) digests `System._fingerprint`, which is label-free
   by design — but the stored serialized SIA/CES embeds `node_labels`. A hit
   therefore serves substrate A's labels to a mathematically identical
   substrate labeled differently.

3. **FactoredTPM aliases caller arrays.** `FactoredTPM.__init__` stores
   `np.asarray(f, dtype=np.float64)`, which is a no-copy pass-through for
   float64 input. Post-construction mutation of the caller's array changes a
   hashed, fingerprinted value type; the cached `Substrate._fingerprint` goes
   stale. Both storage backends alias (the xarray backend wraps without
   copying and `.values` returns the underlying buffer).

4. **`cache_repertoires = False` is silently ignored.** The only reader of
   the option is the `cache.method` decorator, which has zero production
   users — a leftover of the pre-fingerprint per-instance cache architecture,
   along with `validate_parent_cache` and `DictCache`. The kernel `_memoize`
   never consults the option. (The sibling option `cache_potential_purviews`
   *is* correctly wired, via `get_or_compute(..., store=...)` in
   `substrate.py` — the pattern to copy.)

## Root-cause view of finding 2

The label collision exists because serialized results denormalize display
metadata: roughly ten model classes carry `node_labels`, all of them exclude
it from `__eq__`/`__hash__` (the codebase already treats labels as
non-mathematical), and the serializer writes the same label table into every
nested RIA, MICE, and Part. The fix is to normalize labels in the
persistence layer: one label frame per serialized document, stamped onto the
decoded objects at load time. In-memory objects keep their labels exactly as
today — no user-facing display change.

This makes the disk cache correct *without touching its key*: the cache
stays label-free (cross-label sharing preserved), and every hit is decoded
with the requesting system's labels.

---

## Design

### A. Kernel cache returns read-only arrays

`_memoize`'s wrapper freezes every computed result with the existing
`pyphi.utils.np_immutable` before it is stored or returned:

```python
return cache.get_or_compute(
    fp, key_args, lambda: _utils.np_immutable(fn(cs, *args))
)
```

(Section D adds a `store=` argument to this same call; the final form
appears there.) All seven memoized functions return ndarrays, so no type
guard is needed.
Every path now hands out read-only arrays — cache hits (frozen at store
time), misses, and memory-full uncached returns alike — so caller mutation
raises `ValueError` instead of silently corrupting later computations.

`max_entropy_distribution` gets the same treatment: its return statement is
wrapped in `np_immutable`, so the array stored by `@cache()` and every
served hit is read-only. The other `@cache()` users return lists of tuples
and ints and are out of scope.

Freezing may surface latent in-place mutation inside pyphi's own compute
paths. The fast test lane runs immediately after the flip; any offender gets
an explicit `.copy()` at its own mutation site (mutating a cached operand
was already a latent bug).

### B. FactoredTPM owns its storage

`FactoredTPM.__init__` takes ownership of each factor: copy if storing the
input as float64 would share writable memory with the caller's array, then
freeze the stored array. The caller's own array is never modified.

```python
def _own_factor(f) -> NDArray[np.float64]:
    a = np.asarray(f, dtype=np.float64)
    if a.flags.writeable:
        if isinstance(f, np.ndarray) and np.may_share_memory(a, f):
            a = a.copy()
        a.flags.writeable = False
    return a
```

- Fresh conversions (lists, non-float64 arrays) are frozen without a copy.
- Writable float64 ndarray input is copied, then the copy is frozen.
- Read-only input is stored as-is (aliasing a read-only array is safe; this
  keeps the internal squeezed-view construction paths in
  `System.proper_cause_marginal` / `proper_effect_marginal` copy-free once
  their sources are frozen).

The fix lives at the single public choke point (`FactoredTPM.__init__`);
both storage backends then receive already-owned, frozen arrays and need no
change. Construction sites are all cold paths (Substrate construction,
System marginal properties, macro construction, deserialization), so the
copy cost is negligible.

This also protects `Substrate._fingerprint` at the root: the bytes of a
hashed, fingerprinted value type can no longer change after construction.

### C. Serialization label frame + disk-cache stamping

**Envelope.** `_Document` (`pyphi/serialize/__init__.py`) gains an optional
frame field:

```python
class _Document(msgspec.Struct, frozen=True):
    format_version: int
    payload: schema.Schema | float
    node_labels: schema.NodeLabelsSchema | None = None
```

**Encode rule — first labeled object claims the frame.** Encoders currently
write `node_labels=_enc_optional(obj.node_labels)` into their structs. That
call is replaced by a shared helper backed by an encode context (a
`contextvars.ContextVar` set up by `dumps()`):

- If the object's labels are `None`: write `None` (unchanged).
- If no frame has been claimed yet: claim these labels as the document
  frame; write `None` into the struct.
- If the labels equal the claimed frame: write `None`.
- If they differ (heterogeneous document): write them per-object as an
  explicit override.

`dumps()` then places the claimed frame on the document envelope. One label
table per document in the common case; heterogeneous label sets remain
exactly representable via per-object overrides.

**Decode rule — inherit the frame.** `loads()` establishes the decode
frame: the caller-supplied override if given, else the document frame
decoded to a domain `NodeLabels`. Each decoder resolves labels as "own
stored labels if present, else the frame":

```python
def loads(data, *, format="json", node_labels=None): ...
```

The parameter accepts a domain `NodeLabels` (or `None`, the default, which
uses the document frame).

- Old files (every object self-carries labels, no envelope frame) decode
  byte-identically to today: per-object labels always win.
- New files inherit the frame everywhere except explicit overrides.
- `format_version` stays 1 — the change is purely additive (a new optional
  envelope field; nested label fields become optional-in-practice but keep
  their schema slots).

**Documents with a single root object.** `dumps()` accepts exactly one
domain object per document (the payload is one tagged struct). Serializing a
bare nested structure — a single Distinction, a `Distinctions` container —
works under the same rule: the first labeled object in that document claims
the frame, whatever the root type is. Plain Python lists were never
serializable and remain so.

**Disk cache.** `maybe_disk_cached` decodes hits with the requesting
system's labels:

```python
result = _decode_or_none(hit, node_labels=system.node_labels)
```

(`_decode_or_none` forwards the parameter to `serialize.loads`.) The cache
key is unchanged: label-free, cross-label sharing preserved, and every
requester receives results in its own labels. A cached result computed by an
unlabeled twin stamps correctly too, since the frame override applies
regardless of what the document carries.

### D. `cache_repertoires` wired; dead code deleted

`_memoize`'s wrapper threads the option into the store decision, mirroring
the `cache_potential_purviews` gate in `substrate.py`:

```python
return cache.get_or_compute(
    fp,
    key_args,
    lambda: _utils.np_immutable(fn(cs, *args)),
    store=config.infrastructure.cache_repertoires,
)
```

Semantics match the sibling option: with the option off, entries already
cached are still served, but the cache stops growing. The config option's
documented description ("memoize repertoire computations") is already
accurate.

Deleted, with their tests:

- `pyphi.cache.method` (the dead decorator — the option's only current
  reader),
- `pyphi.cache.validate_parent_cache`,
- `pyphi.cache.DictCache`,
- the autosummary stubs `docs/reference/_autosummary/pyphi.cache.DictCache.rst`
  and `pyphi.cache.method.rst`, and their entries in
  `docs/reference/_autosummary/pyphi.cache.rst`.

The registry adapter `_DictCacheAdapter` (`pyphi/cache/policy.py`) is a
different, live class and stays. The module-level `@cache()` decorator has
live users (`partition.py`, `distribution.py`, `combinatorics.py`) and
stays. The pinning test `test_cache_repertoires_config_option` is rewritten
against the real kernel path: compute a repertoire under
`cache_repertoires=False` and assert the kernel cache holds no new entries,
then under `True` and assert it does.

---

## Alternatives considered

- **Labels in the disk-cache key.** Two lines, correct, but each labeling of
  a mathematically identical system gets its own disk entry. Superseded by
  the label frame, which preserves cross-label sharing at moderate,
  well-contained cost (the decode traversal already exists and is covered by
  the round-trip test suite).
- **Relabel-on-hit via `relabel.py`.** Rejected: `relabel.py` supports only
  IIT 4.0 structures (IIT 3.0 raises `NotImplementedError`) and drops tie
  back-references by documented contract — it would strip tie peers from
  cached results.
- **Label mismatch treated as a miss.** Rejected: strictly dominated — the
  first labeling squats on the entry and every differently-labeled request
  recomputes forever.
- **Removing labels from in-memory model objects** (full display/value
  decoupling). Rejected after cost-benefit analysis: nested objects share
  one `NodeLabels` instance by reference in memory, so the duplication cost
  is on disk, not in RAM; the persistence-layer frame captures every
  material benefit (label-independent caching, serialization size, one
  source of truth per document). What the in-memory refactor would add is
  conceptual purity, purchased with a ten-class blast radius and a real
  regression risk for interactive display of nested objects
  (`print(sia.cause.partition)` is a core researcher workflow).

## Testing

TDD throughout; each finding starts from a failing repro:

1. **Kernel poison repro:** mutate the array returned by
   `cause_repertoire()` → the mutation raises (read-only); a fresh call
   returns correct values. Same for `max_entropy_distribution`.
2. **FactoredTPM ownership:** construct from a float64 array, mutate the
   caller's array afterward → factors, equality, hash, and
   `Substrate._fingerprint` are unaffected; stored factors are read-only;
   constructing from a read-only array does not copy.
3. **Disk-cache stamping:** cache a result from system A, request with a
   relabeled twin B → hit, result carries B's labels (both `sia` and `ces`
   kinds; IIT 3.0 and 4.0).
4. **Label frame round-trips:** per result family, encode → document carries
   one frame, nested structs carry `None` → decode restores labels
   everywhere (leans on the existing round-trip suite). Old-format
   back-compat: a document with per-object labels and no envelope frame
   decodes unchanged. Heterogeneous override: an object whose labels differ
   from the frame keeps its own on round-trip. `loads(node_labels=...)`
   override wins over the document frame.
5. **Config gate:** `cache_repertoires=False` → kernel cache gains no
   entries; `True` → it does.

Completion gate: pathless `uv run pytest` green in the worktree and on main
after merge. The fast lane runs immediately after the freeze flips (step A)
to surface latent internal mutators. The perf call-count gate
(`test/data/perf/call_counts.json`) must be unaffected; if the freeze
forces copies on a repertoire hot path, `just bench` sanity-checks wall
time.
