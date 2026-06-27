# Math-fingerprint cache keys — design

**Status:** draft for review
**Roadmap item:** P9.5 (Wave 4)

## Problem

The repertoire kernel cache (`pyphi/core/repertoire_algebra._memoize`) keys on
`(id(cs), args)`, where `cs` is a `System`. Object identity means two *distinct*
`System` objects never share cache entries — even when they are mathematically
identical, and even when they differ only in presentation labels (`node_labels`
or `state_space` label strings). The same is true of the per-`Substrate`
`purview_cache`: `Substrate.potential_purviews` is an instance-level
`DictCache`, so two substrates with identical connectivity but different labels
keep separate caches. Both lead to redundant recomputation when a user explores
the same substrate under different labelings, re-creates an equivalent system,
or scripts many related computations.

## Key observation

`System` value identity is **already label-free**. `System.__eq__` / `__hash__`
compare `(substrate, state, node_indices, partition, external_indices)`, where:

- `Substrate.__eq__` / `__hash__` use the `FactoredTPM` and `cm` only;
- `FactoredTPM.__eq__` / `__hash__` use `alphabet_sizes` (the per-node state
  *count*, which is mathematical) and the factor *array bytes* — **not** the
  `state_space` label strings, **not** `node_labels`;
- `state` is coerced to integer indices in `System.__post_init__`;
- `partition` is a hashable cut whose mathematical content is its
  `removed_edges()`.

So the value identity is exactly the mathematical content, with labels already
excluded. Keying the cache on that value (rather than `id`) yields correct
cross-object reuse with no risk of label leakage.

## Why a content digest rather than the object itself

Keying the dict on the `System` object (`(cs, args)`) would already be correct —
Python dicts resolve by `__hash__` + `__eq__`. It is rejected for two reasons,
both about lifetime, not correctness:

1. **It pins objects alive.** A dict key holds a strong reference, so every
   cached `System` (and its `Substrate`, which carries the full TPM) could never
   be garbage-collected. Today's `id()` + `WeakValueDictionary` + `finalize`
   machinery exists precisely to avoid that.
2. **It is not serializable.** P9.5 exists to unblock N4 (a disk-backed cache)
   and the distributed/scripting workflows that motivate it. A disk or
   cross-process cache needs a key that survives serialization; an object
   identity cannot cross a process boundary, a fixed-width content digest can.

A digest also makes lookups cheaper: `FactoredTPM.__hash__` recomputes
`factor.tobytes()` on every call, so keying on the object re-hashes the TPM on
every cache lookup, whereas a `@cached_property` digest is computed once per
object.

A raw `hash()` (64-bit) is **not** an acceptable key: a hash collision between
two mathematically distinct systems would return a wrong repertoire — a silent
correctness failure. The fingerprint is therefore a wide cryptographic digest:
**`blake2b`, 32-byte / 256-bit**.

On the choice of digest: collisions here are accidental (scientific inputs), not
adversarial, so what is needed is a hash with no structural weaknesses (which
excludes MD5/SHA-1) that is wide enough that the birthday-bound collision
probability `≈ k²/2^(n+1)` over the `k` distinct keys a session produces is
negligible. `blake2b` is a modern cryptographic hash with no known collision
weakness and is faster than SHA-2/SHA-3, so there is no reason to substitute a
different algorithm. 128-bit already gives ~1e-21 at a billion keys; 256-bit is
chosen because widening is effectively free — `blake2b` truncates its output, so
256-bit is no slower than 128-bit, and 16 extra bytes per key is <0.1% beside the
cached arrays — and it pushes the collision probability (~1e-62 at 100M keys)
below uncorrected-hardware-error rates, so the digest is no longer the weakest
link in the correctness chain. Verifying a digest match against the full value
identity is deliberately **not** done: it would reintroduce the object-retention
and serialization costs the digest exists to avoid, for a margin already below
hardware error. A wide content hash is the correctness boundary, as in git, nix,
and content-addressed build caches.

## Design

**Guiding principle:** each cache keys on *exactly what its function reads*, no
more. Repertoires depend on the full system math (TPM included); potential
purviews depend only on connectivity. Keying each on its true dependency
maximizes correct reuse — in particular, a parameter sweep over a fixed topology
(same `cm`, different TPM weights) shares all of its potential-purview work.

### 1. `Substrate._cm_fingerprint` and `Substrate._fingerprint`

Two `@cached_property` digests (32-byte `blake2b`):

- **`_cm_fingerprint`** — over the connectivity matrix only: `cm.shape` and
  `cm.tobytes()` at a fixed dtype. This is what the purview cache keys on.
- **`_fingerprint`** — over the full label-free substrate content:
  `alphabet_sizes` (tuple of ints); each factor array's canonical bytes
  (`(factor(i) + 0.0).tobytes()`, matching `FactoredTPM.__hash__`'s normalization
  so `-0.0`/`+0.0` do not diverge), in node order; and `_cm_fingerprint`. This is
  what the system fingerprint composes.

Both exclude `node_labels` and `state_space` label strings.

### 2. `System._fingerprint`

A `@cached_property` returning a 32-byte `blake2b` digest over:

- `substrate._fingerprint` (the bytes above);
- `state` as the coerced integer-index tuple;
- `node_indices` and `external_indices` (tuples of ints);
- the partition's mathematical content: `tuple(sorted(partition.indices))` and
  `tuple(sorted(partition.removed_edges()))`.

These are exactly the components `System.__eq__` compares, serialized to bytes.
A correctness invariant (below) asserts the mapping cannot drift: equal value
identity ⇔ equal fingerprint.

### 3. `ContentCache` — the refcounted content cache

A reusable class in a new `pyphi/cache/content.py`, replacing the `id`-keyed
dict + per-object `_evict`. `pyphi/cache/` is the home: it already hosts
`DictCache`, `PurviewCache`, `method`, and the registry-adapter stats wiring,
and it imports neither `substrate` nor `repertoire_algebra` (no cycle). Both call
sites already depend on `pyphi.cache`. Centralizing the refcount + `finalize`
logic — the one genuinely tricky part — means getting it right and testing it
once rather than duplicating it.

`ContentCache` owns:

- the backing dict, keyed on `(fingerprint, args)`;
- a `live: dict[bytes, int]` counting how many live source objects currently
  carry each fingerprint;
- `observe(source_obj, fingerprint)` — the first time a source object (tracked by
  `id` to avoid double-counting) is seen, increment `live[fp]` and register
  `weakref.finalize(source_obj, _on_death, fp)`;
- `_on_death(fp)` — decrement `live[fp]`; at zero, purge every entry whose key
  starts with `fp` and drop `live[fp]`;
- `get`/`set` with `memory_full()` gating and hit/miss stats, and registry
  registration by name.

It is agnostic about *what* the fingerprint is — the kernel hands it a `System`
fingerprint, the purview path hands it a `cm` fingerprint.

This preserves today's prompt-release behavior: when the last source object
carrying a given fingerprint is collected (e.g. after a one-shot SIA goes out of
scope), its entries are evicted immediately. The only new behavior is the
intended one — while two equivalent objects are alive at once, they share the
entry instead of recomputing. The existing `memory_full()` insert gate and
`clear_caches()` / `clear_system_caches_after_computing_sia` levers are
unchanged.

### 4. Apply to both caches

- **Kernel cache** (`_memoize`): source object = `System`, fingerprint =
  `System._fingerprint`. `_memoize` stays as the ergonomic decorator but
  delegates to one `ContentCache` per function; replace `cs_id = id(cs)` with the
  fingerprint and drop the `_observers`/`_evict` pair.
- **Purview cache** (`Substrate.potential_purviews`): move from the per-instance
  `PurviewCache` to a module-level `ContentCache` keyed on
  `(substrate._cm_fingerprint, direction, mechanism)`, source object =
  `Substrate`. `potential_purviews` reads only `cm` and `node_indices`
  (`irreducible_purviews(self.cm, …)` over `powerset(self._node_indices)`) and
  never touches the TPM, so the `cm` fingerprint is its exact dependency — two
  substrates that share a topology but differ in TPM weights (a parameter sweep)
  then share all potential-purview work. The `cache_potential_purviews` config
  gate is preserved.

## Correctness

The fingerprint is sound iff it excludes only quantities that never affect the
computed value. It serializes precisely the components `System.__eq__` /
`Substrate.__eq__` compare, and those equalities are already the contract the
kernel relies on (the kernel reads `cs._index2node`, `cs.substrate.factored_tpm`,
`cs.substrate.node_indices` — all fixed by the value identity). Labels are
presentation only and never read by the repertoire algebra.

Verification:

1. **Hypothesis — relabeling collides and agrees.** For a random substrate and a
   random relabeling (permuted `node_labels`, aliased `state_space` labels with
   the same alphabet sizes), assert equal `_fingerprint` *and* byte-identical
   cause/effect repertoires and `sia`/`ces` φ.
2. **Hypothesis — math difference separates.** For two systems differing in any
   mathematical component (a TPM factor entry, `cm`, `state`, `node_indices`, or
   `partition`), assert *distinct* fingerprints (no false sharing). Distinct
   value identity ⇒ distinct fingerprint, checked over random small substrates.
3. **Golden suite byte-identical.** `uv run --all-extras pytest` (no path
   argument) green; every φ/α unchanged.
4. **Perf-counter gate.** The cProfile call-count gate
   (`test/integration/test_perf_counters.py`) confirms the hit rate improves
   where expected (cross-object reuse now lands) and nothing regresses; pins
   regenerated if the counts legitimately shift, reviewed like a golden.
5. **Eviction test.** Construct two equivalent systems, drop one, assert the
   shared entry survives; drop the second, assert the entry is purged (refcount
   reaches zero).
6. **cm-sharing test.** Two substrates with identical `cm` but different TPM
   weights return identical `potential_purviews` and share one cache entry (equal
   `_cm_fingerprint`); a `cm` change produces a distinct `_cm_fingerprint` and a
   separate entry.

## Files

- `pyphi/cache/content.py` (new) — the `ContentCache` class (refcounted,
  fingerprint-keyed); exported from `pyphi/cache/__init__.py`.
- `pyphi/substrate.py` — `Substrate._cm_fingerprint` and
  `Substrate._fingerprint`; migrate `potential_purviews` to a module-level
  `ContentCache` keyed on `_cm_fingerprint`.
- `pyphi/system.py` — `System._fingerprint`.
- `pyphi/core/repertoire_algebra.py` — re-key `_memoize` onto a `ContentCache`
  per function using `System._fingerprint`; drop `_observers`/`_evict`.
- `pyphi/cache/__init__.py` — re-export `ContentCache`; the per-instance
  `PurviewCache` is retired (or left unused) once `potential_purviews` moves.
- `test/cache/`, `test/core/` — `ContentCache` refcount/eviction unit tests;
  fingerprint correctness and cross-object reuse (Hypothesis).

## Risk

Low-to-medium. The change is correctness-sensitive (a bad key returns a wrong
repertoire), but the fingerprint is a direct serialization of the existing
value identity and is gated by the Hypothesis invariants, the byte-identical
golden suite, and the perf-counter gate. The eviction refactor is the main
moving part; its behavior is pinned by the eviction test and is a strict superset
of today's (same prompt release, plus cross-object reuse).

## Out of scope

- **N4** (disk-backed cache) — this builds the fingerprint it will key on, but
  the disk layer itself is separate.
- **Automorphism canonicalization** — sharing across node *permutations* (using
  `substrate_canonical_form`) is a strictly stronger, riskier equivalence (it
  must permute cached results back) and is the separate B10 research item. P9.5
  shares only across relabelings and re-construction, not permutations.
- **Thread safety** — the kernel cache is already documented as not thread-safe;
  the refcount dict inherits the same single-process caveat. Cross-process
  sharing is N4's concern, not this in-memory cache's.
