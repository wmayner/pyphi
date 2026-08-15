# Cache-key hashing and the performance guards that missed it

**Date:** 2026-08-14
**Status:** landed on `main` in `4623c412..d95689e6`

## The defect

`FrozenMap.__hash__` combined the key set and the value set separately:

```python
hash((frozenset(self._dict), frozenset(iter(self._dict.values()))))
```

This satisfies the hash contract — equal mappings hash equally — but does not
distinguish mappings that differ only in which key holds which value. The
repertoire cache keys effect repertoires on exactly such a mapping (mechanism
node → that node's state), so for binary units every condition over a given
mechanism shares one key set and draws its values from `{0, 1}`: 2ⁿ distinct
conditions collapse onto three hashes. Every cache operation then degenerated
to a linear scan of the bucket compared under `Mapping.__eq__`, which is itself
O(k) and allocates two dictionaries per comparison. Filling a cache of m
entries cost O(m²).

The specified-state search inserts one entry per system state, so its cost grew
as 4ⁿ. Measured on a fully connected substrate: 2.5 s at 8 units, 55 s at 10,
279 s at 11, extrapolating to roughly ten days at 16.

Introduced in `f99b6694` (2022-11-21), which created the class, and made
load-bearing 48 seconds later by `7daa0261`, which merged a `mechanism`
argument and a `conditioning_state` **tuple** into one `FrozenMap` cache key.
Tuples hash positionally, so the key it replaced distributed correctly. The
data model improved and the hash quality collapsed silently. Reported
externally after a 16-unit analysis failed to return.

A second defect shares the call path: the full-state sweeps cached one full
repertoire per system state and read each back exactly once — order 4ⁿ cells
for a hit rate of zero — exhausting memory at 16 units. The
`unconstrained_forward_effect_repertoire` docstring claimed memory stayed at a
single repertoire; memoization of its inner calls made that false. The cause
direction had the same shape via `forward_cause_probability`, which builds a
full repertoire per purview state and reads one scalar from it.

A third defect surfaced while calibrating the guard's limit. The kernel's size
bound on full-state sweeps was checked only in the effect-direction sweep,
while `Direction.both()` walks the cause direction first — and the cause sweep
had no bound at all. An oversized system therefore ground through its entire
cause sweep (at 17 units, roughly 1.7×10¹⁰ cells) before the effect direction
refused. Both sweeps now share one `_validate_sweep_size` check.

## Why nothing caught it

Six independent reasons, each sufficient on its own.

1. **No test covered `FrozenMap`.** `test/data_structures/` held only
   `__init__.py`; the class had no coverage of its hash, its equality, or its
   use as a key.
2. **No wrong answers.** Every cached value and every φ was correct. All
   correctness gates were structurally blind.
3. **The call-count gate could not see it.** `test/golden/perf.py::FRAMES`
   counted PyPhi frames only. Measured under both hashes on
   `grid3_iit4_2026::sia`, the pinned counts were byte-identical: the extra
   work happens inside `dict.__setitem__`.
4. **Every fixture was too small.** The golden zoo tops out at four units.
   Instrumenting both lanes, the largest mechanism ever passed to the
   pathological sweep was 8 units in the fast lane and 7 in the slow lane —
   against the 16 that failed. Cost being quadratic in 2ⁿ, the suite's worst
   case was 1/65,536 of the failing workload.
5. **The regression harnesses postdate the defect.** The perf-counter gate,
   the shared perf harness, and the nightly ASV workflow all landed in June
   2026, three and a half years after the hash. A harness introduced after a
   defect adopts it as its baseline.
6. **The cost pre-flight had no term for it.** `estimate_analysis` counted
   partitions, mechanisms, and purview evaluations. `state_space_size` was
   documented as "reported as a weight, never multiplied into the counts", so
   the 2ⁿ sweep that dominated the run was charged to no axis.

## What changed

### The fixes

- `FrozenMap.__hash__` hashes `frozenset(self._dict.items())`. A class
  docstring states the requirement and why it matters.
- A `transient_repertoires()` context variable scope in
  `pyphi/core/repertoire_algebra.py` returns computed repertoires without
  admitting them. Both full-state sweeps run under it. Chosen over threading a
  `store=False` keyword through three public functions: the scope covers the
  whole call subtree, is per-thread by construction, and changes no signature.

A fast `FrozenMap.__eq__` was evaluated and **rejected**: it gives no measured
speedup once the hash is fixed (medians over three trials with cleared caches:
1.27 s vs 1.25 s at 12 units, 3.39 s vs 3.21 s at 13), and it bypasses the two
generic frames the collision sentinel below relies on, narrowing that guard
from any Mapping-based key type to `FrozenMap` alone.

### The guards

| Gap | Guard |
|---|---|
| No `FrozenMap` coverage | `test/data_structures/test_frozen_map.py` — hash separation over the 2ⁿ conditions, plus equality and mapping behaviour |
| Nothing generalizes | `test/data_structures/test_hash_quality.py` — a declared registry of every cache-key type with a hash-separation assertion, and a companion test that instruments `ContentCache` during real analyses and fails on an undeclared key type |
| Call counts blind to cost per operation | `Mapping.__eq__` and `FrozenMap.__getitem__` added to `FRAMES` and pinned. Verified deterministic across `PYTHONHASHSEED` and, on the *existing* 3-unit fixtures, 1321 vs 76 and 8484 vs 1014 |
| No gate sees memory | `test/cache/test_transient_repertoires.py` asserts a sweep's cache admissions scale with the unit count, not the state count (11 vs 11,264 at ten units) |
| Fixtures too small | A `specified_state` grain and seeded 10- and 12-unit ring fixtures in `test/golden/perf_fixtures.py`, perf-only (no φ goldens, absent from `ALL_FIXTURES`) |
| Cost model silent | `AnalysisEstimate.specified_state_evaluations` = 2 × state space, pinned against the evaluations actually performed; checked by the MCP `analyze` guard at a limit pinned to the kernel's own sweep bound (16 binary units — measured at 64 s and 0.19 GiB), with a test asserting the two stay in step |
| Sweep bound one-sided | `_validate_sweep_size` shared by both directions, with a test for each and one asserting the search refuses up front rather than after the cause sweep |

The nightly ASV suite picks up the new grain and fixtures automatically and
gains `track_*` metrics for the two collision frames.

## Verification

Against a branch reverting only the hash, **11 tests fail**: five direct
`FrozenMap` hash tests, the registry test, and the call-count pins on five
fixture/grain combinations. Against a branch neutralizing only the transient
scope, **9 fail**: three occupancy tests and six pins.

One honest negative result: `ring10_iit4_2026::specified_state` is insensitive
to both defects — identical counts in all three conditions — because the
transient scope leaves the cache empty during the sweep, so there is nothing to
collide with. It is kept for pinning the sweep's call structure at a size no
other fixture reaches, but it is not what guards these bugs; the small fixtures
are.

The series was checked commit by commit in a detached worktree, each one
building and running the full fast lane. This caught a real defect in the first
split: the ten-unit occupancy guard imports the ring fixtures, which had been
placed one commit later, so that commit failed to collect. The fixtures moved
to the commit whose guard consumes them. The reasoning that had passed for
verification — that no pinned count moves, so every commit is green — was true
about the counts and silent about module ordering.

## Results

| | before | after |
|---|---|---|
| specified-state search, 10 units | 55 s | 0.21 s |
| specified-state search, 11 units | 279 s | 0.5 s |
| specified-state search, 14 units | ~10 h (extrapolated) | 7.0 s |
| specified-state search, 16 units | ~10 d (extrapolated) | 64.2 s |
| peak memory, 14 units | 1.23 GiB | 0.14 GiB |
| peak memory, 16 units | out of memory (~48 GiB) | 0.19 GiB |

No φ value changes. Fast lane 4296 passed, slow lane 254 passed, docs build clean.

## Not done

A fast `FrozenMap.__eq__` (measured, no benefit, costs guard generality). An
audit of all 38 custom `__hash__` implementations by hand — the registry covers
those that are actually cache keys, and the discovery test surfaces any new
one. Wall-time budget assertions on large fixtures — the deterministic counters
cover the same ground without CI flakiness.
