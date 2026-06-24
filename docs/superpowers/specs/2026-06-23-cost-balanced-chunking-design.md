# Cost-Balanced Chunking (B18) — Design

**Status:** proposed
**Author:** brainstorming session, 2026-06-23
**Sub-project:** 2 of 2 (depends on *Unify the Parallel Engine*, which landed)

## Goal

Activate the dormant `ChunkingPolicy.size_func` so that parallel work is split
into chunks of roughly equal *estimated cost* rather than equal item count.
Under PyPhi's heterogeneous workloads (a relation over high-order distinctions,
or a partition over a large purview, costs orders of magnitude more than a small
one), equal-count chunks let one worker draw all the expensive items and become
a straggler while the rest idle. Cost-balanced chunks eliminate that straggler.

Chunking never changes results, so this is guarded for free by the N2
invariant (`parallel ≡ sequential`).

## Background: the path this lands on

After unification there is a single flat map-reduce path:

```
map_reduce(fn, items, ...) → ChunkingPolicy → LocalProcessScheduler.map_reduce
  → compute_chunksize (sets chunksize) → LocalMapReduce._get_chunks → futures
```

- `ChunkingPolicy` already carries the dormant fields `size_func:
  Callable[[Any], float] | None` and `target_seconds: float` (`scheduler.py`).
  Nothing reads `size_func` yet.
- `LocalMapReduce._get_chunks` chunks **each iterable separately** with
  `chunked_even(it, self.chunksize)` and zips the results, so parallel
  iterables (`more_items`) stay aligned only because every iterable is cut at
  the same positions.
- The number of chunks is `ceil(total / chunksize)`, where `chunksize` is
  explicit (from each level's config — relations `4096`, concepts/purviews
  `256`) or sampled by `compute_chunksize` when unspecified.

## Design

`size_func(item) → float` returns a cheap, relative a-priori cost estimate. It
controls only **which items are grouped together**, never the chunk count or
the results.

This work separates two independent concerns that the old chunking conflated:
**chunk count** (how many chunks — a general parallelism question) and **chunk
contents** (which items group together — where cost-balancing lives).

#### Chunk count (general; applies to both paths)

The chunk count becomes `k = max(ceil(total / chunksize), num_workers)` whenever
a workload parallelizes — for the default equal-count path *and* the
cost-balanced path. Today the count is just `ceil(total / chunksize)`, so with a
coarse config chunksize a medium workload produces fewer chunks than there are
workers and leaves cores idle (the relations level, `chunksize=4096`, does not
saturate 11 cores until ~45,000 candidates). Flooring at `num_workers` keeps
every core busy. This is result-preserving (N2) — only wall-clock changes — and
is a deliberate change to default behavior, justified because PyPhi's per-item
work (repertoire and relation evaluations) dominates the added per-chunk
dispatch overhead even at the smaller chunk sizes the floor produces. The
existing `sequential_threshold` still gates whether a workload parallelizes at
all, so genuinely tiny workloads are unaffected.

#### Chunk contents (cost-balanced; needs `size_func`)

With `k` fixed above:

1. Compute `weights = [size_func(x) for x in items]`.
2. **Longest-processing-time-first (LPT) bin-packing into `k` bins:** sort item
   indices by weight descending; walk them, assigning each index to the bin
   with the least accumulated weight (a min-heap keyed by bin weight). This is
   the standard makespan-minimizing heuristic.
3. The result is an **index partition** — a list of `k` index lists. Apply the
   *same* partition to every iterable so zipped `more_items` stay aligned.

`size_func=None` packs the same `k` bins by *equal count* instead (the existing
`chunked_even` behavior, just at the floored count). So `size_func` controls
only contents, never count.

### Where it lives

`LocalMapReduce._get_chunks` is reworked to:

1. compute the floored count `k = max(ceil(total / chunksize), num_workers)`
   (`num_workers` via the existing `get_num_processes()`),
2. build an **index partition** into `k` bins — `cost_balanced_partition(weights,
   k)` (LPT) when `size_func` is set, otherwise an even contiguous split into
   `k` bins, and
3. apply that single index partition to every materialized iterable, then zip,
   so `more_items` stay aligned.

`cost_balanced_partition(weights, k) -> list[list[int]]` is a pure,
unit-testable helper with no PyPhi imports. The even-split path is the
structural successor to today's `chunked_even`, now at the floored count.
`size_func` threads through: `map_reduce(size_func=...)` →
`ChunkingPolicy(size_func=...)` → `LocalProcessScheduler.map_reduce` →
`LocalMapReduce(size_func=...)`.

`size_func` is evaluated in the parent process (the items are already
materialized there for chunking), so it must be cheap and must not require
worker state.

Because the chunk *count* now changes for every parallel workload (not just the
`size_func` ones), the default path is **no longer byte-identical to today** —
but it remains result-identical (N2), and the change is the intended
core-utilization improvement.

### Public surface

`map_reduce()` gains one parameter: `size_func: Callable[[Any], float] | None =
None`. The thread and dask backends ignore it (the thread backend does not
chunk; dask is a stub) — only the process backend honors it, which is where
parallel cost matters.

## Cost functions at the hot call-sites

Each call-site passes a `size_func` closure capturing the context it needs. The
formulas are the considered ones from the ROADMAP; only *relative* ordering
matters (absolute scale is irrelevant to LPT), so they need to be directionally
right, not calibrated.

### 1. Relations (`pyphi/relations.py`)

Items are candidate distinction-combinations. Cost rises with how many
distinctions are related and how large their shared purview overlap is:

```
size_func(combination) = overlap_size × degree
```

where `degree = len(combination)` and `overlap_size` is the number of units in
the intersection of the combination's purviews (the relation is computed over
that overlap). Both are already available where candidates are generated.

### 2. Partition / purview search (`pyphi/formalism/queries.py`)

Items are system or mechanism partitions. Cost scales with how much the
partition severs (more cut edges → more repertoire recomputation):

```
size_func(partition) = severed_edge_count(partition)
```

the number of directed edges cut by the partition.

### 3. Concept / distinction evaluation (`pyphi/formalism/iit3`, `iit4`)

Items are mechanisms. Cost scales with the purview search space and the state
space each repertoire spans:

```
size_func(mechanism) = |potential_purviews(mechanism)| × alphabet_product
```

where `alphabet_product` is the product of the units' alphabet sizes (`2` for
binary units, larger for multivalued). `potential_purviews` is already computed
on this path; the closure captures the system/direction.

## Measurement

Two distinct effects to measure — the count floor (general) and cost-balancing
(heterogeneous only) — neither of which the current benchmark suite exercises.

- **Cost-balancing A/B** (`benchmarks/benchmarks/`): run a relations computation
  with `parallel=True` on a fixture whose distinctions have a wide purview-size
  spread, parametrized `size_func` off vs on. The "off" arm is the baseline and
  the "on" arm the cost-balanced result — same binary, same machine, same
  fixture, so the only variable is the packing policy. The tightest possible
  paired measurement of the cost-balancing win.
- **Count-floor check** (homogeneous): a `parallel=True` workload sized in the
  medium range (more than one `chunksize` but fewer than `num_workers ×
  chunksize`) with uniform per-item cost, measured with the floor vs without.
  This isolates the count floor from cost-balancing and confirms it helps (or,
  if it regresses because dispatch overhead dominates, that result is reported
  honestly and the floor reconsidered rather than assumed).
- **Regression safety**: the deterministic perf gate (call counts) is unaffected
  by chunking and stays green; the N2 invariant confirms results are identical
  with the floor on/off and with `size_func` on/off.

## Error handling

- A `size_func` that raises propagates as an ordinary exception from the parent
  (before any dispatch), surfaced to the caller — no partial parallel state.
- `size_func` returning a non-positive or non-finite weight is treated as a
  small positive epsilon so the LPT heap stays well-ordered (a zero-cost item
  must still land somewhere); this is clamped in `cost_balanced_partition`, not
  at the call-sites.

## Testing strategy

- **`cost_balanced_partition` unit tests**: equal weights reproduce balanced
  counts; a single heavy item lands alone with light items distributed; `k=1`
  returns one bin; empty input returns empty; the union of bins is exactly the
  input index set (a partition — no dropped or duplicated indices).
- **Property test**: for random weights and `k`, every index appears exactly
  once and the max bin weight is within the LPT bound of optimal.
- **N2 invariant**: `parallel ≡ sequential` must hold with `size_func` provided
  at each wired call-site (results identical to cost-blind chunking).
- **Golden suite + perf gate**: unchanged (chunking is not on the counted
  frames; results are byte-identical).
- **Alignment test**: a `map_reduce` over two zipped iterables with a
  `size_func` returns results in which each `fn(a, b)` saw the originally
  paired `(a, b)` — confirming the index partition is applied identically to
  every iterable.
- Full `uv run pytest` (no path argument) before declaring complete.

## Out of scope

- **Re-tuning the per-level `chunksize` config *values*** (e.g. relations
  `4096`) — the `num_workers` floor is a structural minimum on the count, not a
  change to the configured chunk sizes; empirically re-tuning those values is a
  separate question.
- **Cost-sampling changes** — `compute_chunksize` still derives `chunksize` when
  it is unspecified; the floor and `size_func` apply on top of whatever count
  that yields.
- **Thread / dask cost-balancing** — only the process backend chunks; the
  others ignore `size_func`.
- **Any change to computed results, measures, or config defaults.**

## Verification checklist

- [ ] Chunk count is floored at `num_workers` on both paths;
      `map_reduce(size_func=...)` produces cost-balanced chunks and `size_func=None`
      produces even-count chunks at that floored count. Results are identical
      (N2), though chunk boundaries differ from today.
- [ ] Cost functions wired at relations, partition/purview search, and
      concept/distinction evaluation.
- [ ] N2, goldens, and perf gate green.
- [ ] The new A/B benchmark shows the "on" arm reducing wall time on the
      heterogeneous parallel workload (or, if it does not, the result is
      reported honestly rather than the benchmark dropped).
- [ ] Full `uv run pytest` green, including `pyphi/` doctests.
