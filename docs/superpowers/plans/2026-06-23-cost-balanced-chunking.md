# Cost-Balanced Chunking (B18) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate `ChunkingPolicy.size_func` so parallel work is split into cost-balanced chunks (LPT) instead of equal-count, and floor the chunk count at `num_workers`, eliminating the straggler on heterogeneous workloads.

**Architecture:** A pure `cost_balanced_partition(weights, k)` helper produces an index partition; `LocalMapReduce._get_chunks` computes `k = max(ceil(total/chunksize), num_workers)` and partitions either by cost (when a `size_func` is given) or evenly, applying the same index partition to every iterable. Cheap `size_func` closures are wired at the heterogeneous call-sites. Chunking never changes results (N2 guards it).

**Tech Stack:** Python 3.12+, `heapq`, loky (process pool), pytest + Hypothesis, ASV, uv.

## Global Constraints

- Python 3.12+ only. **No change to any computed result, measure, or config default** — chunking is result-invariant.
- Run commands with `uv run`. Full verification is `uv run pytest` with **no path argument** (collects `pyphi/` doctests).
- Primary guards after every task: N2 (`test/test_parallel_equals_sequential.py`), goldens (`test/test_golden_regression.py`), perf gate (`test/test_perf_counters.py`).
- `size_func` is evaluated in the **parent process** on already-materialized items, so every cost function MUST be cheap (no φ computation, no expensive recomputation) and self-contained.
- `size_func` controls chunk *contents* only; the `num_workers` floor controls chunk *count* and applies on **both** paths (default even-count and cost-balanced).
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- Pre-commit runs ruff + pyright and aborts when ruff reformats; re-`git add` and re-commit. Never `--no-verify`. `scripts/` is not T201-exempt; `test/` is pyright-excluded.
- Stage only files this plan touches; never `git add -A` (branch `2.0` is shared).

---

## File Structure

- `pyphi/parallel/chunking.py` — **new**: the pure `cost_balanced_partition` and `even_partition` helpers (no PyPhi imports).
- `pyphi/parallel/__init__.py` — `map_reduce()` gains `size_func`; an `ordered`+`size_func` guard.
- `pyphi/parallel/backends/local_process.py` — `LocalMapReduce` gains `size_func`; `_get_chunks` reworked to floor the count and apply an index partition; `LocalProcessScheduler.map_reduce` threads `size_func`.
- `pyphi/relations.py`, `pyphi/formalism/queries.py`, `pyphi/formalism/iit3/__init__.py` — pass a `size_func` closure at the heterogeneous call-sites.
- `benchmarks/benchmarks/chunking.py` — **new**: the A/B and count-floor benchmarks.
- Tests: `test/test_chunking.py` (**new**, helper units + property), `test/test_parallel.py` / `test/test_scheduler.py` (engine behavior), `test/test_parallel_equals_sequential.py` (N2 with `size_func`).

### Call-site selection (read before Tasks 3–5)

Cost-balancing only helps when items in one `map_reduce` call have *heterogeneous* cost. Verified targets:
- **Relations** (`relations.py:249`) — items are index-tuples of distinctions; cost ≈ `overlap × degree`. Heterogeneous, high value.
- **Purview evaluation** (`queries.py:280`) — items are purviews of varying size; cost ≈ `2^|purview|`. Heterogeneous, cheap-exact.
- **Concept evaluation** (`iit3/__init__.py:145`) — items are mechanisms; cost rises with mechanism size. Heterogeneous; a cheap `2^|mechanism|` proxy.

**Not wired** (intentionally): `queries.py:118` evaluates partitions of a *fixed* (mechanism, purview) — homogeneous cost, so cost-balancing degenerates to even-count. System-partition severed-edge weighting (`iit3:336`/`iit4:1092`) is a possible later extension but its cheap-estimate is less clear-cut; left out of this cut.

---

## Task 1: Pure partition helpers

**Files:**
- Create: `pyphi/parallel/chunking.py`
- Test: `test/test_chunking.py`

**Interfaces:**
- Produces: `cost_balanced_partition(weights: list[float], k: int) -> list[list[int]]` and `even_partition(n: int, k: int) -> list[list[int]]`. Both return a list of index bins partitioning `range(len(weights))` / `range(n)`; no index dropped or duplicated; at most `min(k, n)` non-empty bins.

- [ ] **Step 1: Write failing tests** in `test/test_chunking.py`:

```python
from pyphi.parallel.chunking import cost_balanced_partition, even_partition


def _is_partition(bins, n):
    flat = [i for b in bins for i in b]
    return sorted(flat) == list(range(n))


def test_even_partition_splits_into_k_bins():
    bins = even_partition(10, 3)
    assert len(bins) == 3
    assert _is_partition(bins, 10)
    assert sorted(len(b) for b in bins) == [3, 3, 4]


def test_even_partition_caps_bins_at_n():
    bins = even_partition(2, 5)
    assert _is_partition(bins, 2)
    assert all(b for b in bins)  # no empty bins


def test_cost_balanced_is_a_partition():
    weights = [5.0, 1.0, 1.0, 1.0, 1.0, 5.0]
    bins = cost_balanced_partition(weights, 2)
    assert _is_partition(bins, len(weights))


def test_cost_balanced_separates_heavy_items():
    # two heavy items must not land in the same bin when k=2
    weights = [10.0, 10.0, 1.0, 1.0]
    bins = cost_balanced_partition(weights, 2)
    heavy_bins = [bi for bi, b in enumerate(bins) if 0 in b or 1 in b]
    assert len({bi for bi, b in enumerate(bins) for i in b if i in (0, 1)}) == 2


def test_cost_balanced_clamps_nonpositive_and_nonfinite():
    weights = [0.0, -3.0, float("inf"), float("nan"), 2.0]
    bins = cost_balanced_partition(weights, 2)
    assert _is_partition(bins, len(weights))


def test_k_of_one_returns_single_bin():
    assert cost_balanced_partition([1.0, 2.0, 3.0], 1) == [[0, 1, 2]]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_chunking.py -x -q`
Expected: FAIL — `ModuleNotFoundError: pyphi.parallel.chunking`.

- [ ] **Step 3: Implement** `pyphi/parallel/chunking.py`:

```python
"""Pure index-partition helpers for parallel chunking.

No PyPhi imports: these decide how item indices are grouped into chunks,
either evenly (count-balanced) or by estimated cost (weight-balanced).
"""

from __future__ import annotations

import heapq
import math

_EPS = 1e-12


def even_partition(n: int, k: int) -> list[list[int]]:
    """Split ``range(n)`` into ``min(k, n)`` contiguous, near-equal bins."""
    k = max(1, min(k, n))
    base, extra = divmod(n, k)
    bins: list[list[int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        bins.append(list(range(start, start + size)))
        start += size
    return bins


def cost_balanced_partition(weights: list[float], k: int) -> list[list[int]]:
    """Greedily LPT-pack item indices into ``min(k, n)`` cost-balanced bins.

    Sorts indices by weight descending and assigns each to the currently
    lightest bin. Non-positive / non-finite weights are clamped to a small
    epsilon so every item still lands in exactly one bin.
    """
    n = len(weights)
    k = max(1, min(k, n))
    bins: list[list[int]] = [[] for _ in range(k)]
    heap = [(0.0, i) for i in range(k)]  # (accumulated weight, bin index)
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    for idx in order:
        w = weights[idx]
        if not math.isfinite(w) or w <= 0.0:
            w = _EPS
        acc, b = heapq.heappop(heap)
        bins[b].append(idx)
        heapq.heappush(heap, (acc + w, b))
    return bins
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_chunking.py -x -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Add a property test** to `test/test_chunking.py`:

```python
from hypothesis import given
from hypothesis import strategies as st


@given(
    weights=st.lists(st.floats(min_value=0.01, max_value=1e6), min_size=1, max_size=200),
    k=st.integers(min_value=1, max_value=64),
)
def test_cost_balanced_partition_property(weights, k):
    bins = cost_balanced_partition(weights, k)
    flat = [i for b in bins for i in b]
    assert sorted(flat) == list(range(len(weights)))  # exact partition
    assert len([b for b in bins if b]) <= min(k, len(weights))
```

- [ ] **Step 6: Run + commit**

Run: `uv run pytest test/test_chunking.py -q` → PASS
```bash
git add pyphi/parallel/chunking.py test/test_chunking.py
git commit -m "Add pure cost-balanced and even index-partition helpers"
```

---

## Task 2: Thread `size_func` and the count floor through the engine

**Files:**
- Modify: `pyphi/parallel/__init__.py` (`map_reduce` gains `size_func`; guard)
- Modify: `pyphi/parallel/backends/local_process.py` (`LocalMapReduce.__init__` + `_get_chunks`; `LocalProcessScheduler.map_reduce`)
- Test: `test/test_scheduler.py`

**Interfaces:**
- Consumes: `cost_balanced_partition`, `even_partition` (Task 1); `get_num_processes` (existing).
- Produces: `map_reduce(..., size_func: Callable[[Any], float] | None = None)`; `LocalMapReduce(..., size_func=None)`.

- [ ] **Step 1: Write failing tests** in `test/test_scheduler.py`:

```python
def test_map_reduce_size_func_matches_cost_blind_results():
    from pyphi.parallel import map_reduce

    items = list(range(40))
    blind = map_reduce(_square, items, chunksize=4)
    weighted = map_reduce(_square, items, chunksize=4, size_func=lambda x: x + 1)
    assert sorted(blind) == sorted(weighted)  # chunking never changes results


def test_map_reduce_size_func_with_ordered_raises():
    from pyphi.parallel import map_reduce

    with pytest.raises(ValueError, match="size_func.*ordered"):
        map_reduce(_square, [1, 2, 3], size_func=lambda x: x, ordered=True)


def test_map_reduce_zipped_iterables_stay_aligned_under_size_func():
    from pyphi.parallel import map_reduce

    a = list(range(20))
    b = [x * 10 for x in a]
    # fn returns (a_i, b_i); each pair must be the originally zipped pair
    out = map_reduce(_pair, a, b, chunksize=3, size_func=lambda x: x + 1)
    assert sorted(out) == sorted((x, x * 10) for x in a)
```

Add the module-level helper `def _pair(x, y): return (x, y)` near `_square`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_scheduler.py -k "size_func" -x -q`
Expected: FAIL — `map_reduce() got an unexpected keyword argument 'size_func'`.

- [ ] **Step 3: Add `size_func` + guard to `map_reduce`** (`pyphi/parallel/__init__.py`). Add the parameter to the signature (after `map_kwargs`):

```python
    map_kwargs: dict[str, Any] | None = None,
    size_func: Callable[..., float] | None = None,
    backend: str = "auto",
```
Immediately after the docstring, before the `iterables = ...` line, add the guard:
```python
    if size_func is not None and ordered:
        raise ValueError(
            "size_func cost-balancing reorders items across chunks and is "
            "incompatible with ordered=True"
        )
```
In the parallel branch, pass `size_func` into the `ChunkingPolicy`:
```python
        chunking=ChunkingPolicy(
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
            size_func=size_func,
        ),
```

- [ ] **Step 4: Thread `size_func` into `LocalMapReduce`** (`backends/local_process.py`). Add the constructor parameter and store it:
```python
        chunksize: int,
        sequential_threshold: int = 1,
        size_func: Callable[..., float] | None = None,
        shortcircuit_func: Callable = false,
```
```python
        self.chunksize = chunksize
        self.sequential_threshold = sequential_threshold
        self.size_func = size_func
```
In `LocalProcessScheduler.map_reduce`, pass it through (it already builds `LocalMapReduce`):
```python
            chunksize=chunksize,
            sequential_threshold=chunking.sequential_threshold,
            size_func=chunking.size_func,
```

- [ ] **Step 5: Rework `_get_chunks`** (`backends/local_process.py`) to floor the count and apply an index partition. Replace the body from "Chunk each iterable" onward:

```python
        # Chunk each iterable and zip them together
        if not materialized or not materialized[0]:
            return

        from pyphi.parallel.chunking import cost_balanced_partition
        from pyphi.parallel.chunking import even_partition

        n = len(materialized[0])
        k = max(math.ceil(n / self.chunksize), get_num_processes())
        if self.size_func is not None:
            weights = [self.size_func(x) for x in materialized[0]]
            index_bins = cost_balanced_partition(weights, k)
        else:
            index_bins = even_partition(n, k)

        for indices in index_bins:
            if not indices:
                continue
            yield tuple([it[i] for i in indices] for it in materialized)
```
Add `import math` at the top of the file if not present.

- [ ] **Step 6: Run to verify pass**

Run: `uv run pytest test/test_scheduler.py -k "size_func" -x -q`
Expected: PASS (3 tests).

- [ ] **Step 7: Run the guards** (the count floor changes chunk boundaries for every parallel workload)

Run: `uv run pytest test/test_parallel_equals_sequential.py test/test_golden_regression.py test/test_perf_counters.py test/test_parallel.py test/test_scheduler.py -q`
Expected: PASS. (If any N2 test fails, the index partition is misapplied across iterables — revisit Step 5.)

- [ ] **Step 8: Commit**

```bash
git add pyphi/parallel/__init__.py pyphi/parallel/backends/local_process.py test/test_scheduler.py
git commit -m "Cost-balanced chunking engine: size_func + num_workers count floor"
```

---

## Task 3: Relations cost function

**Files:**
- Modify: `pyphi/relations.py` (`all_relations`)
- Test: `test/test_parallel_equals_sequential.py`

**Interfaces:**
- Consumes: `map_reduce(size_func=...)` (Task 2). Items are index-tuples; the closure captures the distinctions' precomputed `purview_union` sets.

**Correctness oracle:** relations are part of the `phi_structure` computation, so the existing **golden suite** (`test/test_golden_regression.py`, IIT-4 fixtures) and the **`phi_structure` N2 path** already compare full relation sets to fixed values. Wiring the cost function and re-running those is the test — no bespoke relations test needed (and avoids guessing the relations entry-point API).

- [ ] **Step 1: Capture the current golden state** so any divergence is caught:

Run: `uv run pytest test/test_golden_regression.py -q`
Expected: PASS (baseline — relation goldens currently green).

- [ ] **Step 2: Add the cost closure and wire it** in `pyphi/relations.py` `all_relations`, just before the `map_reduce(` call:

```python
    purview_unions = [d.purview_union for d in distinctions]

    def _relation_cost(combination):
        overlap = set.intersection(*(purview_unions[i] for i in combination))
        return len(overlap) * len(combination)
```
and pass it:
```python
    result = map_reduce(
        worker,
        combinations,
        desc="Evaluating relations",
        size_func=_relation_cost,
        **pkwargs,  # type: ignore[arg-type]
    )
```

- [ ] **Step 3: Run goldens + the phi_structure N2 path** (relations are part of `phi_structure`, so these compare full relation sets)

Run: `uv run pytest test/test_golden_regression.py test/test_parallel_equals_sequential.py -q`
Expected: PASS (relation goldens byte-identical; N2 holds with cost-balancing active on the relations level).

- [ ] **Step 4: Commit**

```bash
git add pyphi/relations.py
git commit -m "Wire overlap×degree cost function at the relations call-site"
```

---

## Task 4: Purview-evaluation cost function

**Files:**
- Modify: `pyphi/formalism/queries.py` (the purview search at line ~280)
- Test: covered by existing IIT-4 SIA N2 + goldens

**Interfaces:**
- Items are purviews (tuples of node indices). Cost ≈ `2^|purview|` (the repertoire state space over the purview). Cheap and exact.

- [ ] **Step 1: Add the cost function and wire it** in `pyphi/formalism/queries.py`, at the `mip_results = map_reduce(_find_mip, purviews_list, ...)` call:

```python
    mip_results = map_reduce(
        _find_mip,
        purviews_list,
        total=len(purviews_list),
        desc="Evaluating purviews",
        size_func=lambda purview: 2 ** len(purview),
        **parallel_kwargs,
    )
```

- [ ] **Step 2: Run the IIT-4 N2 + goldens** (the purview search runs under IIT-4 SIA)

Run: `uv run pytest test/test_parallel_equals_sequential.py -k iit4 -q && uv run pytest test/test_golden_regression.py -q`
Expected: PASS (results identical).

- [ ] **Step 3: Commit**

```bash
git add pyphi/formalism/queries.py
git commit -m "Wire 2^|purview| cost function at the purview-evaluation call-site"
```

---

## Task 5: Concept-evaluation cost function

**Files:**
- Modify: `pyphi/formalism/iit3/__init__.py` (the concept search at line ~145)
- Test: `test/test_parallel_equals_sequential.py` (covered by existing IIT-3 SIA/CES N2)

**Interfaces:**
- Items are mechanisms (tuples of node indices). Cost rises with mechanism size; use the cheap proxy `2^|mechanism|` (larger mechanisms span larger cause/effect repertoires). A directional estimate is sufficient — N2 guarantees results are unaffected.

- [ ] **Step 1: Add the cost function and wire it** in `pyphi/formalism/iit3/__init__.py`, at the `concepts = map_reduce(compute_concept, mechanisms, ...)` call:

```python
    concepts = map_reduce(
        compute_concept,
        mechanisms,
        map_kwargs={  # type: ignore[arg-type]
            "purviews": purviews,
            "cause_purviews": cause_purviews,
            "effect_purviews": effect_purviews,
            "directions": directions,
        },
        reduce_func=reduce_func,
        desc="Computing concepts",
        total=total,
        size_func=lambda mechanism: 2 ** len(mechanism),
        **parallel_kwargs,  # type: ignore[arg-type]
    )
```

- [ ] **Step 2: Run the IIT-3 N2 (CES path) + goldens**

Run: `uv run pytest test/test_parallel_equals_sequential.py -k iit3 -q && uv run pytest test/test_golden_regression.py -q`
Expected: PASS (CES identical with cost-balancing on the concept level).

- [ ] **Step 3: Commit**

```bash
git add pyphi/formalism/iit3/__init__.py
git commit -m "Wire 2^|mechanism| cost proxy at the concept-evaluation call-site"
```

---

## Task 6: Benchmarks (cost-balancing A/B + count-floor)

**Files:**
- Create: `benchmarks/benchmarks/chunking.py`

**Interfaces:**
- Consumes: the golden fixtures via `benchmarks/benchmarks/_fixtures.py` (`FIXTURES_BY_NAME`, `build_system`); `pyphi.parallel.map_reduce`.

- [ ] **Step 1: Add the count-floor benchmark** (`benchmarks/benchmarks/chunking.py`): a synthetic homogeneous parallel `map_reduce` over a CPU-bound function, parametrized by chunk count (`few` vs `num_workers`), measuring wall time. This isolates the floor:

```python
"""Cost-balanced chunking benchmarks (wall-time)."""

from __future__ import annotations

import math

from pyphi import config
from pyphi.parallel import map_reduce
from pyphi.parallel.backends.local_process import get_num_processes


def _work(x: int, iters: int) -> float:
    s = 0.0
    for i in range(iters):
        s += math.sin(i * 0.5 + x)
    return s


class CountFloor:
    # "few" forces ~2 chunks; "floored" gives ~num_workers chunks
    params = ["few", "floored"]
    param_names = ("regime",)
    timeout = 600.0
    number = 1

    def setup(self, regime: str) -> None:
        self.total = 4000
        self.iters = 4000

    def time_homogeneous(self, regime: str) -> None:
        nw = get_num_processes()
        chunksize = (
            math.ceil(self.total / 2)
            if regime == "few"
            else math.ceil(self.total / nw)
        )
        with config.override(parallel=True, progress_bars=False):
            map_reduce(
                _work, list(range(self.total)),
                map_kwargs={"iters": self.iters},
                parallel=True, chunksize=chunksize,
            )
```

- [ ] **Step 2: Add a heterogeneous relations benchmark** (single-arm wall-time) to the same file. Because the relations `size_func` is wired unconditionally (Task 3), this is **not** an in-process A/B; the cost-balancing win is measured by `asv continuous` across the B18 commit boundary (baseline vs this branch), which the existing nightly workflow already supports. The benchmark just provides the heterogeneous parallel workload to compare:

```python
from ._fixtures import FIXTURES_BY_NAME, build_system


class RelationsParallel:
    """Heterogeneous parallel relations workload.

    Wall-time win from cost-balanced chunking is measured by comparing this
    benchmark across the B18 commit boundary (`asv continuous BASE HEAD`),
    not in-process — the relations cost function is always on post-B18.
    """

    timeout = 600.0
    number = 1

    def setup(self) -> None:
        self.fixture = FIXTURES_BY_NAME["rule110_iit4_2023"]

    def time_relations(self) -> None:
        with self.fixture.config_context(), config.override(
            parallel=True,
            parallel_relation_evaluation={
                **config.infrastructure.parallel_relation_evaluation,
                "parallel": True,
                "sequential_threshold": 1,
            },
        ):
            build_system(self.fixture).sia()  # drives the full relations path
```

Confirm `rule110_iit4_2023` actually parallelizes its relations (enough candidates to exceed one chunk); if not, pick a larger fixture from `FIXTURES_BY_NAME` and note the choice in the docstring.

- [ ] **Step 3: Validate the benchmarks import and run quick**

Run: `cd benchmarks && uv run asv run --python=same --quick --bench chunking 2>&1 | tail -20`
Expected: both benchmarks execute and report times (quick mode, single run).

- [ ] **Step 4: Commit**

```bash
git add benchmarks/benchmarks/chunking.py
git commit -m "Add cost-balanced chunking and count-floor benchmarks"
```

---

## Task 7: Final verification

**Files:** none (verification)

- [ ] **Step 1: Full suite (no path argument — runs doctests)**

Run: `uv run pytest -q`
Expected: PASS, no failures, no collection errors.

- [ ] **Step 2: Confirm the floor and balancing are live**

Run: `uv run python -c "from pyphi.parallel.chunking import cost_balanced_partition; print(cost_balanced_partition([9,1,1,1,9,1], 2))"`
Expected: two bins, the two `9`-weight indices (0 and 4) in different bins.

- [ ] **Step 3: Update ROADMAP** — move the B18 dashboard row to ✅ landed with a one-line summary (size_func + LPT + num_workers floor; cost functions at relations/purview/concept; result-preserving). Commit:
```bash
git add ROADMAP.md
git commit -m "Record B18 cost-balanced chunking as landed in ROADMAP"
```

- [ ] **Step 4:** After all tasks, use superpowers:finishing-a-development-branch to complete.

## Self-review notes (for the implementer)

- The N2 invariant is the correctness backbone: a `size_func` can never change results, only timing. If any N2 test diverges after wiring a cost function, the bug is in the index-partition application (Task 2 Step 5), not the cost function.
- The cost functions are deliberately cheap and approximate. If a benchmark shows a cost function isn't helping, that is a tuning result to report — not a correctness problem and not a reason to revert the engine.
