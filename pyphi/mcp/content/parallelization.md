# Parallelizing PyPhi computations

PyPhi can distribute its expensive inner loops across CPU cores. Parallelism
changes only how work is scheduled, never a result — the same computation
returns the identical value sequentially and in parallel, so wall-clock time
is the only thing at stake. The default backend is a local process pool
(loky, via joblib, a core dependency): **nothing extra needs to be
installed**. Older documentation mentions a `pyphi[parallel]` extra; no such
extra exists.

## The rule: two gates and a threshold

Work at a given level runs in parallel only when **all three** conditions
hold:

1. **The global gate is open**: `pyphi.config.parallel = True`. This flag is
   necessary but *not sufficient* — off, it forces everything sequential;
   on, it merely permits levels that are themselves enabled. On its own it
   parallelizes nothing.
2. **The level's own flag is on**: each level has a config option that is a
   dict with keys `parallel`, `sequential_threshold`, `chunksize`, and
   `progress`, and every level defaults to `parallel: False`.
3. **The workload is big enough**: an enabled level still runs sequentially
   when it has fewer items than its `sequential_threshold`. Dispatching to
   workers only pays once there is roughly 0.5–1 s of total work; the
   thresholds encode each level's measured per-item cost.

The single most common mistake is setting `parallel = True` alone and
concluding that parallelism does not work.

## The seven levels

| Level | Config option | Parallelizes over | Threshold | When it pays |
| --- | --- | --- | --- | --- |
| partitions | `parallel_partition_evaluation` | System partitions in the SIA's search for the minimum information partition | 64 | A large single SIA |
| purviews | `parallel_purview_evaluation` | Candidate purviews for one mechanism | 64 | Mechanisms with many candidate purviews |
| distinctions | `parallel_distinction_evaluation` | Mechanisms, when unfolding the distinctions of a cause-effect structure (both formalisms share this loop) | 64 | A CES over many mechanisms |
| complexes | `parallel_complex_evaluation` | Candidate systems within a substrate | 16 | Complex searches over many candidate systems |
| macro_systems | `parallel_macro_system_evaluation` | Coarse-grained candidate systems in a macro search | 16 | Macro searches |
| mechanism_partitions | `parallel_mechanism_partition_evaluation` | Partitions of a single mechanism | 8192 | Almost never — items cost ~50 µs, and no benefit was measured below 8192 of them |
| relations | `parallel_relation_evaluation` | Relations among distinctions | 8192 | **Never at any measured size.** Relation objects are lazy, so the mapped work is microseconds and the cost is dominated by pickling results back to the parent. Leave it off. |

`chunksize` governs how many items each worker receives per batch (it never
affects results), and `progress` controls that level's progress bar (also
gated by the global `progress_bars` option).

## Which levels to enable for which job

- **One big system irreducibility analysis (SIA)** — enable `partitions` and
  `purviews`.
- **A full Φ-structure or cause-effect structure** — enable `distinctions`
  and `purviews` (plus `partitions` for the embedded SIA).
- **Many systems** (a complex search, or your own loop over substrates or
  states) — enable the *outer* level only (`complexes`, or `macro_systems`
  for macro searches) and leave the inner levels off. Nesting pools
  oversubscribes the cores and runs slower, not faster. For parameter sweeps
  and optimization, `pyphi.sweep(..., parallel=True)` and
  `pyphi.optimize(..., parallel=True)` already implement this pattern: they
  parallelize over whole computations and force each worker's inner loops
  sequential.
- **Relations** — leave sequential (see the table).

## Workers and backend

- `parallel_workers` — how many workers to use. The default `-1` means all
  cores; `-2` means all but one; a positive integer caps the count.
- `parallel_backend` — `"local"` (the default; a loky process pool),
  `"thread"` (a thread pool), or `"auto"` (threads on a free-threaded
  Python, processes otherwise). `"dask"` is an unimplemented stub.

## Recipes

Scoped, for one run (the safe default — settings are restored on exit).
Note that writing a per-level option replaces the whole dict, so
read-modify-write:

```python
opts = {**dict(pyphi.config.infrastructure.parallel_partition_evaluation),
        "parallel": True}
with pyphi.config.override(parallel=True, parallel_partition_evaluation=opts):
    analysis = pyphi.analyze(substrate, state)
```

Your own loop over many systems, parallelized at the outer level only:

```python
# Workers inherit the config snapshot, so inner levels stay sequential.
opts = {**dict(pyphi.config.infrastructure.parallel_complex_evaluation),
        "parallel": True}
with pyphi.config.override(parallel=True, parallel_complex_evaluation=opts):
    result = substrate.complexes(state)
```

Persistent, via `pyphi_config.yml` in the working directory (read at import
time; nested format):

```yaml
infrastructure:
  parallel: true
  parallel_workers: -1
  parallel_partition_evaluation:
    parallel: true
    sequential_threshold: 64
    chunksize: 4096
    progress: false
```

## Through this server's tools

- `analyze(..., parallel=true)` runs that one call on multiple cores at the
  recommended levels (`partitions`, `purviews`, `distinctions`). Pass a list
  to pick levels explicitly (e.g. `parallel=["partitions"]`), `false` to
  force the call fully sequential, and `workers` to cap the worker count.
  The setting is scoped to the call.
- `configure_parallel(...)` sets the server's parallelization configuration
  persistently: `configure_parallel()` alone reports the current state,
  `enable`/`levels`/`workers` change it, and `reset=true` restores the
  defaults. A per-call `analyze` setting takes precedence over it.
- Parallelism divides the constants, not the exponents — the `confirm_large`
  guardrail on `analyze` applies regardless.
