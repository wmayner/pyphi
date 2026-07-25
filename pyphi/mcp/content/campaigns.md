# Campaigns: batch sweeps on an HTCondor cluster

A **campaign** turns a sweep into a directory of self-contained batch jobs
for an HTCondor pool (for example UW–Madison's CHTC). Where the Dask backend
distributes a live computation over a connected worker pool, a campaign has
no live connection at all: every task is a serialized file, every result is
a file, and the scheduler only needs to run `python -m pyphi.campaign run`
inside a PyPhi container. A task is **done exactly when its output file
exists and loads** — status, collection, and resubmission are computed
purely from the campaign directory, from any machine, at any time.

## Workflow

1. **Prepare** (`prepare_campaign` tool, or `pyphi.campaign.prepare` in
   Python): enumerates the sweep cells — substrates × formalisms × subsets ×
   states — estimates each cell's workload, packs cells into cost-balanced
   tasks, and writes the campaign directory:

   - `manifest.json` — axes, per-cell estimates, packing, seed
   - `substrates/` — each substrate serialized once
   - `tasks/task-NNNN.json.gz` — one file per condor job
   - `pyphi.sub`, `run_task.sh`, `remaining.txt` — the condor interface
   - `outputs/`, `logs/` — filled in by the jobs

2. **Submit**: copy the directory to the cluster access point and run
   `condor_submit pyphi.sub`. The submit file queues one job per task id in
   `remaining.txt`, running inside an Apptainer container image (default
   `pyphi.sif`; see the CHTC how-to for building it).

3. **Monitor** (`campaign_status`): classifies every task as done, failed,
   or pending from the output files, and rewrites `remaining.txt` with the
   failed and pending ids. **Resubmission is just `condor_submit pyphi.sub`
   again** — the submit file re-queues exactly what remains.

4. **Collect** (`collect_campaign`): reassembles the per-task outputs into
   the identical `SweepResult` a local sweep over the same axes would have
   produced, registered as a result handle for `inspect`. With missing or
   failed tasks it raises a summary by default; `partial=true` returns the
   completed subset.

## Packing and admission control

Cells are weighted by `estimate_analysis` counts under each cell's
formalism preset. Pass `jobs=K` to pack into exactly K cost-balanced tasks,
or `units_per_job=X` to target X work units per task; with neither, each
cell is its own job (the canonical high-throughput shape). A work unit is
one mechanism partition, and a purview evaluation counts for twelve — its
measured relative cost — so a unit means the same work whatever mix produces
it. `pyphi.cost.units_for_runtime(seconds)` turns a per-shard runtime target
into a `units_per_job` budget, at the reference `SECONDS_PER_UNIT`; each task
output records its own wall and CPU seconds so the constant can be
re-derived per machine. A single cell whose estimate exceeds the infeasibility
threshold triggers a loud warning at prepare time: that cell likely cannot
finish within a typical 72-hour slot and needs narrower axes or a dedicated
fat-node run.

## Failure model

The runner writes one output document per task with a per-cell outcome:
`ok` (with the embedded result), `skipped` (dynamically unreachable state
under `"all"` enumeration, mirroring `sweep()`), or `error` (with the full
traceback). A cell error makes the job exit nonzero — visible in condor
logs — but never discards the task's other cells. Outputs are written
atomically, so a partially transferred file can never count as done, and a
re-run preserves the previous attempt under an `attempt-N` name.

## When to use a campaign vs. the Dask backend

- **Campaign**: many independent cells, pool-scale throughput, no
  interactive session, works on any HTCondor pool with only containers —
  the recommended path on CHTC.
- **Dask backend** (`parallel_backend = "dask"`): interactive distribution
  of one computation over a live worker pool you already have connected —
  requires open ports between scheduler and workers, which CHTC generally
  does not provide.

## Scoped CES campaigns

For one system too large to analyze whole, `prepare_ces_campaign`
distributes the cause-effect structure computation itself. Declare the
combinatorially feasible surface as a **scope** — per-axis constraint
objects such as `{"mechanisms": {"max_order": 3, "containing": ["A"]},
"cause_purviews": {"within": ["A", "B", "C"]}}` — and the planner shards
the scoped work to a per-job budget, descending mechanism → purview-range →
partition-stride only where needed. A scope can also bound purview order
per mechanism order (`max_purview_order_by_mechanism_order`), expressing
order-tied caps exactly. `estimate_cost` accepts the same `scope` to price
the surface first; for very large scoped systems, raise the planning
`limit`. Every shard requests memory sized to its largest purview
repertoire (the `request_memory` argument is the floor), so shards are
not held for exceeding a uniform request. The SIA is sharded too, or
supplied precomputed via `sia_ref`. Collection (`collect_campaign`)
merges shards exactly (tie sets preserved), assembles the
`CauseEffectStructure` through the standard path, and returns a **scope
report**: the computed Σφ_r is an exact lower bound for the full
structure, with certified measured upper bounds on Σφ_r and Φ for what
the scope excluded. Within the scope every value is exact — a scope
narrows the computation, never approximates it. Sweeps of many states or
substrates under one scope, collected as a single `SweepResult`, are a
library-level feature of `pyphi.campaign.prepare_ces`.
