# HTCondor campaign surface — cycle 1: infrastructure + sweep cells

**Date:** 2026-07-20
**Status:** Draft for review

## Context

The Dask backend distributes a live computation over a connected worker pool,
but on CHTC a held pool is unconfirmed (port access) and wasteful for
campaign-scale work. This design adds the HTCondor-native alternative: a
**campaign** is a directory of self-describing task files that plain condor
jobs execute independently, with no long-lived coordinator and no network
connectivity between Python processes. The user runs `condor_submit`
themselves; PyPhi prepares inputs beforehand and collects outputs afterwards.

This is cycle 1 of 2. It builds the campaign infrastructure and the
**sweep-cell** task type (the cartesian sweep, now with a substrates axis).
Cycle 2 adds **CES shards** for a single system (mechanism/purview/partition
ranges with declarative scope) on top of this infrastructure; nothing in
cycle 1 may assume sweep cells are the only task type, but nothing is built
for shards yet either.

### Principles (settled during design)

- **Tasks are data, not closures.** Everything a job needs is serialized via
  `pyphi.serialize` into files; the runner is a fixed entry point.
- **Done = the output file exists and loads.** Status, collection, and
  resubmission are pure functions of the campaign directory. No condor log
  parsing; the same directory answers questions from any machine.
- **Scheduler-neutral core.** Only the generated submit file and wrapper
  script know about HTCondor.
- **Zero new required dependencies.** Everything runs on the standard
  install; condor is only ever invoked by the user.

## Goals

1. `pyphi.sweep()` gains a substrates axis (shared cell model, so a campaign
   is exactly a distributed sweep).
2. `pyphi.campaign.prepare()` materializes sweep cells into a self-contained
   campaign directory with cost-balanced job packing and admission control.
3. A `python -m pyphi.campaign run` entry point executes one task file
   identically in the Apptainer container, a local shell, or a test.
4. `pyphi.campaign.status()` / `collect()` reassemble the exact
   `SweepResult` a local `sweep()` would have returned, and drive
   resubmission of missing/failed tasks.
5. `docs/howto/chtc.md` teaches the campaign workflow.

## Non-goals (cycle 1)

- CES sharding, declarative scope objects, partition-range enumeration
  (cycle 2).
- Multi-core condor jobs (`request_cpus > 1` with in-task parallelism); the
  runner is sequential.
- Any scheduler besides HTCondor; any condor invocation from Python.
- Retrying individual cells within a task (resubmit granularity = task).
- DAGMan, checkpointing, or streaming partial results.

## Verified mechanics (all confirmed by running code on main)

- `estimate_analysis(substrate, subset, compute=...)` works under
  `config.override(**presets.by_name[formalism])` and returns per-axis
  counts; `compute="sia"` counts only the system-partition axis.
- `cost_balanced_partition(weights, k)` (`pyphi/parallel/chunking.py`)
  LPT-packs item indices into `min(k, n)` bins; non-positive weights are
  clamped. This is the packer; no new packing code.
- `pyphi.save`/`pyphi.load` roundtrip `Substrate` and SIA objects through
  `.json.gz` files with φ and equality preserved.
- Config shipping: `config.snapshot().as_overrides()` →
  `msgspec.to_builtins(..., enc_hook=<Mapping→dict, else str>)` → JSON →
  `config.override(**{**snapshot_overrides, **presets.by_name[formalism]})`
  restores the preparing session's full configuration with the cell's
  formalism preset on top. The merge must happen dict-first: passing both
  expansions as separate `**` kwargs collides on shared keys (`precision`),
  and the seven `parallel_*_evaluation` FrozenMap options need the
  mapping-aware encode hook (plain `enc_hook=str` stringifies them and
  breaks restore).

## Design

### 1. Substrates axis on `sweep()`

`sweep()`'s first parameter is renamed `substrate` → `substrates` (a real
rename; keyword callers update — no compatibility alias) and accepts:

- a single `Substrate` (current behavior, unchanged results),
- a sequence of substrates (labeled by integer position), or
- a mapping `{label: substrate}` (labeled by key).

Semantics:

- Cells become `(substrate_label, formalism, subset, state)`. A
  `substrate` index level appears in `SweepResult.df` when more than one
  substrate is given, following the existing conditional-level pattern in
  `_build_df` (single substrate → constant column, as with the other axes).
- Explicit `states`/`subsets` apply to **every** substrate and must be valid
  for each; `"all"` enumerates per substrate, so substrates of different
  sizes and alphabets coexist in one sweep.
- Skip semantics unchanged: `"all"`-enumerated axes skip unreachable states
  (recorded in `skipped`, now as `(substrate_label, formalism, subset,
  state)` 4-tuples); explicit cells fail loud.
- `SweepResult` gains no new fields; `skipped` entries grow the substrate
  element. The serialization schema is updated accordingly.

### 2. `pyphi.campaign` module surface

Module-level functions operating on a campaign directory; no campaign class.

```python
def prepare(
    substrates,                  # Substrate | Sequence[Substrate] | Mapping[str, Substrate]
    *,
    states,                      # tuple | Iterable[tuple] | "all"
    subsets="full",              # "full" | "all" | Iterable[tuple]
    formalisms=None,             # None | Iterable[str]
    compute="sia",               # "sia" | "ces" | callable
    directory,                   # str | Path; created; must not already exist
    jobs=None,                   # int: pack into this many cost-balanced jobs
    units_per_job=None,          # float: target work units per job
    infeasible_threshold=1e9,    # warn when one cell's estimate exceeds this
    strict=False,                # True: infeasible cell is an error
    container_image="pyphi.sif",
    request_memory="4GB",
    request_disk="4GB",
    seed=None,                   # stamped into provenance at collect time
) -> CampaignStatus
```

- Enumerates exactly the cells `sweep()` would run (same normalization
  helpers — shared, not copied).
- Estimates each cell under its formalism preset; packs (section 4); writes
  the directory (section 3); returns a `CampaignStatus`.
- Refuses to write into an existing directory (no clobbering; a campaign
  directory is created exactly once).
- `compute` callables must be importable by the runner (a module-level
  function available in the container's environment); `prepare` records the
  callable's `module:qualname` and rejects lambdas and locals with a clear
  error. Cost estimation for callables falls back to uniform weights
  (recorded as such in the manifest).

```python
def status(directory) -> CampaignStatus
```

- Scans `outputs/`: **done** = output file exists, loads, and every cell in
  it is `ok` or `skipped`; **failed** = output loads but contains an error
  cell, or an attempt was recorded without a loadable output; **pending** =
  no output. Rewrites `remaining.txt` to the pending + failed task ids, so
  resubmission is simply running `condor_submit pyphi.sub` again.

```python
def collect(directory, partial=False) -> SweepResult
```

- Loads all task outputs, reassembles cells in the deterministic preparation
  order, and builds the `SweepResult` with the same row-extraction and
  DataFrame code as a local sweep. With missing/failed tasks: raise with a
  per-task summary by default; `partial=True` returns the result built from
  available cells (skipped cells folded into `skipped` as in a local sweep)
  and emits a warning summarizing the missing tasks — full detail via
  `status()`.
- Stamps `seed` from the manifest into result provenance, mirroring
  `sweep()`.

`CampaignStatus` is a small frozen dataclass: task counts by state
(`done`/`failed`/`pending`), cell counts, total estimated work units, and
the per-task id lists. It renders through the standard display machinery.

### 3. Campaign directory format

```
campaign/
  manifest.json        axes as given, cell list, per-cell estimates,
                       packing (cells per task, per-task unit totals),
                       formalism preset names, seed, pyphi version,
                       provenance (git info, timestamp)
  substrates/
    substrate-<label>.json.gz     each substrate serialized once
  tasks/
    task-0000.json.gz             one per condor job
  outputs/                        runner writes task-<id>.json.gz here
  logs/                           condor stdout/err/log land here
  remaining.txt                   task ids not yet done (initially all)
  run_task.sh                     wrapper the condor job executes
  pyphi.sub                       generated submit file
```

- **`manifest.json`** is plain JSON, human-inspectable bookkeeping. It
  records every per-cell estimate and per-task total — the raw data behind
  the packing summary.
- **Task files** are `pyphi.serialize` documents (new schema, tag
  `campaign_task`): task id, compute spec (`"sia"`/`"ces"`/callable
  reference), the JSON-wire config overrides described above, the ordered
  cell list (substrate label, formalism name, subset, state), and the
  skip-uncomputable flag (computed at prepare time from whether any axis
  was `"all"`).
- **Task outputs** are `pyphi.serialize` documents (tag
  `campaign_task_output`): the task id and one entry per cell —
  `status` (`ok` | `skipped` | `error`) with the embedded serialized result
  object for `ok`, nothing extra for `skipped`, and the traceback string for
  `error`. Embedding reuses the existing registered result schemas.
- Substrate files are ordinary `pyphi.save` outputs, loadable on their own.

### 4. Packing and admission control

- Per-cell weight = the sum of the non-`None` counted axes of that cell's
  `AnalysisEstimate` (`system_partitions`, `purview_evaluations`,
  `mechanism_partition_sweeps`), estimated under the cell's formalism
  preset. Capped estimates are used as-is and flagged in the manifest as
  lower bounds.
- `jobs=K` → `cost_balanced_partition(weights, K)`.
  `units_per_job=X` → `K = ceil(total_units / X)`, then the same call.
  Neither → one cell per task (canonical HTC shape). Both → `ValueError`.
- Within a task, cells run in preparation order (deterministic).
- **Admission control:** any single cell whose weight exceeds
  `infeasible_threshold` triggers a loud warning naming the cell and its
  estimate; `strict=True` raises instead. Default `1e9`: above ~10⁹ work
  units a single cell exceeds CHTC's 72-hour slot unless per-unit cost is
  below ~0.26 ms, so this is the "you probably want scope shaping or a fat
  node" line. The threshold, like the estimates, is recorded in the
  manifest.

### 5. Runner

`python -m pyphi.campaign run TASK_FILE [--substrates DIR] [--outputs DIR]`

- Defaults: `--substrates substrates`, `--outputs .` — matching the condor
  scratch directory layout after input transfer, where the wrapper runs the
  task from the job's working directory. Tests and local runs pass explicit
  directories.
- For each cell, in order: `config.override(**{**wire_overrides,
  **presets.by_name[formalism]}, parallel=False, progress_bars=False)`,
  build the `System`, dispatch the compute, append the cell entry. The
  existing per-cell function from `pyphi.sweep` is reused, not copied.
- Unreachable states follow the task's skip flag (skip → `skipped` entry;
  explicit-axes tasks record an `error`).
- Any other exception in a cell → `error` entry with traceback; the runner
  continues with remaining cells, writes the output document, and exits
  nonzero (so condor's logs show the failure) — the output document is
  written in every case.
- **Atomic writes:** output is written to a temporary name in the target
  directory and `os.replace`d into place, so a partially transferred or
  interrupted write can never satisfy "exists and loads".
- **Attempts:** if the output file already exists (a resubmitted task), the
  runner renames the previous file to `task-<id>.attempt-<n>.json.gz`
  before writing, preserving every attempt's raw record. `status()` and
  `collect()` read only the unsuffixed file.

### 6. Condor emitter

`prepare()` generates two files:

`run_task.sh` (executed inside the container):

```bash
#!/bin/bash
set -e
exec python -m pyphi.campaign run tasks/task-$1.json.gz --substrates substrates --outputs .
```

`pyphi.sub`:

```
universe            = container
container_image     = <container_image>
executable          = run_task.sh
arguments           = $(task_id)
transfer_input_files = tasks/task-$(task_id).json.gz, substrates/
transfer_output_remaps = "task-$(task_id).json.gz = outputs/task-$(task_id).json.gz"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
request_cpus        = 1
request_memory      = <request_memory>
request_disk        = <request_disk>
log                 = logs/task-$(task_id).log
output              = logs/task-$(task_id).out
error               = logs/task-$(task_id).err
queue task_id from remaining.txt
```

The user's whole cluster interaction is: copy the campaign directory to the
access point, `condor_submit pyphi.sub`, and later run `status()`/
`collect()` on the directory (locally after copying back, or on the access
point). Resubmission after failures: `status()` (rewrites `remaining.txt`),
then `condor_submit pyphi.sub` again.

### 7. Documentation

- **New dedicated how-to page** `docs/howto/campaigns.md` (added to the
  how-to toctree): the campaign workflow start to finish — prepare locally,
  ship the directory to the access point, `condor_submit`, monitor with
  `status()`, resubmit, `collect()` — including packing/admission-control
  usage and a worked example. Cluster-specific setup (accounts, CHTC
  systems, container build) stays in `docs/howto/chtc.md`, which
  cross-links the campaign page and retains the manual per-job recipe as a
  short "rolling your own" note.
- `docs/howto/parallel.md` cluster section: one-line pointer to campaigns.

### 8. MCP server integration

Three new tools in `pyphi/mcp/server.py`, thin wrappers over the library
functions (the runner is not an MCP concern — jobs execute on the cluster):

- `prepare_campaign(substrate_handles, states, subsets, formalisms,
  compute, directory, jobs, units_per_job, …)` → prepares the directory and
  returns the `CampaignStatus` summary (cell/task counts, total units,
  admission-control warnings).
- `campaign_status(directory)` → the `CampaignStatus` summary.
- `collect_campaign(directory, partial=False)` → collects, registers the
  `SweepResult` as a result handle (the existing handle registry), and
  returns the summary plus the handle for `inspect`.

Plus a new reference content page `pyphi/mcp/content/campaigns.md`
(concepts, workflow, when to use a campaign vs. the Dask backend) and a
pointer from `pyphi/mcp/content/parallelization.md`. MCP tool tests follow
the existing `test/mcp` patterns.

## Error handling summary

| Failure | Where caught | Behavior |
|---|---|---|
| Cell raises during compute | runner | `error` entry + traceback; runner continues, exits nonzero |
| Unreachable state, `"all"` axes | runner | `skipped` entry (mirrors `sweep()`) |
| Unreachable state, explicit axes | runner | `error` entry (mirrors `sweep()`'s fail-loud) |
| Output missing / truncated | `status`/`collect` | task pending/failed; atomic write prevents truncated files loading |
| Task failed | `status` | listed in `remaining.txt`; resubmit re-runs whole task |
| `collect` with gaps | `collect` | raise with per-task summary; `partial=True` returns available cells |
| Single cell over threshold | `prepare` | loud warning (or error with `strict=True`) |
| Callable compute not importable | `prepare` | error naming the requirement |
| Campaign directory exists | `prepare` | error; directories are never overwritten |

## Testing

Headline invariant — **campaign ≡ local sweep**, no condor anywhere:

1. `prepare()` into a tmpdir (2 small substrates × formalisms × states,
   packed into few jobs).
2. Execute every task via `subprocess`:
   `python -m pyphi.campaign run tasks/task-NNNN.json.gz --substrates … --outputs …`.
3. `collect()` and assert the `SweepResult` equals a local
   `sweep()` over the same axes: identical DataFrames (values and index)
   and per-cell φ.

Around it:

- **Cell parity:** the cells `prepare()` enumerates are exactly the cells
  `sweep()` runs, for each axis form (explicit, `"all"`, dict-labeled
  substrates).
- **Packing:** deterministic for fixed inputs; `jobs`/`units_per_job`
  arithmetic; single-cell default; both-given `ValueError`; weights recorded
  in the manifest.
- **Admission control:** warning fires for an over-threshold cell (small
  threshold in the test); `strict=True` raises.
- **Roundtrips:** task file and task output through `pyphi.serialize`;
  config restoration installs the right formalism in a subprocess.
- **Failure paths:** erroring compute callable → error entry + nonzero
  exit + `status()` marks failed + `partial=True` collect works; deleted
  output → pending; `remaining.txt` rewriting; attempt renaming on re-run.
- **Submit file:** generated content contains the queue line, remaps, and
  parameters (string checks, not condor execution).
- **Sweep substrates axis:** new index level appears only with >1
  substrate; per-substrate `"all"` enumeration with different sizes; 4-tuple
  skip records; single-substrate results byte-identical to current behavior.

## Deliverables

- `pyphi/sweep.py`: substrates axis.
- `pyphi/campaign/__init__.py`: `prepare`, `status`, `collect`,
  `CampaignStatus`, runner (`__main__` dispatch via
  `python -m pyphi.campaign`). A package, not a module, so the CES-sharding
  cycle (`scope.py`, `shards.py`, `merge.py`; see the companion spec
  `2026-07-20-ces-sharding-design.md`) slots in without a rename. Task
  files carry a `kind` discriminator (`"sweep_cells"` now; the shard kinds
  later); nothing in this cycle may assume `sweep_cells` is the only kind.
- `pyphi/serialize/schema.py` + `convert.py`: `campaign_task`,
  `campaign_task_output` schemas; updated sweep-result schema (skip
  4-tuples).
- `pyphi/__init__.py`: lazy-import registry entries.
- `pyphi/mcp/server.py`: `prepare_campaign`, `campaign_status`,
  `collect_campaign` tools; `pyphi/mcp/content/campaigns.md` +
  parallelization-content pointer.
- `test/campaign/` (mirrors the package), additions to `test/test_sweep.py`
  and the `test/mcp` suite.
- Docs: new `docs/howto/campaigns.md` (+ toctree entry),
  `docs/howto/chtc.md` rework, `docs/howto/parallel.md` pointer.
- Changelog fragments: campaign feature, sweep-substrates feature.
- ROADMAP: P11 row update on merge.

## Accepted simplifications

- Resubmit granularity is the task: a failed cell re-runs its whole task.
  Mitigation: keep tasks modest via `units_per_job`. Per-cell resume can be
  added inside the runner later without changing the directory format.
- The runner is sequential (`request_cpus = 1`). Multi-core fat tasks are a
  later addition riding the same task format.
- Work units are not calibrated to wall time; the manifest reports units and
  the docs give the measured rule-of-thumb range. Calibration (a timed probe
  cell) can be added to `prepare()` later.
- `remaining.txt` is the only mutable state, rewritten whole by `status()`.
  Concurrent `status()` calls on the same directory are not defended against.
