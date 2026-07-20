# Dask cluster backend + CHTC deployment guide — design

**Roadmap item:** P11 cluster backends (partial: the Dask half; the
HTCondor-native batch surface is recorded as a follow-up).

**Goal:** Fill the `DaskScheduler` stub so that every parallel operation in
the library can distribute across a `dask.distributed` cluster, and document
how to deploy PyPhi on UW–Madison's CHTC — including the paths that need no
Dask at all.

**Motivating use case:** the most common anticipated workload is a single
system whose CES/SIA computation (mechanisms → purviews → partitions) should
spread across cluster machines. All of PyPhi's parallelism already routes
through `pyphi.parallel.map_reduce()` over the `Scheduler` Protocol (16 call
sites: CES distinctions, MICE purview/partition searches, SIA system
partitions, `pyphi.sweep` cells), so a real distributed backend behind the
Protocol serves that case — and sweeps — with no changes to compute code.

## Background

- `pyphi/parallel/backends/dask.py` is a Protocol-conforming stub whose
  `map_reduce` raises `NotImplementedError`. `default_scheduler("dask")`
  already resolves to it via `config.parallel_backend = "dask"`.
- The process backend (`local_process.py`) is the semantic reference:
  cost-sampled chunking (`compute_chunksize`), chunk submission via
  `_process_chunk`, deterministic submission-order collection under
  `ordered`/short-circuit, cancel-remaining on error/short-circuit,
  chunk-level progress updates, and config propagation via
  `_make_worker_fn` (snapshot closure + hash-deduplicated
  `install_snapshot` on the worker side).
- CHTC context (researched 2026-07-20 against chtc.cs.wisc.edu):
  - The **HTC system** (~40k slots, HTCondor, 72 h default job runtime,
    container delivery via Apptainer/Docker) is the right resource for
    PyPhi work; the HPC cluster is for multi-node MPI jobs and CHTC directs
    single-node work to HTC.
  - CHTC's first-party Python-on-the-pool layers are gone: `htmap` was
    archived in March 2026; `dask-chtc` is a dead prototype (v0.1.0,
    single designated submit node, engagement-gated).
  - Generic, maintained tooling exists: `dask_jobqueue.HTCondorCluster`
    spawns Dask workers as ordinary condor jobs.
  - **Port caveat:** most ports on CHTC submit and execute nodes are
    closed. A Dask cluster needs bidirectional scheduler↔worker TCP;
    dask-chtc's single-node restriction suggests port allowances were
    machine-specific. Whether the generic recipe works on a current access
    point must be confirmed with CHTC facilitation; the guide states this
    plainly and provides the questions to ask.

## Backend design (`pyphi/parallel/backends/dask.py`)

`DaskScheduler.map_reduce` mirrors `LocalProcessScheduler.map_reduce`,
reusing its building blocks:

- **Client resolution.** The user creates and connects a
  `dask.distributed.Client` (a `LocalCluster`, an `HTCondorCluster`, or an
  address) before computing; the backend obtains it with
  `distributed.get_client()`. If `distributed` is not importable, raise
  `ImportError` naming the `cluster` extra
  (`pip install "pyphi[cluster]"`). If no client is active, raise a clear
  error instructing the user to create one. `distributed` is imported
  lazily inside `map_reduce`; importing the module stays free.
- **Policies.** Same defaulting as the process backend: `ChunkingPolicy()`,
  `ProgressPolicy()`, `ShortcircuitPolicy()` when `None`;
  `config.snapshot()` when no snapshot is passed.
- **Sequential fallback.** Materialize items; empty input returns
  `reducer([])`. Run sequentially (locally, no cluster round-trip) under
  the same conditions as the process backend: below
  `sequential_threshold`, or when the workload fits in a single chunk
  (the sampled-chunksize fold rule, reused verbatim).
- **Chunking.** Reuse `compute_chunksize` (client-side cost sampling with
  the same call-shape rules for `map_kwargs`/multi-iterable maps) and the
  same chunk construction as `LocalMapReduce._get_chunks` (even or
  cost-balanced index bins). Chunks are submitted as
  `client.submit(_process_chunk, chunk_tuple, wrapped_fn, map_kwargs,
  shortcircuit.func)` futures — `_process_chunk` is imported from
  `local_process` unchanged.
- **Config propagation.** Reuse `_make_worker_fn(fn, snapshot)`. Dask
  workers are processes holding module globals, so the existing
  hash-deduplicated `_apply_snapshot_if_changed` works unchanged. One
  hardening change in `local_process.py`: guard snapshot installation with
  a module-level `threading.Lock`, because Dask workers may run more than
  one thread (the process/thread backends are unaffected: loky workers are
  single-threaded and the thread backend never installs snapshots). The
  guide additionally recommends single-threaded workers
  (`--nthreads 1`), since PyPhi's work is GIL-bound CPU.
- **Collection.** Identical semantics to the process backend:
  - unordered: `distributed.as_completed(futures)`;
  - `ordered` or an active short-circuit predicate: iterate futures in
    submission order, preserving the deterministic sequential-prefix
    guarantee that order-sensitive reductions (tie resolution) rely on;
  - short-circuit hit: cancel remaining futures, fire the callback, stop;
  - any exception: cancel remaining futures, re-raise;
  - progress: client-side `LocalProgressBar`, updated by chunk length,
    closed in a `finally`.
- **Reduction** happens client-side via `reducer` exactly as elsewhere.
- `supports_shared_state` stays `False`.

Module docstring is rewritten to describe the real backend (final-state,
NumPy style; no stub or deferral narrative).

## Packaging

- Activate the placeholder in `pyproject.toml`:
  `cluster = ["dask[distributed]>=2024.1.0", "dask-jobqueue>=0.8.0"]`.
- Add `distributed` to the dev dependency group so the suite exercises the
  backend.
- `dask-jobqueue` is not imported anywhere in `pyphi/`; it is part of the
  extra because the CHTC guide uses it.

## Testing (`test/parallel/test_dask_backend.py`; local only, no cluster)

Module-scoped `LocalCluster` fixture: 2 worker processes,
`threads_per_worker=1`; the whole module skips cleanly when `distributed`
is not installed (`pytest.importorskip`).

- Protocol conformance (`isinstance(DaskScheduler(), Scheduler)`).
- Basic map-reduce over a small list (result equality with sequential).
- `ordered=True` returns results in submission order.
- Config-snapshot propagation: under `config.override(precision=11)`,
  workers read `config.numerics.precision == 11`.
- Short-circuit: deterministic prefix (submission-order collection), the
  callback fires, remaining futures cancelled.
- Empty input returns `reducer([])` without touching the cluster.
- No-client error: clear message; `distributed`-missing path covered by
  the importorskip structure.
- Progress: recorder-patched `LocalProgressBar` sees per-chunk updates and
  a final close (same recorder pattern as the thread-backend tests).
- One parallel≡sequential invariant: a small φ computation (e.g. the
  basic-substrate SIA) computed under `parallel_backend="dask"` equals the
  sequential result, formalism pinned with a complete preset. Placed in
  the fast lane only if LocalCluster startup keeps it within a few
  seconds; otherwise marked slow per the existing lane conventions.

## CHTC how-to (`docs/howto/chtc.md`)

Narrative documentation (not executable). Three deployment paths, each
labeled with its CHTC support status:

- **Path A — many independent runs (fully supported).** Plain condor jobs,
  optionally DAGMan; each job runs a PyPhi script inside a container.
  Includes a complete Apptainer definition that installs PyPhi from a
  locally built wheel (`uv build`; works for the unreleased 2.0 and
  switches to PyPI after release), a minimal submit file, and the
  file-transfer size guidance.
- **Path B — one big analysis on a fat node (fully supported).** A single
  high-memory / many-core condor job using the default process backend;
  submit-file resource requests and the 72 h runtime limit noted.
- **Path C — distributing one analysis across machines (pilot).**
  `dask_jobqueue.HTCondorCluster` on an access point: worker jobs run the
  PyPhi container, `cluster.scale(jobs=N)`, dashboard via SSH tunnel,
  single-threaded workers. States the port caveat plainly and lists the
  facilitation questions (which ports/nodes permit worker→manager
  traffic; policy on long-lived manager processes on access points;
  worker-job wall-time guidance for held pools). Notes Dask's resilience
  to preempted workers and its cost (lost work is recomputed).
- Sidebar: HTC vs HPC — why PyPhi work belongs on HTC.

A drafted CHTC facilitation email is delivered in-conversation (not
committed) when the guide lands.

## Bookkeeping

- Changelog fragment (`cluster-backend.feature.md`): Dask backend + extra +
  guide, no roadmap references.
- ROADMAP: P11 cluster-backends row → partial (Dask landed; HTCondor-native
  batch surface = the remaining half), with the follow-up's design sketch
  recorded on the row: sweep cells (with a substrate axis) materialized as
  independent condor jobs + collect into `SweepResult`; trigger = pool-scale
  campaigns where a held worker pool is wasteful.
- MCP `performance.md`: one paragraph noting the dask backend exists and
  that `parallel_backend` selects it.

## Out of scope

- HTCondor-native batch-submission surface (follow-up; see ROADMAP row).
- Multi-node distribution guarantees on CHTC specifically (pending the
  port-access answer; the backend is infrastructure-agnostic regardless).
- TaskVine integration (conda-only dependency; revisit only if the Dask
  path fails on CHTC).
- Published container images (recipe installs from a local wheel until 2.0
  is released).
