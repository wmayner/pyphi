# Run a sweep as an HTCondor campaign

A **campaign** turns a sweep into a directory of self-contained batch jobs
for an HTCondor pool. Every task is a serialized file, every result is a
file, and no live connection between machines is ever needed: a task is done
exactly when its output file exists and loads, so status, collection, and
resubmission are computed purely from the campaign directory. The lifecycle
is prepare → submit → monitor → collect:

1. `pyphi.campaign.prepare(...)` writes the campaign directory on your
   machine.
2. You copy it to the cluster access point and run `condor_submit pyphi.sub`.
3. `pyphi.campaign.status(...)` reports progress and refreshes the
   resubmission list; failed or missing tasks are resubmitted by running
   `condor_submit pyphi.sub` again.
4. `pyphi.campaign.collect(...)` reassembles the outputs into the exact
   {class}`~pyphi.sweep.SweepResult` a local {func}`~pyphi.sweep.sweep`
   would have returned.

This guide covers the campaign workflow; cluster-specific setup — accounts,
choosing the HTC system, and building the `pyphi.sif` container image — is
in {doc}`chtc`.

## Prepare the campaign

`prepare` takes the same axes as {func}`~pyphi.sweep.sweep` — substrates,
states, subsets, formalisms, and the computation — plus packing controls,
and writes a self-contained directory:

```python
import pyphi
from pyphi import campaign, examples

status = campaign.prepare(
    {
        "grid": examples.fig4_substrate(),
        "basic": examples.basic_substrate(),
    },
    states="all",
    formalisms=["IIT_4_0_2026"],
    compute="sia",
    directory="my-campaign",
    units_per_job=1e6,
    seed=42,
)
print(status)
```

The returned {class}`~pyphi.campaign.CampaignStatus` summarizes the plan:
how many cells, how many tasks, and the total estimated work. The directory
contains everything a job needs:

```
my-campaign/
  manifest.json     axes, per-cell estimates, packing, seed
  substrates/       each substrate serialized once
  tasks/            one task file per condor job
  outputs/          filled in by the jobs
  logs/             condor stdout/err/log
  remaining.txt     task id, memory rows not yet done
  run_task.sh       the wrapper each job executes
  pyphi.sub         the generated submit file
```

`prepare` never overwrites an existing directory; each campaign gets a fresh
one.

## Submit

Copy the campaign directory to the cluster access point (for example with
`scp -r`), make sure the `pyphi.sif` container image is present (see the
container section of {doc}`chtc`; the image path can be customized with
`prepare(..., container_image=...)`), and submit from inside the directory:

```console
$ condor_submit pyphi.sub
```

The submit file queues one job per `task id, memory` row in `remaining.txt`,
requesting each task's own memory. Each job runs
`run_task.sh` inside the container, which executes the task file and writes
its output document; condor transfers the output back into `outputs/`.

## Monitor and resubmit

From any machine that has the campaign directory (the access point, or your
laptop after copying `outputs/` back):

```python
status = campaign.status("my-campaign")
print(status)
```

A task is **done** when its output file exists, loads, and every cell in it
succeeded or was skipped; **failed** when the output records an error;
**pending** when there is no output yet. `status` rewrites `remaining.txt`
with the failed and pending rows (memory column included), so resubmission is
simply:

```console
$ condor_submit pyphi.sub
```

A re-run task preserves its previous output under an `attempt-N` name, so no
attempt's record is lost.

## Collect

```python
result = campaign.collect("my-campaign", partial=False)
result.df
```

`collect` restores cells to their preparation order and rebuilds the
identical `SweepResult` a local sweep over the same axes would produce —
same DataFrame, same raw result objects, same skip list. With missing or
failed tasks it raises with a per-task summary; pass `partial=True` to get
the completed subset (with a warning). The `seed` given at prepare time is
stamped into each result's provenance.

## Running tasks without condor

The runner is plain Python, so tasks can be executed anywhere — useful for
testing a campaign locally before shipping it, or for finishing a few
stragglers on your own machine:

```console
$ python -m pyphi.campaign run my-campaign/tasks/task-0000.json.gz \
    --substrates my-campaign/substrates --outputs my-campaign/outputs
```

The exit code is nonzero when any cell errored; the output document is
written either way, with a per-cell outcome (`ok`, `skipped`, or `error`
with the full traceback).

## Packing and admission control

Each cell is weighted by {func}`~pyphi.cost.estimate_analysis` counts under
its formalism preset. Work units are enumeration counts — partition sweeps,
purview evaluations — not seconds; use them to balance tasks and compare
workloads, not to predict wall time.

- Default: one cell per condor job (the canonical high-throughput shape).
- `jobs=K`: pack the cells into exactly K cost-balanced tasks.
- `units_per_job=X`: target X work units per task; the task count is
  `ceil(total / X)`.

If any single cell's estimate exceeds `infeasible_threshold` (default 10⁹),
`prepare` warns loudly naming the cell — such a cell likely cannot finish
within CHTC's 72-hour slot and needs narrower axes or a dedicated fat-node
run ({doc}`chtc`). Pass `strict=True` to make the warning an error. Every
per-cell estimate and packing decision is recorded in `manifest.json`.

## Declare the feasible surface (scope)

For large systems the full distinction computation is combinatorially out of
reach; a **scope** declares which part of it you compute. A scope changes
*what* is computed — with the exclusions recorded and certified — never a
silent approximation: within the scope, every value is exact.

```python
from pyphi.campaign.scope import AxisScope, CESScope

scope = CESScope(
    # Mechanisms up to order 3 that involve unit A:
    mechanisms=AxisScope(containing=("A",), max_order=3),
    # Purviews of any order, but only within these units:
    cause_purviews=AxisScope(within=("A", "B", "C")),
)
```

Each axis (`mechanisms`, `cause_purviews`, `effect_purviews`) accepts an
explicit list of unit sets (`explicit=...`, exclusive of the other fields)
or any combination of `min_order`, `max_order`, `containing` (must include
all these units), and `within` (must be a subset), combined by
intersection. Units may be labels or indices. Purview constraints only
narrow the connectivity-pruned candidates; partition sweeps cannot be
scoped — a partial sweep would silently change φ.

`pyphi.estimate_analysis(substrate, compute="ces", scope=scope)` prices the
scoped workload before you commit to it.

## Distribute scoped cause-effect structures

`prepare_ces` turns a scoped analysis into a campaign. It takes the same
axes as `prepare` — substrates, states, subsets, formalisms, with scalars
accepted anywhere — all sharing one scope:

```python
status = campaign.prepare_ces(
    substrate,
    states=(1, 0, 0),
    scope=scope,
    directory="ces-campaign",
    units_per_job=1e6,
    seed=42,
)
```

A sweep over many states (or substrates, or formalisms) under the same
scope is one campaign directory rather than many:

```python
status = campaign.prepare_ces(
    substrate,
    states=[(1, 0, 0), (1, 1, 0), (1, 0, 1)],
    scope=scope,
    directory="ces-sweep-campaign",
    units_per_job=1e6,
    seed=42,
)
```

The shard plan depends only on the substrate, subset, and formalism — not
the state — so cells differing only by state share one planning pass and
replicate its shard tasks. For large scoped systems whose planning walk
exceeds the default work budget, raise `limit=`.

The planner descends only as deep as the budget requires: whole mechanisms
are cost-balanced into jobs; a mechanism over budget splits its purview
list into ranges; a single (mechanism, purview) pair over budget splits its
partition sweep into interleaved strides. System-partition strides for the
SIA are planned the same way — unless you pass a precomputed `sia=`
(single-cell campaigns only) or a `resolution_state=`, in which case no
SIA shards are planned, the collected structures carry no Φₛ, and each
cell's congruence resolves against its own given state. A single-cell
campaign takes one specification (the result of
{func}`~pyphi.formalism.iit4.system_intrinsic_information`); a multi-cell
campaign takes a mapping keyed by the full
`(label, formalism, subset, state)` cell tuples, a mapping keyed by state
alone when the other axes are singletons, or a callable
`cell -> specification`. A sweep over many states of one substrate then
plans once and resolves each state's structure against its own specified
state:

```python
resolution = {
    state: system_intrinsic_information(
        pyphi.System(substrate, state), specification_measure=measure
    )
    for state in states
}
campaign.prepare_ces(
    substrate,
    states=states,
    scope=scope,
    directory="ces-campaign",
    units_per_job=2000.0,
    resolution_state=resolution,
)
```

Values are validated at preparation time; without a `resolution_state`,
collect falls back to computing the intrinsic-information state itself,
which is infeasible for large systems.

Every shard requests memory sized to the largest purview repertoire it
holds, and packing groups purviews by memory class, so small work never
occupies a big-memory slot. `request_memory=` is the floor under those
estimates (default `"4GB"`); a large floor effectively opts out of
stratification.

When purview size should track mechanism size, give the scope an explicit
order table instead of one permissive fixed cap:

```python
scope = CESScope(
    mechanisms=AxisScope(max_order=5),
    max_purview_order_by_mechanism_order=(
        (1, 3), (2, 5), (3, 7), (4, 9), (5, 11),
    ),
)
```

Mechanism orders absent from the table fall back to the static purview
axes alone.

On sparse substrates, pass `ordering="bottleneck_first"` so each stride
evaluates partitions that sever the fewest present connections first —
reducibility then short-circuits within the first evaluations. Ordering
never affects results.

Submission, monitoring, and resubmission work exactly as for sweep
campaigns. `collect()` merges the shards exactly — tie sets preserved,
identical to what a single machine would have produced over the same
scope — and assembles each cell's `CauseEffectStructure` through the
standard analysis path. A single-cell campaign returns the structure; a
multi-cell campaign returns the same `SweepResult` a whole-cell
`compute="ces"` sweep produces, and `scope_report` returns one report per
cell keyed by `(label, formalism, subset, state)`:

```python
ces = campaign.collect("ces-campaign")
report = campaign.scope_report("ces-campaign")
print(report)
```

The **scope report** records what was computed and what the scope
excluded, with certificates: the computed Σφ_r is an exact lower bound for
the full structure (partial structures are exact substructures), and the
measured upper bounds on Σφ_r and Φ come from the certified bound
machinery. Missing shard groups (from failed or pending tasks collected
with `partial=True`) are listed separately from scope exclusions.

## When one fat node beats a sharded campaign

For a sparse, scoped, mid-size system (tens of units, mechanism order
capped), the alternative to sharding is one job per state that runs the
whole scoped analysis with native parallelism: `request_cpus = 32`,
`request_memory` sized to the analysis, and `pyphi.config.parallel`
enabled. Prefer the fat-node pattern when:

- the shard count at your budget is large (thousands) while a single
  state's whole analysis fits comfortably in one slot's memory and a
  72-hour window — per-shard scheduling overhead then dominates; or
- most shards' memory requests approach the whole-analysis footprint
  anyway (peak memory is set by the largest purview repertoire, which
  sharding cannot reduce).

Prefer sharding when a single state cannot finish in one slot, when
big-memory slots are scarce (stratified shard requests keep small work in
small slots), or when you need per-shard retry granularity on a busy
pool. Per-shard memory requests are estimated automatically, so holds
from underestimated memory are no longer the deciding factor.
