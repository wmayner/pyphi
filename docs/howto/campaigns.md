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
  remaining.txt     task ids not yet done
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

The submit file queues one job per task id in `remaining.txt`. Each job runs
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
with the failed and pending ids, so resubmission is simply:

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

## Distribute one system's cause-effect structure

`prepare_ces` turns one system's scoped analysis into a campaign:

```python
status = campaign.prepare_ces(
    substrate,
    state=(1, 0, 0),
    scope=scope,
    directory="ces-campaign",
    units_per_job=1e6,
    seed=42,
)
```

The planner descends only as deep as the budget requires: whole mechanisms
are cost-balanced into jobs; a mechanism over budget splits its purview
list into ranges; a single (mechanism, purview) pair over budget splits its
partition sweep into interleaved strides. System-partition strides for the
SIA are planned the same way — unless you pass a precomputed `sia=`, or a
`resolution_state=` (in which case the collected structure carries no Φₛ
and congruence resolves against the given state, or the system's
intrinsic-information state by default).

On sparse substrates, pass `ordering="bottleneck_first"` so each stride
evaluates partitions that sever the fewest present connections first —
reducibility then short-circuits within the first evaluations. Ordering
never affects results.

Submission, monitoring, and resubmission work exactly as for sweep
campaigns. `collect()` merges the shards exactly — tie sets preserved,
identical to what a single machine would have produced over the same
scope — and assembles the `CauseEffectStructure` through the standard
analysis path:

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
