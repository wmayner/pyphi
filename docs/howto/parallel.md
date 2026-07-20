---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Run computations in parallel

PyPhi's core computations are combinatorially expensive: finding the minimum
information partition means evaluating many candidate partitions, purviews,
concepts, and relations. These evaluations are independent of one another, so
PyPhi can distribute them across CPU cores. Parallelism is off by default;
this page shows how to turn it on and how to tune it.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The global gate

`pyphi.config.parallel` is the master gate for parallelism. It is necessary
but not sufficient: when it is `False` (the default), everything runs
sequentially no matter what else is configured — and setting it to `True` by
itself parallelizes nothing, because each level of the computation also has
its own switch (see the per-level options below). The gate only ever forces
things *off*.

It is in the infrastructure layer of the configuration, so it can also be
read and written through its full path:

```{code-cell} python
pyphi.config.infrastructure.parallel
```

Writing to the flat name routes to the same underlying option:

```{code-cell} python
pyphi.config.parallel = True
pyphi.config.infrastructure.parallel
```

```{code-cell} python
pyphi.config.parallel = False  # back to the default
```

## Number of workers

`parallel_workers` sets how many CPU cores to use. The default of `-1` means
"all available cores".

```{code-cell} python
pyphi.config.infrastructure.parallel_workers
```

Set it to a positive integer to cap the worker count, for example to leave
cores free for other work:

```{code-cell} python
pyphi.config.parallel_workers = 4
pyphi.config.infrastructure.parallel_workers
```

```{code-cell} python
pyphi.config.parallel_workers = -1  # back to the default
```

## Enabling parallelism for a single computation

The best way to run one computation in parallel, without leaving the
global configuration changed, is the `override` context manager. Everything
inside the `with` block sees the overridden settings; outside, the previous
values are restored. Two things must be switched on: the global gate, and
the specific level (here, the system-partition search):

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
state = (0, 1, 1)

opts = dict(pyphi.config.infrastructure.parallel_partition_evaluation)
opts["parallel"] = True

with pyphi.config.override(
    parallel=True, parallel_partition_evaluation=opts, parallel_workers=2
):
    analysis = pyphi.analyze(substrate, state)

round(float(analysis.sia.phi), 6)
```

(This small example has fewer partitions than the level's
`sequential_threshold`, so it runs sequentially anyway — the threshold
exists because dispatching tiny workloads to workers costs more than it
saves. The configuration is what matters here; a larger system fans out
automatically.)

Parallelism changes only how the work is scheduled, never the result. The same
computation run sequentially gives the identical value:

```{code-cell} python
sequential = pyphi.analyze(substrate, state)
round(float(sequential.sia.phi), 6)
```

## Per-level options

Beyond the global gate, PyPhi parallelizes at several distinct levels of the
computation, each with its own option. Each is a dictionary with the keys
`parallel`, `sequential_threshold`, `chunksize`, and `progress`:

```{code-cell} python
dict(pyphi.config.infrastructure.parallel_distinction_evaluation)
```

The keys mean:

- **`parallel`** — whether to parallelize this level at all.
- **`sequential_threshold`** — workloads with fewer than this many items run
  sequentially, because for small counts the overhead of spawning workers and
  sending data to them costs more than it saves.
- **`chunksize`** — how many items each worker receives per batch. This
  governs granularity only, not the result.
- **`progress`** — whether to show a progress bar for this level.

The available levels are:

| Config option | Parallelizes over |
| --- | --- |
| `parallel_complex_evaluation` | Candidate systems (complexes) within a substrate |
| `parallel_distinction_evaluation` | Distinctions within a cause-effect structure |
| `parallel_partition_evaluation` | System partitions when searching for the minimum information partition |
| `parallel_purview_evaluation` | Candidate purviews for a mechanism |
| `parallel_mechanism_partition_evaluation` | Partitions of a single mechanism |
| `parallel_relation_evaluation` | Relations between distinctions |
| `parallel_macro_system_evaluation` | Macro (coarse-grained) systems |

To tune one level, assign a new dictionary. A common pattern is to read the
current value, change one key, and write it back:

```{code-cell} python
opts = dict(pyphi.config.infrastructure.parallel_distinction_evaluation)
opts["parallel"] = True
opts["sequential_threshold"] = 32
pyphi.config.parallel_distinction_evaluation = opts
dict(pyphi.config.infrastructure.parallel_distinction_evaluation)
```

```{code-cell} python
# Restore the default (parallel off, threshold 64)
pyphi.config.parallel_distinction_evaluation = {
    "parallel": False,
    "sequential_threshold": 64,
    "chunksize": 256,
    "progress": True,
}
```

## When parallelism helps

Parallelism is not free: starting worker processes and sending data to and
from them takes time, and that cost is fixed regardless of how much real work
is being done. It pays off only when the work per level is large enough to
dwarf that overhead. This is exactly what the `sequential_threshold` on each
level encodes.

As a rule of thumb:

- **Small networks** (a handful of nodes) usually run *faster* sequentially.
  The default `sequential_threshold` on each level already runs these
  workloads sequentially, so turning on `pyphi.config.parallel`
  costs little for small problems.
- **Larger networks** — where there are many mechanisms, purviews, or
  partitions to evaluate — benefit the most.
- Parallelism needs no extra installation: the process-pool backend is part
  of PyPhi's core dependencies.

When in doubt, benchmark both. Because the result is identical either way, you
can compare wall-clock time directly and pick whichever is faster for your
network size.

## Running on a cluster

The `dask` backend distributes the same parallel levels across a
`dask.distributed` cluster — a laptop `LocalCluster`, lab workstations, or
an HTCondor/Slurm pool via `dask-jobqueue`. Install the `cluster` extra
(`pip install "pyphi[cluster]"`), connect a `distributed.Client`, and set
`pyphi.config.parallel_backend = "dask"`. See {doc}`chtc` for cluster
deployment, including UW–Madison's CHTC.
