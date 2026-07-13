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

## The global switch

The single flag that enables parallelism is `pyphi.config.parallel`. It is in
the infrastructure layer of the configuration, so it can also be read and
written through its full path:

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
values are restored.

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
state = (0, 1, 1)

with pyphi.config.override(parallel=True, parallel_workers=2):
    analysis = pyphi.analyze(substrate, state)

round(float(analysis.sia.phi), 6)
```

Parallelism changes only how the work is scheduled, never the result. The same
computation run sequentially gives the identical value:

```{code-cell} python
sequential = pyphi.analyze(substrate, state)
round(float(sequential.sia.phi), 6)
```

## Per-level options

Beyond the global switch, PyPhi parallelizes at several distinct levels of the
computation, each with its own option. Each is a dictionary with the keys
`parallel`, `sequential_threshold`, `chunksize`, and `progress`:

```{code-cell} python
dict(pyphi.config.infrastructure.parallel_concept_evaluation)
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
| `parallel_concept_evaluation` | Concepts (distinctions) within a cause-effect structure |
| `parallel_partition_evaluation` | System partitions when searching for the minimum information partition |
| `parallel_purview_evaluation` | Candidate purviews for a mechanism |
| `parallel_mechanism_partition_evaluation` | Partitions of a single mechanism |
| `parallel_relation_evaluation` | Relations between distinctions |
| `parallel_macro_system_evaluation` | Macro (coarse-grained) systems |

To tune one level, assign a new dictionary. A common pattern is to read the
current value, change one key, and write it back:

```{code-cell} python
opts = dict(pyphi.config.infrastructure.parallel_concept_evaluation)
opts["parallel"] = True
opts["sequential_threshold"] = 32
pyphi.config.parallel_concept_evaluation = opts
dict(pyphi.config.infrastructure.parallel_concept_evaluation)
```

```{code-cell} python
# Restore the default (parallel off, threshold 64)
pyphi.config.parallel_concept_evaluation = {
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
- Parallelism requires the optional dependency to be installed
  (`uv pip install "pyphi[parallel]"`). Without it, PyPhi falls back to
  sequential evaluation.

When in doubt, benchmark both. Because the result is identical either way, you
can compare wall-clock time directly and pick whichever is faster for your
network size.
