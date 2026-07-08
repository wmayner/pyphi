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

# Sweep parameter landscapes

`pyphi.sweep` runs one IIT computation across many states, candidate
subsystems, and formalisms in a single call, and collects every result into
one tidy long-format DataFrame. It saves you from writing the nested loops,
building each `System` by hand, and stitching the results back together.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## A first sweep

`sweep` takes a substrate and at least one axis to vary. The `states`
argument is required. Pass `"all"` to enumerate every state of the substrate:

```{code-cell} python
substrate = pyphi.examples.basic_substrate()

result = pyphi.sweep(substrate, states="all")
result.df.round(6)
```

Each row is one system-level integrated information analysis (an
{abbr}`SIA (System Irreducibility Analysis)`). The index holds the axis that
varied — here, the state — and the columns hold the extracted quantities:
`phi`, `normalized_phi`, and whether the system is irreducible. The axes that
did not vary appear as constant context columns (`formalism`, `subset`).

States that cannot be reached from any previous state have no defined
repertoire, so their $\Phi$ is undefined. When you enumerate an axis with
`"all"`, those cells are dropped rather than raised, and the dropped cells are
recorded on the result:

```{code-cell} python
result.skipped
```

## The result object

`sweep` returns a `SweepResult` with three fields:

- `df` — the tidy table (also available as `result.to_pandas()`).
- `results` — the raw result objects, aligned one-to-one with the rows of
  `df`, so you can reach into any cell for detail the table does not carry.
- `skipped` — the `(formalism, subset, state)` cells dropped as unreachable.

```{code-cell} python
type(result.results[0]).__name__
```

## Sweeping over subsystems

The `subsets` argument chooses which subsets of nodes to treat as the
candidate system. Pass `"all"` for the non-empty powerset, `"full"` (the
default) for the whole substrate, or an explicit list of node-index tuples.
Here we hold the state fixed and vary the subsystem:

```{code-cell} python
pyphi.sweep(substrate, states=(1, 0, 0), subsets="all").df.round(6)
```

## Sweeping over formalisms

Pass a list of version names to `formalisms` to compute the same cell under
each formalism. The active formalism is used when this argument is omitted.

```{code-cell} python
pyphi.sweep(
    substrate,
    states=(1, 0, 0),
    formalisms=["IIT_4_0_2023", "IIT_4_0_2026"],
).df.round(6)
```

When more than one axis varies at once, the index becomes a `MultiIndex` with
one level per varying axis.

## Choosing what to compute

By default each cell computes an SIA. Pass `compute="ces"` to compute the full
cause-effect structure instead. The extracted columns change to match: the
number of distinctions and the summed relation $\varphi$.

```{code-cell} python
pyphi.sweep(substrate, states="all", compute="ces").df.round(6)
```

`compute` also accepts any callable taking a `System`. The callable's return
value is stored in `result.results` for every cell; reach into that list to
work with whatever it returns.

## Running in parallel

Set `parallel=True` to spread the cells across worker processes. Each cell is a
whole SIA or CES computation, so this parallelizes at the level of the sweep
rather than inside any single computation. Passing `None` (the default) follows
`config.infrastructure.parallel`.

```{code-cell} python
pyphi.sweep(substrate, states="all", parallel=True).df.round(6)["phi"]
```

The results are returned in the same order whether or not the sweep runs in
parallel, so the table is identical either way.

## Reproducibility

Pass a `seed` to stamp it into every result's provenance record, so a saved
sweep carries the seed that produced it alongside the numbers:

```{code-cell} python
seeded = pyphi.sweep(substrate, states="all", seed=42)
seeded.df.shape
```
