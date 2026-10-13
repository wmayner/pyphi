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

# Sweep states and subsystems

`pyphi.sweep` runs one IIT computation across many states and candidate
subsystems in a single call, and collects every result into
one long-format DataFrame. It saves you from writing the nested loops,
building each `System` by hand, and stitching the results back together.
(To vary the substrate's *parameters* — its connection weights — rather than
its state or the candidate system, see
{doc}`Explore substrate parameter landscapes <landscape>`.)

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## A first sweep

`sweep` takes a substrate and at least one axis to vary. The `states`
argument is required. Pass `"all"` to enumerate every state of the substrate:

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()

result = pyphi.sweep(substrate, states="all")
result.df.round(6)
```

Each row is one system-level integrated information analysis (an
{abbr}`SIA (System Irreducibility Analysis)`). The first four columns say
which cell the row is: the `substrate` (its label), the `formalism` (the
version of IIT), the candidate system (`subset`) and the `state`. Every row
has all four, whether or not that axis varied, and the index is a plain
row number, so code written against one sweep works on any other. The
remaining columns hold the extracted quantities: φ_s (`phi`),
`normalized_phi`, whether the system is irreducible, and the selection
margins (`partition_margin`, `cause_state_margin`, `effect_state_margin`,
`effectively_tied` — see {doc}`Control tie-breaking <tie-breaking>`).

States that cannot be reached from any previous state have no defined
repertoire, so they cannot be analyzed. When you enumerate an axis with
`"all"`, those cells are dropped rather than raised, and the dropped cells
are recorded on the result. Our probabilistic substrate reaches every state,
so nothing was dropped here:

```{code-cell} python
result.skipped
```

With a deterministic substrate you can see this happen — two of the eight
states of the three-gate `basic` network are unreachable:

```{code-cell} python
pyphi.sweep(pyphi.examples.basic_substrate(), states="all").skipped
```

## The result object

`sweep` returns a `SweepResult` with three fields:

- `df` — the table (also available as `result.to_pandas()`).
- `results` — the raw result objects, aligned one-to-one with the rows of
  `df`, so you can reach into any cell for detail that is not in the table.
- `skipped` — the `(substrate, formalism, subset, state)` cells dropped as
  unreachable.

Displaying the result itself shows a summary: the number of rows, how many
cells were skipped, the size of each axis, and the leading columns of the
table.

```{code-cell} python
result
```

```{code-cell} python
type(result.results[0]).__name__
```

## Sweeping over subsets

The `subsets` argument chooses which subsets of nodes to treat as the
candidate system. Pass `"all"` for the non-empty powerset, `"full"` (the
default) for the whole substrate, one tuple of node indices or labels, or a
list of them. The `subset` column always holds sorted node indices, so a
subset spelled with labels and the same subset spelled with indices group
together. Here we hold the state fixed and vary the subset:

```{code-cell} python
pyphi.sweep(substrate, states=(0, 1, 1), subsets="all").df.round(6)
```

`states` takes the same kinds of value: `"all"`, one state, or a list of
states. NumPy integers, the rows of a 2-D array and iterators such as
`itertools.product(...)` work too, so states drawn with a random number
generator can be passed directly. An empty axis, an unknown string, or a
subset or state that does not fit a substrate raises a `ValueError` before
anything is computed.

The `formalisms=` argument adds one more axis, for computing the same cells
under earlier versions of IIT; see {doc}`earlier-versions`. However many axes
vary, they stay ordinary columns. To index the table by one of them, call
`set_index`:

```{code-cell} python
result.df.set_index("state")["phi"].round(6)
```

## Choosing what to compute

`compute` takes the same values as in {func}`pyphi.analyze`. Its default
here is `"sia"`, which computes only the SIA of each cell, because a sweep
multiplies the cost of one analysis by the number of cells and the SIA is the
cheapest result. Pass `compute="ces"` to compute the cause-effect structure
instead. The columns change to match, and are the ones
`Analysis.to_pandas()` returns: φ_s (`phi`), `normalized_phi`, Φ (`big_phi`),
the number of distinctions, and the summed $\varphi$ of the distinctions
(`sum_phi_d`) and of the relations (`sum_phi_r`).

```{code-cell} python
pyphi.sweep(substrate, states="all", compute="ces").df.round(6)
```

`compute=None` produces the same columns and stores an `Analysis` for each
cell in `result.results`. `compute="distinctions"` skips the system-partition
search and reports `n_distinctions`, `sum_phi_d` and a `congruence` column;
where `congruence` is `"unresolved"` the system's specified state is tied, so
the count and the sum are upper bounds (see
{meth}`System.distinctions <pyphi.system.System.distinctions>`).

`compute` also accepts any callable taking a `System`. The callable's return
value is stored in `result.results` for every cell; reach into that list to
work with whatever it returns. When the return value is a single number (or
another scalar), the table also holds it in a `value` column.

## Find near-tied cells

The margin columns make it a one-liner to find the cells whose selections
were effectively tied at the configured `precision` — the results whose
reported partitions or specified states are sensitive to tie-breaking rules
— none, for this asymmetric substrate:

```{code-cell} python
tied = result.df[result.df.effectively_tied.astype(bool)]
tied[["phi", "partition_margin", "cause_state_margin", "effect_state_margin"]].round(6)
```

Cells computed under IIT 3.0, which does not report margins, have `None` in
these columns.

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

A sweep draws no random numbers, so the same arguments always give the same
table. Each result carries the configuration it was computed under. To record
the seed of a larger experiment that a sweep belongs to, save the table with
{func}`pyphi.provenance.save_dataframe`, which embeds the seed and the run's
parameters in the file:

```{code-cell} python
import tempfile

from pyphi import provenance

path = provenance.save_dataframe(
    result.df, tempfile.mkdtemp(), "fig1a_states", seed=42
)
provenance.read_metadata(path)["provenance"]["seed"]
```
