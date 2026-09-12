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

# Estimate the cost before you run

Analyses grow faster than exponentially with the number of units. Count the
work first; it is far cheaper than discovering the answer by waiting.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## The practical ceilings

| What | Formalism | About |
| --- | --- | --- |
| system integrated information φ_s | IIT 4.0 | 10–12 units |
| distinctions and relations (the Φ-structure) | IIT 4.0 | 6–8 units |
| cause–effect structure, Φ | IIT 3.0 | 10–12 units |

These are for fully connected substrates on one core; sparser connectivity
raises them, since absent connections shrink every search.

## Count the work

{func}`pyphi.cost.estimate_analysis` walks the same enumerations the
analysis would, without computing any φ. It needs no state.

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig6d_substrate()
pyphi.cost.estimate_analysis(substrate)
```

The counting walk has its own budget, `limit` (one million steps by
default). On ten or more fully connected units it can take tens of seconds
and stop early, reporting `capped=True`; the counts are then lower bounds.
A small budget shows what that looks like on this six-unit substrate:

```{code-cell} python
estimate = pyphi.cost.estimate_analysis(substrate, limit=1_000)
estimate.capped, estimate.mechanism_partition_sweeps
```

Raise the budget for an exact count when you need one:

```{code-cell} python
estimate = pyphi.cost.estimate_analysis(substrate, limit=10_000_000)
estimate.capped, estimate.mechanism_partition_sweeps
```

`compute="sia"` counts only the system-partition search; `"distinctions"`
only the distinction axis; the default counts everything. The
system-partition count grows fastest with the number of units and is the
axis to watch for φ_s.

```{code-cell} python
pyphi.cost.estimate_analysis(substrate, compute="sia").system_partitions
```

## What to reduce

- **Connectivity.** Pass the real connectivity matrix if you know it.
- **The candidate system.** Analyze a subset (`subset=`) rather than the
  whole substrate; `substrate.complexes(state)` still has to consider every
  subset.
- **What you compute.** `compute="sia"` for φ_s alone.
- **Settings.** The cost-reduction table in
  {doc}`../theory/computational-complexity` lists the partition schemes and
  short-circuit options and what each gives up.
- **Cores and clusters.** {doc}`parallel` divides the constants;
  {doc}`campaigns` shards one analysis across machines.

## Where to go next

- {doc}`../theory/computational-complexity` derives the scaling.
- {doc}`grain-search` has its own pre-flight estimate.
