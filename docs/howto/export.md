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

# Export results

PyPhi's result objects and substrates have their own analysis methods, but
you will often want to hand the numbers to another tool: a plotting library, a
statistics package, or a probabilistic-graphical-model toolkit. This guide
shows the three export paths PyPhi provides: a Pandas `DataFrame` view of a
cause-effect structure, a labeled xarray view of a substrate's transition
probabilities, and a dynamic Bayesian network view of a substrate's dynamics.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

We use the IIT 4.0 paper's three-unit Fig 1A substrate throughout.

```{code-cell} python
from pyphi import examples

substrate = examples.iit4_2023_fig1a_substrate()
state = (0, 1, 1)
```

## Cause-effect structures as a DataFrame

Call `.to_pandas()` on a cause-effect structure to get one row per distinction,
indexed by mechanism, with the mechanism state, cause and effect purviews, and
$\varphi$.

```{code-cell} python
analysis = pyphi.analyze(substrate, state)
ces = analysis.ces

df = ces.to_pandas()
df
```

The result is an ordinary `DataFrame`, so the full Pandas API is available for
filtering, aggregation, and export to CSV, Excel, or Parquet.

```{code-cell} python
df["phi"].sum()
```

The `Analysis` object itself also has a `.to_pandas()` returning a single-row
summary of the system-level quantities.

```{code-cell} python
analysis.to_pandas()
```

## Transition probabilities as an xarray Dataset

A substrate's transition probability matrix exports to a labeled
`xarray.Dataset` via `substrate.tpm.to_xarray()`. Each unit becomes a data
variable holding its conditional distribution over the next state, with named
dimensions and integer coordinates drawn from the state space. This requires
the optional `xarray` dependency (`pip install pyphi[xarray]`).

```{code-cell} python
ds = substrate.tpm.to_xarray()
ds
```

Named dimensions make it straightforward to select a conditional slice by state
rather than by raw axis position. Here we read unit 0's next-state distribution
given that its inputs are all off.

```{code-cell} python
ds["unit_0"].sel(u0=0, u1=0, u2=0).values
```

## Dynamics as a dynamic Bayesian network

A substrate's transition dynamics can be exported as a two-timeslice dynamic
Bayesian network. Two forms are available.

`substrate.to_dbn_dict()` returns a plain dictionary with no third-party
dependency. It has three keys: `variables` maps each node label to its alphabet
size, `edges` lists the inter-slice parent-to-child pairs (implicitly from time
$t$ to $t+1$), and `cpds` maps each node to its parents and its conditional
probability table.

```{code-cell} python
dbn = substrate.to_dbn_dict()

print("variables:", dbn["variables"])
print("edges:    ", dbn["edges"])
print("A parents:", dbn["cpds"]["A"]["parents"])
```

`substrate.to_dbn()` returns the same structure as a `networkx.DiGraph`, ready
for graph tooling. Each node label `X` becomes two variables, `(X, 0)` at time
$t$ and `(X, 1)` at time $t+1$, connected only by inter-slice edges. The graph
is therefore acyclic even when the substrate has feedback. This requires the
`visualize` extra (`pip install pyphi[visualize]`), which provides networkx.

```{code-cell} python
g = substrate.to_dbn()

print("nodes:", sorted(g.nodes, key=str))
print("edges:", sorted(g.edges, key=str))
```

Each next-timeslice node stores its conditional probability table and its
ordered parent labels as node attributes.

```{code-cell} python
g.nodes[("A", 1)]["parents"]
```
