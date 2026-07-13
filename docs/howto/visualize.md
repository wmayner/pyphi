---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Visualizing PyPhi results

PyPhi ships plotting helpers for the objects it computes: the cause-effect
structure and its relations, the connectivity and transition probabilities of a
substrate, cause and effect repertoires, a simulated trajectory, and the Ising
building blocks used to generate substrates. This guide walks through each on a
small example system.

The cause-effect-structure plots (`plot_ces`, `highlight_phi_fold`) use
[Plotly](https://plotly.com/python/) and are interactive — drag to rotate the
3-D hypergraph, hover for detail. The rest use Matplotlib. Rendering the
interactive figures in a page like this one needs the connected Plotly renderer
selected once:

```{code-cell} python
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio

pio.renderers.default = "notebook_connected"

import pyphi
from pyphi import examples, visualize as viz

pyphi.config.progress_bars = False
```

## Plotting a cause-effect structure

`plot_ces` renders the cause-effect structure of a system — its distinctions and
the relations among them. Compute one first:

```{code-cell} python
ces = examples.xor_system().ces()
```

`plot_ces` offers five views of the same structure, chosen with `view`. The
**lattice** view is a 2-D Hasse diagram of the distinctions ordered by inclusion;
marker size encodes each distinction's total relation φ (Σφ_r), color its own φ.

```{code-cell} python
viz.plot_ces(ces, view="lattice")
```

The **hypergraph** view places each distinction's cause and effect purviews as
vertices in 3-D and draws the relation faces among them:

```{code-cell} python
viz.plot_ces(ces, view="hypergraph")
```

The **scatter** view embeds distinctions on a deterministic projection of their
unit composition, sized by relation φ and colored by relational role:

```{code-cell} python
viz.plot_ces(ces, view="scatter")
```

The **matrix** view is a distinction-by-distinction heatmap of shared relation φ,
with each distinction's self-relation strength on the diagonal:

```{code-cell} python
viz.plot_ces(ces, view="matrix")
```

The **spectrum** view summarizes the high-degree structure as a bar panel of
relation count and Σφ_r per relation degree:

```{code-cell} python
viz.plot_ces(ces, view="spectrum")
```

## Visualizing analytically-computed relations

For larger structures the relation set is far too big to enumerate, so PyPhi can
compute relations *analytically* — every aggregate answered in closed form,
nothing materialized (see {doc}`query-relations`). Because that relation set
cannot be listed, `plot_ces` renders the **strongest** relations by φ_r, chosen
with `max_relations`:

```{code-cell} python
with pyphi.config.override(relation_computation="ANALYTICAL"):
    analytical_ces = examples.xor_system().ces()

viz.plot_ces(analytical_ces, view="lattice", max_relations=8)
```

Without `max_relations` the call raises, rather than silently drawing an
arbitrary subset of an unbounded set:

```{code-cell} python
try:
    viz.plot_ces(analytical_ces, view="lattice")
except ValueError as error:
    print(error)
```

The cap limits only how many relation *edges* are drawn. Marker size still
reflects each distinction's true total relation φ over the whole structure,
computed in closed form, so node sizes do not change as the cap tightens:

```{code-cell} python
from pyphi.visualize.projection import project_ces

tight = [n.sum_phi_relations for n in project_ces(analytical_ces, max_relations=1).nodes]
loose = [n.sum_phi_relations for n in project_ces(analytical_ces, max_relations=99).nodes]
print("identical node sizes:", tight == loose)
print(tight)
```

The spectrum view likewise shows the exact per-degree census regardless of the
cap, since it reads the closed-form degree spectrum rather than the drawn edges:

```{code-cell} python
viz.plot_ces(analytical_ces, view="spectrum", max_relations=1)
```

## Highlighting a Φ-fold

A Φ-fold is a seed distinction together with every relation incident to it.
`highlight_phi_fold` draws the whole structure dimmed with the fold in full
color, so a distinction's relational neighborhood stands out. Seed a fold with a
mechanism drawn from the structure:

```{code-cell} python
fold = ces.fold([(0, 1)])
viz.highlight_phi_fold(fold)
```

## Substrate connectivity

`plot_system` draws a substrate's connectivity graph, coloring each node by
whether it belongs to the subsystem and by its current state:

```{code-cell} python
system = examples.xor_system()
viz.plot_system(system)
plt.gcf()
```

`plot_graph` draws the bare directed graph of any connectivity matrix (as a
`networkx` graph), without the system coloring:

```{code-cell} python
import networkx as nx

graph = nx.from_numpy_array(system.cm, create_using=nx.DiGraph)
viz.plot_graph(graph)
plt.gcf()
```

`plot_tpm` shows the transition probability matrix as a heatmap. It expects a
state-by-state matrix (both dimensions a power of two), so convert from the
factored state-by-node form first:

```{code-cell} python
from pyphi import convert

state_by_node = examples.xor_substrate().tpm.to_pandas().values
state_by_state = convert.state_by_node2state_by_state(state_by_node)
fig, ax = viz.plot_tpm(state_by_state)
fig
```

## Repertoires

A repertoire is a probability distribution over states. `plot_distribution` draws
one (or several, overlaid) as a bar panel — here the system's cause repertoire
over its past states:

```{code-cell} python
repertoire = system.cause_repertoire(system.node_indices, system.node_indices)
fig, ax = viz.plot_distribution(repertoire)
fig
```

`plot_repertoires` compares the intact system's forward repertoire against its
minimum-information-partitioned counterpart, in both directions, from a
system-irreducibility analysis:

```{code-cell} python
from pyphi.formalism import queries

sia = queries.sia(system)
fig, axes, reps = viz.plot_repertoires(system, sia)
fig
```

## Trajectories

`plot_dynamics` draws a state raster: time on the x-axis, units on the y-axis,
cell brightness for each unit's state along a trajectory. Simulate one from a
multidimensional state-by-node TPM (with a seeded generator, so it reproduces):

```{code-cell} python
from pyphi import dynamics

tpm = convert.to_multidimensional(examples.xor_substrate().tpm.to_pandas().values)
trajectory = dynamics.simulate(
    tpm, initial_state=(1, 0, 0), timesteps=20, rng=np.random.default_rng(0)
)
fig, ax = viz.plot_dynamics(np.array(trajectory), node_labels=list(system.node_labels))
fig
```

Any array of shape `(timesteps, units)` works, so a recorded or synthetic
trajectory can be drawn the same way.

## Ising building blocks

The `ising` helpers visualize the activation function used by the Ising
substrate generator. `plot` overlays the activation sigmoid with each state's
input energy against its spin-on probability, for a given coupling matrix,
temperature, and field:

```{code-cell} python
weights = np.array([[0.0, 1.0], [1.0, 0.0]])
viz.ising.plot(weights, temperature=1.0, field=0.0)
```

`plot_sigmoid` draws the activation probability curve alone, over a range of
input energies:

```{code-cell} python
viz.ising.plot_sigmoid(np.linspace(-5, 5, 100), temperature=1.0, field=0.0)
plt.gcf()
```
