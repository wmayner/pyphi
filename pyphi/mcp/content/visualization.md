# Plotting a Φ-structure and other figures

PyPhi ships plotting helpers in `pyphi.visualize` (install the extra with
`pip install pyphi[visualize]`). The `plot` tool exposes the common ones. For
anything past its presets — a specific view, an analytically-computed structure,
or a structure you assembled yourself — drive `pyphi.visualize.plot_ces`
directly in Python. This is the reference for doing that without guessing.

## Getting a plottable cause-effect structure

`plot_ces` draws a **cause-effect structure** (Φ-structure): the distinctions a
system specifies and the relations among them. The normal way to get one:

```python
import pyphi
from pyphi import visualize as viz

analysis = pyphi.analyze(substrate, state)   # or: pyphi.analyze(substrate, state, compute="ces")
viz.plot_ces(analysis.ces, view="lattice")
```

`analysis.ces` is a `CauseEffectStructure`. `system.ces()` returns the same
thing. **This needs IIT 4.0** — under IIT 3.0, `ces()` returns bare distinctions
with no relations, which `plot_ces` cannot draw.

## The five views

`plot_ces(ces, view=...)` offers five views of one structure:

- `"lattice"` (default) — 2-D Hasse diagram of the distinctions ordered by
  inclusion; marker size is total relation φ, color is own φ.
- `"hypergraph"` — the interactive **3-D** view: cause/effect purviews as
  vertices, relation faces among them.
- `"scatter"` — distinctions on a deterministic embedding of their unit
  composition.
- `"matrix"` — distinction-by-distinction heatmap of shared relation φ.
- `"spectrum"` — bar panel of relation count and Σφ_r per relation degree.

`"barycentric"` is **not** a view — it is the default value of the separate
`layout=` argument (within-level ordering). Passing `view="barycentric"` fails.

Through the `plot` tool: `plot(result_ref, kind="ces", view="hypergraph")`.

## Concrete vs analytical relations, and `max_relations`

A common mistake is to believe `plot_ces` requires concrete (enumerable)
relations. It does not. The rule:

- The default `relation_computation` is `"ANALYTICAL"`: `ces.relations` is a
  closed-form summary that cannot be listed. `plot_ces(ces)` works with no
  extra arguments — it draws the strongest 1000 relations by φ_r (via
  `relations.strongest(k)`); pass `max_relations=N` to choose the cap.
- Under `relation_computation="CONCRETE"` the relation set is enumerable and
  `plot_ces(ces)` draws every relation unless `max_relations` caps it.

```python
ces = system.ces()
viz.plot_ces(ces, view="lattice")                    # renders the strongest 1000
viz.plot_ces(ces, view="lattice", max_relations=8)    # renders only the strongest 8
```

`max_relations` caps only how many relation *edges* are drawn. Marker sizes and
the spectrum view stay exact regardless of the cap — they read closed-form
totals, not the drawn edges. Through the tool:
`plot(result_ref, kind="ces", max_relations=8)`.

## Plotting a structure you built yourself

A structure you assembled by hand — for example distinctions restored from a
checkpoint — plots the same way. Construct a `CauseEffectStructure`; `sia` is not
read on the plot path, so `None` is fine:

```python
from pyphi.models import CauseEffectStructure

ces = CauseEffectStructure(None, distinctions, relations)
viz.plot_ces(ces, view="lattice")
```

The only requirement is that the structure be **relation-closed** — a `PhiFold`
(a single distinction's relational neighborhood) is not, and raises `TypeError`;
use `highlight_phi_fold` for that. The `plot` tool cannot reach a hand-built
object (it plots results from `analyze`), so this case is Python-only.

## Saving and viewing a figure

`plot_ces` and `highlight_phi_fold` return **Plotly** figures. Save one as a
self-contained HTML file — no network or CDN, opens in any browser — with inline
Plotly.js:

```python
fig = viz.plot_ces(analysis.ces, view="hypergraph")
fig.write_html("ces.html", include_plotlyjs="inline")
```

**Do not try to export these to PNG.** Static image export needs `kaleido`,
which PyPhi does not depend on (and current kaleido versions require a separate
Chrome download), and a still frame of a figure meant to be rotated and hovered
is misleading. Keep cause-effect-structure plots as HTML. In a notebook,
`fig.show()` displays them inline. The other helpers (`plot_system`,
`plot_graph`, `plot_tpm`, `plot_distribution`, `plot_repertoires`,
`plot_dynamics`, the Ising helpers) return Matplotlib figures and save the usual
way, `fig.savefig("figure.png")`.

## The full surface

This covers the cause-effect-structure plots. For the complete walkthrough of
every helper on a worked example — connectivity, repertoires, trajectories, the
Ising building blocks, and Φ-fold highlighting — see the "Visualize results"
how-to at https://pyphi.readthedocs.io, or read the docstrings directly
(`help(pyphi.visualize.plot_ces)`).
