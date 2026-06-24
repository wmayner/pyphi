# Scatter and Matrix Renderers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the last two φ-structure views — relational-role scatter (Haun & Tononi Fig 8) and relation matrix — completing the four-view set; remove `_VIEWS_PENDING`.

**Architecture:** Both renderers are pure consumers of the existing projection (no projection changes). Scatter positions distinctions by deterministic PCA of purview-union composition; matrix aggregates `edges` into an n×n heatmap. Shared rescale/channel helpers move to `render/common.py`.

**Tech Stack:** Python 3.12+, plotly, numpy, pytest. All commands via `uv run`.

**Spec:** `docs/superpowers/specs/2026-06-04-visualize-scatter-matrix-design.md`

**Verified facts** (do not re-probe):
- xor projection: all four purview unions are `{0,1,2}` — identical, so all role flags are False ("none" role) and PCA is fully degenerate (the fallback path). All four distinctions participate in relations with others (connected).
- `DistinctionNode` fields: id, mechanism, label, cause_purview, effect_purview, mechanism_state, phi, sum_phi_relations, includes, included. `RelationEdge`: relata, degree, phi, overlap. Degree-1 self-relation edges exist.
- `NodeLabels(("A","B"), (0,1))`; iterating yields labels.
- Commit dance: hooks may reformat → re-`git add` + fresh commit; never `--no-verify`/amend; targeted `git add` only; `git -c commit.gpgsign=false commit`.

---

### Task 1: Theme "none" role color + shared render helpers

**Files:**
- Modify: `pyphi/visualize/theme.py`
- Create: `pyphi/visualize/render/common.py`
- Modify: `pyphi/visualize/render/lattice.py`, `pyphi/visualize/render/simplicial_complex.py` (use the shared helpers)
- Test: `test/test_visualize_projection.py`

- [ ] **Step 1: Write the failing test** — append to `test/test_visualize_projection.py`:

```python
def test_theme_role_colors_cover_none():
    from pyphi.visualize.theme import DEFAULT_THEME

    roles = dict(DEFAULT_THEME.role_colors)
    assert set(roles) == {"extended", "includes", "included", "none"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_projection.py::test_theme_role_colors_cover_none -q`
Expected: FAIL (missing "none")

- [ ] **Step 3: Implement**

In `pyphi/visualize/theme.py`, extend the default:

```python
    role_colors: tuple[tuple[str, str], ...] = (
        ("extended", "#e6b422"),
        ("includes", "#2f6fdb"),
        ("included", "#d85a46"),
        ("none", "#b0b0b0"),
    )
```

Create `pyphi/visualize/render/common.py`:

```python
"""Helpers shared by the render backends."""

from __future__ import annotations

CHANNEL_TITLES = {"phi": "φ", "sum_phi_relations": "Σφ_R"}


def rescale(values: list[float], lo: float, hi: float) -> list[float]:
    """Map values linearly onto [lo, hi]; midpoint if they are all equal."""
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [(lo + hi) / 2.0] * len(values)
    return [lo + (v - vmin) / (vmax - vmin) * (hi - lo) for v in values]
```

In `pyphi/visualize/render/simplicial_complex.py`: delete `_rescale`, add `from pyphi.visualize.render.common import rescale`, replace the one call site (`_purview_trace`).

In `pyphi/visualize/render/lattice.py`: replace `_CHANNELS` with `from pyphi.visualize.render.common import CHANNEL_TITLES` (update the two uses: validation via `in CHANNEL_TITLES`, colorbar title), and rewrite `_node_sizes`'s body to delegate:

```python
def _node_sizes(
    projection: PhiStructureProjection, theme: Theme, size_by: str | None
) -> list[float]:
    smin, smax = theme.node_size_range
    if size_by is None:
        return [(smin + smax) / 2.0] * len(projection.nodes)
    return rescale([getattr(n, size_by) for n in projection.nodes], smin, smax)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_projection.py test/test_visualize_lattice.py test/test_visualize_simplicial_complex.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/theme.py pyphi/visualize/render/common.py pyphi/visualize/render/lattice.py pyphi/visualize/render/simplicial_complex.py test/test_visualize_projection.py
git -c commit.gpgsign=false commit -m "Add none role color; share rescale/channel helpers across renderers"
```

---

### Task 2: Scatter renderer

**Files:**
- Create: `pyphi/visualize/render/scatter.py`
- Test: `test/test_visualize_scatter.py` (new)

- [ ] **Step 1: Write the failing tests** — create `test/test_visualize_scatter.py`:

```python
"""Tests for the relational-role scatter renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


def _make_projection(nodes, edges=()):
    from pyphi.labels import NodeLabels
    from pyphi.visualize.projection import InclusionOrder
    from pyphi.visualize.projection import PhiStructureProjection

    n = len(nodes)
    order = InclusionOrder(
        covers=((),) * n, rank=(0,) * n, size=(1,) * n
    )
    return PhiStructureProjection(
        nodes=tuple(nodes),
        edges=tuple(edges),
        mechanism_inclusion=order,
        purview_union_inclusion=order,
        node_labels=NodeLabels(("A", "B", "C", "D"), (0, 1, 2, 3)),
    )


def _node(i, label, purview, phi=1.0, sum_phi=0.0, includes=False, included=False):
    from pyphi.visualize.projection import DistinctionNode

    return DistinctionNode(
        id=i,
        mechanism=(i,),
        label=label,
        cause_purview=purview,
        effect_purview=purview,
        mechanism_state=(0,),
        phi=phi,
        sum_phi_relations=sum_phi,
        includes=includes,
        included=included,
    )


@pytest.fixture
def varied_projection():
    """Distinct singleton purview unions (non-degenerate PCA), varied roles,
    one node disconnected."""
    from pyphi.visualize.projection import RelationEdge

    nodes = [
        _node(0, "a", (0,), sum_phi=1.0, includes=True, included=True),
        _node(1, "b", (1,), sum_phi=2.0, includes=True),
        _node(2, "c", (2,), sum_phi=3.0, included=True),
        _node(3, "d", (3,), sum_phi=0.0),
    ]
    edges = (
        RelationEdge(relata=(0, 1), degree=2, phi=1.0, overlap=()),
        RelationEdge(relata=(0, 1, 2), degree=3, phi=0.5, overlap=()),
        # A self-relation does not make node 3 "connected".
        RelationEdge(relata=(3,), degree=1, phi=0.2, overlap=()),
    )
    return _make_projection(nodes, edges)


def _render(projection, **kwargs):
    from pyphi.visualize.render.scatter import render_scatter
    from pyphi.visualize.theme import DEFAULT_THEME

    return render_scatter(projection, DEFAULT_THEME, **kwargs)


def test_scatter_figure_structure(varied_projection):
    import plotly.graph_objects as go

    fig = _render(varied_projection)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert len(trace.x) == 4
    assert tuple(trace.text) == ("a", "b", "c", "d")
    # Largest marker belongs to the highest sum_phi_relations.
    sizes = list(trace.marker.size)
    assert sizes.index(max(sizes)) == 2
    # Role colors from the theme.
    from pyphi.visualize.theme import DEFAULT_THEME

    roles = dict(DEFAULT_THEME.role_colors)
    assert tuple(trace.marker.color) == (
        roles["extended"],
        roles["includes"],
        roles["included"],
        roles["none"],
    )
    # Connectedness symbols: node 3 only self-relates.
    assert tuple(trace.marker.symbol) == (
        "circle",
        "circle",
        "circle",
        "diamond-open",
    )


def test_scatter_positions_deterministic_and_distinct(varied_projection):
    a = _render(varied_projection).data[0]
    b = _render(varied_projection).data[0]
    assert tuple(a.x) == tuple(b.x) and tuple(a.y) == tuple(b.y)
    coords = set(zip(a.x, a.y, strict=True))
    assert len(coords) == 4


def test_scatter_degenerate_fallback(xor_projection):
    # All xor purview unions are identical: PCA variance is zero, the
    # fallback spreads points by node id.
    trace = _render(xor_projection).data[0]
    coords = set(zip(trace.x, trace.y, strict=True))
    assert len(coords) == 4
    # All roles are "none" and everything is connected.
    from pyphi.visualize.theme import DEFAULT_THEME

    roles = dict(DEFAULT_THEME.role_colors)
    assert set(trace.marker.color) == {roles["none"]}
    assert set(trace.marker.symbol) == {"circle"}


def test_scatter_numeric_color_channel(varied_projection):
    trace = _render(varied_projection, color_by="phi").data[0]
    assert tuple(trace.marker.color) == (1.0, 1.0, 1.0, 1.0)


def test_scatter_invalid_channels_raise(varied_projection):
    with pytest.raises(ValueError, match="size_by"):
        _render(varied_projection, size_by="bogus")
    with pytest.raises(ValueError, match="color_by"):
        _render(varied_projection, color_by="bogus")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_scatter.py -q`
Expected: FAIL with `ModuleNotFoundError: ... render.scatter`

- [ ] **Step 3: Implement** — create `pyphi/visualize/render/scatter.py`:

```python
"""Relational-role scatter renderer for phi-structure projections.

Positions come from a deterministic PCA of each distinction's purview-union
membership vector — a reproducible stand-in for the t-SNE composition
embedding of Haun & Tononi 2019 (Figs 7-8). Roles are derived from the
projection's purview-union inclusion flags (the closest computed structure
to the paper's relation-defined extendedness).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from pyphi.visualize.projection import PhiStructureProjection
from pyphi.visualize.render.common import CHANNEL_TITLES
from pyphi.visualize.render.common import rescale
from pyphi.visualize.theme import Theme


def _pca_coords(projection: PhiStructureProjection) -> list[tuple[float, float]]:
    """First two principal components of purview-union composition.

    Components are sign-fixed (largest-magnitude loading positive);
    zero-variance components fall back to spreading nodes evenly by id.
    """
    units = sorted(
        {u for n in projection.nodes for u in (*n.cause_purview, *n.effect_purview)}
    )
    column = {u: k for k, u in enumerate(units)}
    members = np.zeros((len(projection.nodes), len(units)))
    for n in projection.nodes:
        for u in set(n.cause_purview) | set(n.effect_purview):
            members[n.id, column[u]] = 1.0
    centered = members - members.mean(axis=0)
    u_mat, s, vt = np.linalg.svd(centered, full_matrices=False)
    n = len(projection.nodes)
    fallback = np.linspace(-0.5, 0.5, n) if n > 1 else np.zeros(1)
    coords = np.zeros((n, 2))
    for c in range(2):
        if c < len(s) and s[c] > 1e-9:
            component = u_mat[:, c] * s[c]
            if vt[c, np.argmax(np.abs(vt[c]))] < 0:
                component = -component
            coords[:, c] = component
        else:
            coords[:, c] = fallback
    return [(float(x), float(y)) for x, y in coords]


def _role(node) -> str:
    if node.includes and node.included:
        return "extended"
    if node.includes:
        return "includes"
    if node.included:
        return "included"
    return "none"


def _connected(projection: PhiStructureProjection) -> set[int]:
    """Ids of distinctions related to at least one other distinction."""
    connected: set[int] = set()
    for e in projection.edges:
        relata = set(e.relata)
        if len(relata) > 1:
            connected |= relata
    return connected


def render_scatter(
    projection: PhiStructureProjection,
    theme: Theme,
    fig: go.Figure | None = None,
    size_by: str | None = "sum_phi_relations",
    color_by: str = "role",
) -> go.Figure:
    """Scatter distinctions by composition, encoding relational roles.

    Marker size encodes ``size_by``; color encodes the relational-role
    category (``color_by="role"``) or a numeric channel; circles mark
    distinctions related to at least one other distinction, open diamonds
    those that only self-relate.
    """
    if size_by is not None and size_by not in CHANNEL_TITLES:
        raise ValueError(f"unknown size_by {size_by!r}")
    if color_by != "role" and color_by not in CHANNEL_TITLES:
        raise ValueError(f"unknown color_by {color_by!r}")
    nodes = projection.nodes
    coords = _pca_coords(projection)
    connected = _connected(projection)
    roles = [_role(n) for n in nodes]
    if color_by == "role":
        palette = dict(theme.role_colors)
        marker_color = {"color": [palette[r] for r in roles]}
    else:
        marker_color = {
            "color": [getattr(n, color_by) for n in nodes],
            "colorscale": theme.colorscale,
            "colorbar": {"title": CHANNEL_TITLES[color_by]},
        }
    smin, smax = theme.node_size_range
    sizes = (
        [(smin + smax) / 2.0] * len(nodes)
        if size_by is None
        else rescale([getattr(n, size_by) for n in nodes], smin, smax)
    )
    hover = [
        (
            f"<b>{n.label}</b> ({role})"
            f"<br>mechanism {n.mechanism} = {n.mechanism_state}"
            f"<br>cause {n.cause_purview} · effect {n.effect_purview}"
            f"<br>φ = {n.phi:.4g} · Σφ_R = {n.sum_phi_relations:.4g}"
        )
        for n, role in zip(nodes, roles, strict=True)
    ]
    trace = go.Scatter(
        x=[coords[n.id][0] for n in nodes],
        y=[coords[n.id][1] for n in nodes],
        mode="markers+text",
        text=[n.label for n in nodes],
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker={
            "size": sizes,
            "symbol": [
                "circle" if n.id in connected else "diamond-open" for n in nodes
            ],
            "line": {"width": 1, "color": "rgba(0,0,0,0.5)"},
            **marker_color,
        },
        showlegend=False,
    )
    figure = go.Figure() if fig is None else fig
    figure.add_trace(trace)
    figure.update_layout(
        plot_bgcolor=theme.background,
        font={"family": theme.font_family},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_scatter.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/render/scatter.py test/test_visualize_scatter.py
git -c commit.gpgsign=false commit -m "Add relational-role scatter renderer"
```

---

### Task 3: Matrix renderer

**Files:**
- Create: `pyphi/visualize/render/matrix.py`
- Test: `test/test_visualize_scatter.py` (shares the fixtures; rename below)

- [ ] **Step 1: Write the failing tests** — append to `test/test_visualize_scatter.py`:

```python
def test_matrix_exact_values(varied_projection):
    import plotly.graph_objects as go

    from pyphi.visualize.render.matrix import render_matrix
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_matrix(varied_projection, DEFAULT_THEME)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert isinstance(trace, go.Heatmap)
    # Order: all mechanisms size 1, so label order a, b, c, d.
    assert tuple(trace.x) == ("a", "b", "c", "d")
    z = [list(row) for row in trace.z]
    # Off-diagonal: (0,1) edge phi 1.0 plus (0,1,2) edge phi 0.5.
    assert z[0][1] == pytest.approx(1.5)
    assert z[1][0] == pytest.approx(1.5)
    assert z[0][2] == pytest.approx(0.5)
    assert z[1][2] == pytest.approx(0.5)
    # Diagonal: only node 3 has a self-relation.
    assert z[3][3] == pytest.approx(0.2)
    assert z[0][0] == z[1][1] == z[2][2] == 0.0
    # Node 3 shares no relations.
    assert z[0][3] == z[3][0] == 0.0


def test_matrix_xor_smoke(xor_projection):
    from pyphi.visualize.render.matrix import render_matrix
    from pyphi.visualize.theme import DEFAULT_THEME

    trace = render_matrix(xor_projection, DEFAULT_THEME).data[0]
    # Pairs (size 2) before abc (size 3).
    assert tuple(trace.x) == ("ab", "ac", "bc", "abc")
    assert len(trace.z) == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_scatter.py -q`
Expected: matrix tests FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `pyphi/visualize/render/matrix.py`:

```python
"""Relation-matrix renderer for phi-structure projections."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import plotly.graph_objects as go

from pyphi.visualize.projection import PhiStructureProjection
from pyphi.visualize.theme import Theme


def render_matrix(
    projection: PhiStructureProjection,
    theme: Theme,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Heatmap of relation strength between pairs of distinctions.

    An off-diagonal cell sums the phi of every relation involving both
    distinctions; a diagonal cell sums the distinction's self-relations
    (its reflexivity). Rows and columns are ordered by mechanism size,
    then label, so mechanism orders form contiguous blocks.
    """
    order = sorted(projection.nodes, key=lambda n: (len(n.mechanism), n.label))
    pos = {n.id: k for k, n in enumerate(order)}
    labels = [n.label for n in order]
    n = len(order)
    z = np.zeros((n, n))
    for e in projection.edges:
        relata = set(e.relata)
        if len(relata) == 1:
            (i,) = relata
            z[pos[i], pos[i]] += e.phi
        else:
            for a, b in combinations(sorted(relata), 2):
                z[pos[a], pos[b]] += e.phi
                z[pos[b], pos[a]] += e.phi
    hover = [
        [f"{labels[r]} × {labels[c]}<br>Σφ = {z[r, c]:.4g}" for c in range(n)]
        for r in range(n)
    ]
    trace = go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=theme.colorscale,
        colorbar={"title": "Σφ"},
        hovertext=hover,
        hoverinfo="text",
    )
    figure = go.Figure() if fig is None else fig
    figure.add_trace(trace)
    figure.update_layout(
        plot_bgcolor=theme.background,
        font={"family": theme.font_family},
        yaxis={"autorange": "reversed"},
    )
    return figure
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_scatter.py -q`
Expected: PASS

- [ ] **Step 5: Rename the test module** to reflect its scope:

```bash
git mv test/test_visualize_scatter.py test/test_visualize_scatter_matrix.py 2>/dev/null || mv test/test_visualize_scatter.py test/test_visualize_scatter_matrix.py
uv run pytest test/test_visualize_scatter_matrix.py -q
```

(`git mv` fails if the file was never committed; the plain `mv` fallback covers that.)

- [ ] **Step 6: Commit**

```bash
git add pyphi/visualize/render/matrix.py test/test_visualize_scatter_matrix.py
git -c commit.gpgsign=false commit -m "Add relation-matrix renderer"
```

---

### Task 4: Public API — all four views

**Files:**
- Modify: `pyphi/visualize/__init__.py`
- Test: `test/test_visualize_scatter_matrix.py`, `test/test_visualize_lattice.py`

- [ ] **Step 1: Write the failing tests** — append to `test/test_visualize_scatter_matrix.py`:

```python
def test_plot_phi_structure_scatter_and_matrix_views():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    fig = plot_phi_structure(ces, view="scatter")
    assert isinstance(fig.data[0], go.Scatter)
    fig = plot_phi_structure(ces, view="scatter", color_by="phi")
    assert tuple(fig.data[0].marker.color) != ()
    fig = plot_phi_structure(ces, view="matrix")
    assert isinstance(fig.data[0], go.Heatmap)
```

In `test/test_visualize_lattice.py`, replace `test_plot_phi_structure_unimplemented_views_raise`:

```python
def test_plot_phi_structure_unknown_view_raises():
    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    with pytest.raises(ValueError, match="view"):
        plot_phi_structure(ces, view="bogus")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_scatter_matrix.py test/test_visualize_lattice.py -q`
Expected: new tests FAIL (`NotImplementedError` for scatter/matrix)

- [ ] **Step 3: Implement** — in `pyphi/visualize/__init__.py`:

Delete `_VIEWS_PENDING` and its `NotImplementedError` branch. Change the signature default `color_by="phi"` → `color_by=None` and document: "`None` (the default) uses the view's default — `"phi"` for the lattice, `"role"` for the scatter; scatter additionally accepts `"role"`." Update the `view` docstring to list all four values (add: ``"scatter"``: distinctions on a PCA composition embedding, sized by total relation phi and colored by relational role; ``"matrix"``: a distinctions-by-distinctions heatmap of shared relation phi).

Dispatch tail:

```python
    projection = project_phi_structure(ces_, node_labels=node_labels)
    if view == "lattice":
        from .render.lattice import render_lattice

        return render_lattice(
            projection,
            theme,
            fig=fig,
            layout=layout,
            order=order,
            rank=rank,
            size_by=size_by,
            color_by="phi" if color_by is None else color_by,
        )
    if view == "simplicial_complex":
        from .render.simplicial_complex import render_simplicial_complex

        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if show is not None:
            kwargs["show"] = show
        return render_simplicial_complex(
            projection, theme, fig=fig, layout=layout, **kwargs
        )
    if view == "scatter":
        from .render.scatter import render_scatter

        return render_scatter(
            projection,
            theme,
            fig=fig,
            size_by=size_by,
            color_by="role" if color_by is None else color_by,
        )
    if view == "matrix":
        from .render.matrix import render_matrix

        return render_matrix(projection, theme, fig=fig)
    raise ValueError(f"unknown view {view!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_scatter_matrix.py test/test_visualize_lattice.py test/test_visualize_simplicial_complex.py test/test_visualize_projection.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/__init__.py test/test_visualize_scatter_matrix.py test/test_visualize_lattice.py
git -c commit.gpgsign=false commit -m "Dispatch scatter and matrix views; all four views ship"
```

---

### Task 5: Changelog, full verification, visual check

**Files:**
- Modify: `changelog.d/visualize-projection-lattice.feature.md`

- [ ] **Step 1: Update the changelog fragment** — extend the simplicial-complex sentence so the fragment ends:

```
..., `highlight_phi_fold` is reimplemented over it, and the legacy
`pyphi.visualize.ces` module is removed. Two further views complete the
set: `view="scatter"` (distinctions on a deterministic PCA composition
embedding, sized by total relation phi, colored by relational role, with
connectedness symbols) and `view="matrix"` (a distinctions-by-distinctions
heatmap of shared relation phi, with self-relation strength on the
diagonal).
```

- [ ] **Step 2: Full verification**

Run: `uv run pytest -q` (no path — doctest-inclusive) — expected all PASS.
Run: `uv run ruff check pyphi test` — expected clean.

- [ ] **Step 3: Render all four views** of xor + rule154 to the visual companion screen dir for maintainer reaction (not a test).

- [ ] **Step 4: Commit**

```bash
git add changelog.d/visualize-projection-lattice.feature.md
git -c commit.gpgsign=false commit -m "Update visualize changelog fragment for scatter and matrix views"
```
