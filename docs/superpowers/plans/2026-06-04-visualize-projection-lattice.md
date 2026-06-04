# Visualization Projection Foundation + Lattice Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure `projection/` layer (model → plot-data, the single model-coupling point, exact-value unit-tested), a `Theme` dataclass, the `view=`-dispatching public entry point, and the first renderer — the inclusion-lattice (Hasse) view of a φ-structure.

**Architecture:** `project_phi_structure(ces)` extracts plain dataclasses (`DistinctionNode`/`RelationEdge`/`InclusionOrder`) from a `CauseEffectStructure`; `render/lattice.py` turns them into a plotly 2-D Hasse figure; `pyphi.visualize.plot_phi_structure(ces, view="lattice")` composes the two. Pure helpers (`_sum_phi_relations`, `_inclusion_order`) carry exact-value tests; the builder gets an integration test on `xor_system().ces()` (computes in ~0.2 s with 4 distinctions / 15 relations).

**Tech Stack:** Python 3.12+, dataclasses, numpy (projection); plotly (render). Run everything with `uv run`. Spec: `docs/superpowers/specs/2026-06-04-visualize-projection-lattice-design.md`.

---

## Background the engineer needs (verified against the live API)

- `ces = pyphi.examples.xor_system().ces()` → `CauseEffectStructure` with
  `ces.distinctions` (indexable) and `ces.relations` (iterable, `len()` works).
  For xor: 4 distinctions with mechanisms `[(0,1), (0,2), (1,2), (0,1,2)]`,
  15 relations. Computes in ~0.2 s.
- `Distinction` exposes: `.mechanism` (tuple of ints), `.mechanism_label` (str,
  e.g. `'ab'`), `.cause_purview` / `.effect_purview` (tuples), `.purview_union`
  (set-like of units), `.mechanism_state` (tuple), `.phi` (float-like),
  `.node_labels`.
- `Relation` is a frozenset of relata (full `Distinction`-like objects —
  do **not** print them, the reprs are huge). `len(relation)` = degree.
  `relation.phi` (float-like). `relation.mechanisms` → **set of mechanism
  tuples** of the involved distinctions (e.g. `{(0,1,2), (1,2), (0,1)}`) — use
  this to map a relation to projection node ids (mechanism uniquely keys a
  distinction within a CES). `relation.purview` → iterable of unit objects,
  each with `.index`.
- `.purview_union` elements are unit objects; build a `frozenset[int]` via
  `frozenset(u.index for u in d.purview_union)`. (If `purview_union` already
  yields ints in some path, `getattr(u, "index", u)` handles both.)
- **Deviation from spec noted:** the spec said inclusion is "derived from
  `Distinctions.purview_inclusion(...)`"; that helper returns a unit-keyed
  mapping (`frozenset_of_units → distinctions`), the wrong shape for a
  distinction-to-distinction partial order. Instead the projection computes the
  order directly by subset comparison of the distinctions' `purview_union`
  index-sets (4 lines, O(n²) — fine at plotting scale). Same spirit (no
  re-implementation of relation machinery); record this in the spec when done.
- The legacy 3-D plot stays at `pyphi.visualize.ces.plot_phi_structure`
  untouched; the new top-level `plot_phi_structure` replaces the old re-export
  in `pyphi/visualize/__init__.py` (breaking pre-2.0 is fine; the
  `NotImplementedError` for `view="evocative"` points at the legacy path).
- All plotting deps are installed in the dev env. Tests must not require a
  display (plotly figures are pure objects).

---

## Task 1: Projection dataclasses + pure helpers (exact-value TDD)

**Files:**
- Create: `pyphi/visualize/projection/__init__.py`
- Create: `test/test_visualize_projection.py`

- [ ] **Step 1: Write the failing exact-value tests for the pure helpers**

Create `test/test_visualize_projection.py`:

```python
"""Tests for the pure visualization projection layer."""

import pytest

from pyphi.visualize.projection import (
    InclusionOrder,
    RelationEdge,
    _inclusion_order,
    _sum_phi_relations,
)


def test_sum_phi_relations_exact():
    edges = (
        RelationEdge(relata=(0, 1), degree=2, phi=0.5, overlap=(2,)),
        RelationEdge(relata=(0, 1, 2), degree=3, phi=1.0, overlap=(0, 1)),
    )
    assert _sum_phi_relations(4, edges) == (1.5, 1.5, 1.0, 0.0)


def test_inclusion_order_exact():
    # purview-unit sets: two points, a pair, the whole.
    unions = (
        frozenset({0}),
        frozenset({1}),
        frozenset({0, 1}),
        frozenset({0, 1, 2}),
    )
    order = _inclusion_order(unions)
    assert isinstance(order, InclusionOrder)
    # covers = transitive reduction: the whole covers only the pair;
    # the pair covers both points; points cover nothing.
    assert order.covers == ((), (), (0, 1), (2,))
    # rank = longest down-chain: points 0, pair 1, whole 2.
    assert order.rank == (0, 0, 1, 2)


def test_inclusion_order_equal_unions_no_edge():
    # Mutually-equal purview unions: same rank, no cover edge between them.
    unions = (frozenset({0, 1}), frozenset({0, 1}), frozenset({0}))
    order = _inclusion_order(unions)
    assert order.covers == ((2,), (2,), ())
    assert order.rank == (1, 1, 0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest test/test_visualize_projection.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.visualize.projection'`.

- [ ] **Step 3: Implement the dataclasses + helpers**

Create `pyphi/visualize/projection/__init__.py`:

```python
"""Pure projection of IIT result objects into plot-ready data.

This package is the only part of :mod:`pyphi.visualize` that touches
result-object internals (:class:`Distinction`, :class:`Relation`). It imports
no plotting libraries; renderers consume the dataclasses defined here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pyphi.labels import NodeLabels

__all__ = [
    "DistinctionNode",
    "InclusionOrder",
    "PhiStructureProjection",
    "RelationEdge",
    "project_phi_structure",
]


@dataclass(frozen=True)
class DistinctionNode:
    """Plot-ready data for one distinction."""

    id: int
    mechanism: tuple[int, ...]
    label: str
    cause_purview: tuple[int, ...]
    effect_purview: tuple[int, ...]
    mechanism_state: tuple[int, ...]
    phi: float
    sum_phi_relations: float
    includes: bool
    included: bool


@dataclass(frozen=True)
class RelationEdge:
    """Plot-ready data for one relation."""

    relata: tuple[int, ...]
    degree: int
    phi: float
    overlap: tuple[int, ...]


@dataclass(frozen=True)
class InclusionOrder:
    """The purview-inclusion partial order over distinctions.

    ``covers[i]`` lists the node ids that node ``i`` directly down-includes
    (the transitive reduction); ``rank[i]`` is the length of the longest
    down-chain below ``i`` (single-unit "points" have rank 0, the "whole"
    distinction the maximum), so it is monotonic in the partial order and
    suitable as a vertical layout coordinate.
    """

    covers: tuple[tuple[int, ...], ...]
    rank: tuple[int, ...]


@dataclass(frozen=True)
class PhiStructureProjection:
    """Everything a renderer needs to draw a phi-structure."""

    nodes: tuple[DistinctionNode, ...]
    edges: tuple[RelationEdge, ...]
    inclusion: InclusionOrder
    node_labels: NodeLabels


def _sum_phi_relations(
    n_nodes: int, edges: Sequence[RelationEdge]
) -> tuple[float, ...]:
    """Per-node sum of relation phi over the edges involving each node."""
    sums = [0.0] * n_nodes
    for edge in edges:
        for i in edge.relata:
            sums[i] += edge.phi
    return tuple(sums)


def _inclusion_order(purview_unions: Sequence[frozenset]) -> InclusionOrder:
    """Partial order by strict subset relation on purview-unit sets."""
    n = len(purview_unions)
    below: list[set[int]] = [set() for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if a != b and purview_unions[b] < purview_unions[a]:
                below[a].add(b)
    covers = tuple(
        tuple(
            sorted(
                b
                for b in below[a]
                if not any(b in below[c] for c in below[a] if c != b)
            )
        )
        for a in range(n)
    )
    memo: dict[int, int] = {}

    def longest_chain(a: int) -> int:
        if a not in memo:
            memo[a] = (
                1 + max(longest_chain(b) for b in below[a]) if below[a] else 0
            )
        return memo[a]

    rank = tuple(longest_chain(a) for a in range(n))
    return InclusionOrder(covers=covers, rank=rank)


def _unit_indices(units) -> tuple[int, ...]:
    """Sorted integer indices from an iterable of units (or bare ints)."""
    return tuple(sorted(getattr(u, "index", u) for u in units))


def project_phi_structure(ces, node_labels=None) -> PhiStructureProjection:
    """Project a |CauseEffectStructure| into plot-ready data."""
    distinctions = list(ces.distinctions)
    if node_labels is None:
        node_labels = distinctions[0].node_labels
    mechanism_to_id = {
        tuple(d.mechanism): i for i, d in enumerate(distinctions)
    }
    edges = tuple(
        RelationEdge(
            relata=tuple(
                sorted(mechanism_to_id[tuple(m)] for m in relation.mechanisms)
            ),
            degree=len(relation),
            phi=float(relation.phi),
            overlap=_unit_indices(relation.purview),
        )
        for relation in ces.relations
    )
    unions = tuple(
        frozenset(getattr(u, "index", u) for u in d.purview_union)
        for d in distinctions
    )
    inclusion = _inclusion_order(unions)
    sums = _sum_phi_relations(len(distinctions), edges)
    nodes = tuple(
        DistinctionNode(
            id=i,
            mechanism=tuple(d.mechanism),
            label=str(d.mechanism_label),
            cause_purview=tuple(d.cause_purview),
            effect_purview=tuple(d.effect_purview),
            mechanism_state=tuple(d.mechanism_state),
            phi=float(d.phi),
            sum_phi_relations=sums[i],
            includes=bool(inclusion.covers[i]),
            included=any(i in c for c in inclusion.covers),
        )
        for i, d in enumerate(distinctions)
    )
    return PhiStructureProjection(
        nodes=nodes, edges=edges, inclusion=inclusion, node_labels=node_labels
    )
```

- [ ] **Step 4: Run the helper tests**

Run: `uv run pytest test/test_visualize_projection.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/projection/__init__.py test/test_visualize_projection.py
git -c commit.gpgsign=false commit -m "Add pure visualization projection layer with exact-value tests"
```
If the commit doesn't land (hook reformatting), re-`git add` and re-commit
(never `--no-verify`, never amend).

---

## Task 2: Builder integration test on the xor CES

**Files:**
- Modify: `test/test_visualize_projection.py`

- [ ] **Step 1: Add the integration test (session-scoped CES fixture)**

Append to `test/test_visualize_projection.py`:

```python
@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    ces = examples.xor_system().ces()
    return project_phi_structure(ces), ces


def test_project_xor_nodes(xor_projection):
    proj, ces = xor_projection
    assert len(proj.nodes) == 4
    assert [n.mechanism for n in proj.nodes] == [
        (0, 1), (0, 2), (1, 2), (0, 1, 2),
    ]
    assert [n.label for n in proj.nodes] == ["ab", "ac", "bc", "abc"]
    # First distinction's observed values (xor in (0,0,0)):
    assert proj.nodes[0].cause_purview == (0, 1, 2)
    assert proj.nodes[0].effect_purview == (2,)
    assert proj.nodes[0].phi == pytest.approx(0.5)


def test_project_xor_edges(xor_projection):
    proj, ces = xor_projection
    assert len(proj.edges) == 15
    # Every edge references valid node ids, has degree == len(relata) here
    # (xor relations involve distinct distinctions), and phi >= 0.
    for e in proj.edges:
        assert all(0 <= i < 4 for i in e.relata)
        assert e.degree >= 2
        assert e.phi >= 0


def test_project_xor_sum_phi_relations_consistent(xor_projection):
    proj, ces = xor_projection
    # Recompute independently from the projected edges.
    expected = [0.0] * 4
    for e in proj.edges:
        for i in e.relata:
            expected[i] += e.phi
    for node, exp in zip(proj.nodes, expected, strict=True):
        assert node.sum_phi_relations == pytest.approx(exp)


def test_project_xor_inclusion_monotone(xor_projection):
    proj, ces = xor_projection
    order = proj.inclusion
    # rank strictly decreases along covers edges.
    for a, cov in enumerate(order.covers):
        for b in cov:
            assert order.rank[a] > order.rank[b]
```

- [ ] **Step 2: Run**

Run: `uv run pytest test/test_visualize_projection.py -q`
Expected: all pass (module-scoped fixture computes the CES once, ~0.2 s).
If `relation.mechanisms` contains a mechanism not in `mechanism_to_id`
(a KeyError), STOP and inspect — that would mean relations can reference
distinctions outside `ces.distinctions`, and the mapping needs a guard;
investigate before weakening anything.

- [ ] **Step 3: Commit**

```bash
git add test/test_visualize_projection.py
git -c commit.gpgsign=false commit -m "Add xor integration tests for phi-structure projection"
```

---

## Task 3: Theme dataclass

**Files:**
- Create: `pyphi/visualize/theme.py`
- Modify: `test/test_visualize_projection.py` (one test appended)

- [ ] **Step 1: Implement the Theme**

Create `pyphi/visualize/theme.py`:

```python
"""Visual theme for the visualize renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Knobs read by the renderers; pass overrides via ``dataclasses.replace``."""

    colorscale: str = "Viridis"
    node_size_range: tuple[float, float] = (10.0, 36.0)
    edge_color: str = "rgba(60, 60, 60, 0.4)"
    edge_width: float = 1.0
    font_family: str = "Helvetica, Arial, sans-serif"
    background: str = "white"
    role_colors: tuple[tuple[str, str], ...] = (
        ("extended", "#e6b422"),
        ("includes", "#2f6fdb"),
        ("included", "#d85a46"),
    )


DEFAULT_THEME = Theme()
```

- [ ] **Step 2: Test it (frozen + defaults)**

Append to `test/test_visualize_projection.py`:

```python
def test_theme_frozen_with_defaults():
    import dataclasses

    from pyphi.visualize.theme import DEFAULT_THEME, Theme

    assert isinstance(DEFAULT_THEME, Theme)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_THEME.colorscale = "Plasma"  # type: ignore[misc]
    dark = dataclasses.replace(DEFAULT_THEME, background="black")
    assert dark.background == "black" and DEFAULT_THEME.background == "white"
```

Run: `uv run pytest test/test_visualize_projection.py -q` → all pass.

- [ ] **Step 3: Commit**

```bash
git add pyphi/visualize/theme.py test/test_visualize_projection.py
git -c commit.gpgsign=false commit -m "Add Theme dataclass for visualize renderers"
```

---

## Task 4: Lattice renderer

**Files:**
- Create: `pyphi/visualize/render/__init__.py`
- Create: `pyphi/visualize/render/lattice.py`
- Create: `test/test_visualize_lattice.py`

- [ ] **Step 1: Write the failing figure-structure test**

Create `test/test_visualize_lattice.py`:

```python
"""Figure-structure tests for the lattice renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


def test_lattice_figure_structure(xor_projection):
    from pyphi.visualize.render.lattice import render_lattice
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_lattice(xor_projection, DEFAULT_THEME)
    # Two traces: edges (lines) then nodes (markers).
    assert len(fig.data) == 2
    edge_trace, node_trace = fig.data
    n = len(xor_projection.nodes)
    n_edges = sum(len(c) for c in xor_projection.inclusion.covers)
    # Edge trace: each cover edge contributes (x0, x1, None).
    assert len(edge_trace.x) == 3 * n_edges
    # Node trace: one marker per distinction, y = inclusion rank.
    assert len(node_trace.x) == n
    assert tuple(node_trace.y) == tuple(
        float(r) for r in xor_projection.inclusion.rank
    )
    # Hover text mentions each distinction's label.
    for node, text in zip(xor_projection.nodes, node_trace.hovertext, strict=True):
        assert node.label in text
```

Run: `uv run pytest test/test_visualize_lattice.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.visualize.render'`.

- [ ] **Step 2: Implement the renderer**

Create `pyphi/visualize/render/__init__.py`:

```python
"""Renderers: plot-ready projection data -> figures. No IIT knowledge."""
```

Create `pyphi/visualize/render/lattice.py`:

```python
"""Inclusion-lattice (Hasse) renderer for phi-structure projections."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from pyphi.visualize.projection import PhiStructureProjection
from pyphi.visualize.theme import Theme


def _positions(projection: PhiStructureProjection) -> dict[int, tuple[float, float]]:
    """x spread within each rank (label-sorted), y = inclusion rank."""
    by_rank: dict[int, list[int]] = defaultdict(list)
    for node in projection.nodes:
        by_rank[projection.inclusion.rank[node.id]].append(node.id)
    positions: dict[int, tuple[float, float]] = {}
    for rank, ids in by_rank.items():
        ids = sorted(ids, key=lambda i: projection.nodes[i].label)
        width = len(ids) - 1
        for k, i in enumerate(ids):
            x = k - width / 2.0
            positions[i] = (x, float(rank))
    return positions


def _node_sizes(projection: PhiStructureProjection, theme: Theme) -> list[float]:
    values = [n.sum_phi_relations for n in projection.nodes]
    lo, hi = min(values), max(values)
    smin, smax = theme.node_size_range
    if hi == lo:
        return [(smin + smax) / 2.0] * len(values)
    return [smin + (v - lo) / (hi - lo) * (smax - smin) for v in values]


def render_lattice(
    projection: PhiStructureProjection,
    theme: Theme,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Draw the inclusion partial order as a 2-D Hasse diagram."""
    pos = _positions(projection)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for a, cov in enumerate(projection.inclusion.covers):
        for b in cov:
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]
    hover = [
        (
            f"<b>{n.label}</b><br>mechanism {n.mechanism} = {n.mechanism_state}"
            f"<br>cause {n.cause_purview} · effect {n.effect_purview}"
            f"<br>φ = {n.phi:.4g} · Σφ_R = {n.sum_phi_relations:.4g}"
        )
        for n in projection.nodes
    ]
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color=theme.edge_color, width=theme.edge_width),
        hoverinfo="skip",
        showlegend=False,
    )
    node_trace = go.Scatter(
        x=[pos[n.id][0] for n in projection.nodes],
        y=[pos[n.id][1] for n in projection.nodes],
        mode="markers+text",
        text=[n.label for n in projection.nodes],
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(
            size=_node_sizes(projection, theme),
            color=[n.phi for n in projection.nodes],
            colorscale=theme.colorscale,
            colorbar=dict(title="φ"),
            line=dict(width=1, color="rgba(0,0,0,0.5)"),
        ),
        showlegend=False,
    )
    if fig is None:
        fig = go.Figure()
    fig.add_traces([edge_trace, node_trace])
    fig.update_layout(
        plot_bgcolor=theme.background,
        font=dict(family=theme.font_family),
        xaxis=dict(visible=False),
        yaxis=dict(title="inclusion rank", dtick=1),
    )
    return fig
```

- [ ] **Step 3: Run the renderer test**

Run: `uv run pytest test/test_visualize_lattice.py -q`
Expected: PASS. If `node_trace.hovertext` is not iterable as written, inspect
the trace attribute once and adjust the assertion access (the invariant —
hover text contains each label — stands).

- [ ] **Step 4: Commit**

```bash
git add pyphi/visualize/render/__init__.py pyphi/visualize/render/lattice.py test/test_visualize_lattice.py
git -c commit.gpgsign=false commit -m "Add inclusion-lattice renderer for phi-structures"
```

---

## Task 5: Public entry point with view dispatch

**Files:**
- Modify: `pyphi/visualize/__init__.py`
- Modify: `test/test_visualize_lattice.py` (tests appended)

- [ ] **Step 1: Write the failing tests**

Append to `test/test_visualize_lattice.py`:

```python
def test_plot_phi_structure_lattice_view():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    fig = plot_phi_structure(ces, view="lattice")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_phi_structure_unimplemented_views_raise():
    from pyphi import examples
    from pyphi.visualize import plot_phi_structure

    ces = examples.xor_system().ces()
    for view in ("evocative", "scatter", "matrix"):
        with pytest.raises(NotImplementedError, match=view):
            plot_phi_structure(ces, view=view)
```

Run: `uv run pytest test/test_visualize_lattice.py -q`
Expected: the two new tests FAIL (`plot_phi_structure` in
`pyphi.visualize` is still the legacy re-export with a different signature).

- [ ] **Step 2: Replace the top-level export with the dispatching entry point**

In `pyphi/visualize/__init__.py`: remove the line
`from .ces import plot_phi_structure` and add (after the existing imports):

```python
from .projection import project_phi_structure
from .theme import DEFAULT_THEME, Theme

_VIEWS_PENDING = {
    "evocative": "the rebuilt 3-D simplicial-complex view (legacy version: "
    "pyphi.visualize.ces.plot_phi_structure)",
    "scatter": "the relational-role scatter view",
    "matrix": "the relation matrix/heatmap view",
}


def plot_phi_structure(
    ces, *, view="lattice", theme=DEFAULT_THEME, node_labels=None, fig=None
):
    """Plot a |CauseEffectStructure|.

    Args:
        ces (CauseEffectStructure): The phi-structure to plot (distinctions
            and relations).

    Keyword Args:
        view (str): Which rendering of the structure to produce. Currently
            ``"lattice"`` (the inclusion partial order as a 2-D Hasse
            diagram).
        theme (Theme): Visual theme.
        node_labels (NodeLabels): Labels for substrate units. Defaults to the
            labels carried by the distinctions.
        fig: An existing plotly figure to draw into.
    """
    if view in _VIEWS_PENDING:
        raise NotImplementedError(
            f"view={view!r} is not implemented yet ({_VIEWS_PENDING[view]})"
        )
    if view != "lattice":
        raise ValueError(f"unknown view {view!r}")
    from .render.lattice import render_lattice

    projection = project_phi_structure(ces, node_labels=node_labels)
    return render_lattice(projection, theme, fig=fig)
```

Keep `"plot_phi_structure"` in `__all__` (it still exists, now defined here).

- [ ] **Step 3: Run the full visualize test set**

Run: `uv run pytest test/test_visualize_projection.py test/test_visualize_lattice.py -q`
Expected: all pass.

- [ ] **Step 4: Check nothing else imported the old re-export path**

Run: `grep -rn "visualize import plot_phi_structure\|visualize.plot_phi_structure" pyphi/ test/ docs/ --include="*.py" --include="*.rst"`
Expected: only the new definition and the new tests. If a caller passed the
legacy arguments (e.g. `system=`), update it to `pyphi.visualize.ces.plot_phi_structure`.

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/__init__.py test/test_visualize_lattice.py
git -c commit.gpgsign=false commit -m "Add view-dispatching plot_phi_structure with lattice as first view"
```

---

## Task 6: Changelog, spec note, full verification, finish

**Files:**
- Create: `changelog.d/visualize-projection-lattice.feature.md`
- Modify: `docs/superpowers/specs/2026-06-04-visualize-projection-lattice-design.md` (one-line deviation note)

- [ ] **Step 1: Changelog fragment**

```bash
cat > changelog.d/visualize-projection-lattice.feature.md <<'EOF'
Added a pure projection layer for visualization
(`pyphi.visualize.projection`) and a new inclusion-lattice (Hasse) view of
phi-structures: `pyphi.visualize.plot_phi_structure(ces, view="lattice")`
draws distinctions ranked by purview inclusion, sized by total relation phi
and colored by distinction phi, with a `Theme` dataclass replacing ad-hoc
theme overrides. The legacy 3-D plot remains available as
`pyphi.visualize.ces.plot_phi_structure` until its rebuild lands.
EOF
```

- [ ] **Step 2: Record the spec deviation**

In the spec's Component A section, append one sentence to the inclusion
construction note: the implementation derives the order by subset comparison
of `purview_union` index-sets rather than adapting
`Distinctions.purview_inclusion(...)`, whose unit-keyed return shape doesn't
match the distinction-to-distinction order needed.

- [ ] **Step 3: Full verification**

Run: `uv run pytest -q` (no path argument; includes doctests) — expect 0 failures.
Run: `uv run pyright pyphi` (0 errors) and `uv run ruff check pyphi test` (clean).

- [ ] **Step 4: Commit**

```bash
git add changelog.d/visualize-projection-lattice.feature.md docs/superpowers/specs/2026-06-04-visualize-projection-lattice-design.md
git -c commit.gpgsign=false commit -m "Changelog + spec note for projection/lattice foundation"
```

- [ ] **Step 5: Finish**

Use superpowers:finishing-a-development-branch. If executing inline on `2.0`,
this collapses to the verification above — report completion; then render the
lattice on the xor example and push it to the visual companion so the
maintainer can react to the real figure (design iteration happens there).

---

## Notes for the implementer

- The projection package must import **no plotting libraries**; the renderer
  must import **no pyphi model modules** (only `projection` + `theme`).
- Do not touch `pyphi/visualize/ces/` (legacy 3-D), `distribution.py`,
  `connectivity.py`, `dynamics.py`, `ising.py` — later sub-projects.
- No golden impact (visualize feeds no φ values).
- Relation relata objects have enormous reprs — never print them in tests or
  debugging output; print mechanisms/indices instead.
