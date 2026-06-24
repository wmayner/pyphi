# Simplicial-Complex (3-D) Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the 3-D simplicial-complex phi-structure view onto the pure projection layer, port `highlight_phi_fold`, and delete the legacy `pyphi/visualize/ces/` package.

**Architecture:** The projection grows an endpoint level (`EndpointNode` per distinction×direction, `RelationFaceEdge` for degree-2/3 relation faces); a new renderer `render/simplicial_complex.py` computes cylindrical-shell geometry in pure plot space and emits one merged plotly trace per element class; the public API dispatches `view="simplicial_complex"` and reimplements `highlight_phi_fold` as two render passes over shared deterministic geometry.

**Tech Stack:** Python 3.12+, plotly (Scatter3d/Mesh3d), dataclasses, pytest. All commands via `uv run`.

**Spec:** `docs/superpowers/specs/2026-06-04-visualize-simplicial-complex-design.md`

**Verified API facts** (do not re-probe):
- xor CES: 4 distinctions; per-side MICEs via `d.cause`/`d.effect` with `.purview`, `.purview_state`, `.phi`. Values: d0 (ab) cause=(0,1,2) φ=0.5 state=(0,0,0), effect=(2,) φ=1.0 state=(0,); d1 (ac) effect=(1,) φ=1.0; d2 (bc) effect=(0,) φ=1.0; d3 (abc) cause=(0,1,2) φ=1.0, effect=(0,1,2) φ=2.0. All states are 0 → all labels lowercase.
- `ces.relations.faces_by_degree` → dict; xor: {2: 25, 3: 40, 4: 35, 5: 16, 6: 3}. Iterating a `RelationFace` yields MICE objects with `.mechanism`, `.direction` (enum; `.name` is `"CAUSE"`/`"EFFECT"`), `.purview`; face has `.phi`, `.overlap` (units with `.index`). xor has a 2-face with relata = cause+effect of d2 (bc), φ=1/6, overlap=(0,).
- `NodeLabels.indices2labels(idxs)` → tuple of labels; `NodeLabels.set_case_by_state(labels, states)` → list, upper=ON lower=OFF.
- Commit dance: hooks may reformat; if a commit doesn't land, re-`git add` the same files and commit again. Never `--no-verify`, never amend. Targeted `git add <files>` only. Use `git -c commit.gpgsign=false commit`.

---

### Task 1: Theme additions

**Files:**
- Modify: `pyphi/visualize/theme.py`
- Test: `test/test_visualize_projection.py`

- [ ] **Step 1: Write the failing test** — append to `test/test_visualize_projection.py`:

```python
def test_theme_simplicial_complex_fields():
    from pyphi.visualize.theme import DEFAULT_THEME

    assert DEFAULT_THEME.cause_color == "#8D3D00"
    assert DEFAULT_THEME.effect_color == "#006146"
    assert DEFAULT_THEME.face_colorscale == "Blues"
    assert 0 < DEFAULT_THEME.face_opacity <= 1
    assert DEFAULT_THEME.text_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_projection.py::test_theme_simplicial_complex_fields -q`
Expected: FAIL with `AttributeError: ... 'cause_color'`

- [ ] **Step 3: Implement** — in `pyphi/visualize/theme.py`, add fields to `Theme` after `role_colors`:

```python
    cause_color: str = "#8D3D00"
    effect_color: str = "#006146"
    face_colorscale: str = "Blues"
    face_opacity: float = 0.2
    text_size: int = 12
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_projection.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/theme.py test/test_visualize_projection.py
git -c commit.gpgsign=false commit -m "Add simplicial-complex style fields to Theme"
```

---

### Task 2: Projection endpoints and faces

**Files:**
- Modify: `pyphi/visualize/projection/__init__.py`
- Test: `test/test_visualize_projection.py`

- [ ] **Step 1: Write the failing tests** — append to `test/test_visualize_projection.py` (the module-scoped `xor_projection` fixture already exists there):

```python
def test_state_cased_label():
    from pyphi.labels import NodeLabels
    from pyphi.visualize.projection import _state_cased_label

    nl = NodeLabels(("A", "B", "C"), (0, 1, 2))
    assert _state_cased_label((0, 2), (1, 0), nl) == "Ac"
    assert _state_cased_label((0, 1, 2), (0, 0, 0), nl) == "abc"


def test_project_xor_endpoints_exact(xor_projection):
    eps = xor_projection.endpoints
    assert len(eps) == 8
    assert tuple(e.id for e in eps) == tuple(range(8))
    # Interleaved cause/effect per distinction.
    assert [e.direction for e in eps] == ["cause", "effect"] * 4
    assert [e.distinction_id for e in eps] == [0, 0, 1, 1, 2, 2, 3, 3]
    # d0 = ab: cause over the whole substrate, effect on c.
    assert eps[0].purview == (0, 1, 2)
    assert eps[0].purview_state == (0, 0, 0)
    assert eps[0].phi == pytest.approx(0.5)
    assert eps[0].label == "abc"
    assert eps[1].purview == (2,)
    assert eps[1].phi == pytest.approx(1.0)
    assert eps[1].label == "c"
    # d3 = abc: effect phi 2.
    assert eps[7].purview == (0, 1, 2)
    assert eps[7].phi == pytest.approx(2.0)


def test_project_xor_faces(xor_projection):
    faces = xor_projection.faces
    by_degree = {}
    for f in faces:
        by_degree.setdefault(f.degree, []).append(f)
        assert f.degree == len(f.endpoints)
        assert all(0 <= i < 8 for i in f.endpoints)
        assert f.phi >= 0
    assert len(by_degree[2]) == 25
    assert len(by_degree[3]) == 40
    assert set(by_degree) == {2, 3}
    # Known face: cause and effect of d2 (bc) related over unit a.
    assert any(
        f.endpoints == (4, 5)
        and f.phi == pytest.approx(1 / 6)
        and f.overlap == (0,)
        for f in by_degree[2]
    )


def test_projection_faces_deterministic(xor_projection):
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    again = project_phi_structure(examples.xor_system().ces())
    assert again.faces == xor_projection.faces
    assert again.endpoints == xor_projection.endpoints
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_projection.py -q`
Expected: new tests FAIL (`ImportError: _state_cased_label` / `AttributeError: endpoints`)

- [ ] **Step 3: Implement** — in `pyphi/visualize/projection/__init__.py`:

Add to `__all__`: `"EndpointNode"`, `"RelationFaceEdge"`.

Add dataclasses after `RelationEdge`:

```python
@dataclass(frozen=True)
class EndpointNode:
    """Plot-ready data for one side (cause or effect) of a distinction."""

    id: int
    distinction_id: int
    direction: str
    purview: tuple[int, ...]
    purview_state: tuple[int, ...]
    phi: float
    label: str


@dataclass(frozen=True)
class RelationFaceEdge:
    """Plot-ready data for one degree-2 or degree-3 relation face."""

    endpoints: tuple[int, ...]
    degree: int
    phi: float
    overlap: tuple[int, ...]
```

Add fields to `PhiStructureProjection` (after `node_labels`, with defaults so existing constructions remain valid). Extend the class docstring: endpoints are interleaved cause/effect per distinction (`id == 2 * distinction_id + (0 cause, 1 effect)`); faces carry only degrees 2 and 3 (the drawable simplices).

```python
    endpoints: tuple[EndpointNode, ...] = ()
    faces: tuple[RelationFaceEdge, ...] = ()
```

Add builders before `project_phi_structure`:

```python
def _state_cased_label(purview, purview_state, node_labels) -> str:
    """Purview label with case set by state (upper = ON, lower = OFF)."""
    return "".join(
        node_labels.set_case_by_state(
            node_labels.indices2labels(purview), purview_state
        )
    )


def _endpoints(distinctions, node_labels) -> tuple[EndpointNode, ...]:
    endpoints = []
    for i, d in enumerate(distinctions):
        for j, (direction, mice) in enumerate(
            (("cause", d.cause), ("effect", d.effect))
        ):
            purview = tuple(mice.purview)
            state = tuple(mice.purview_state)
            endpoints.append(
                EndpointNode(
                    id=2 * i + j,
                    distinction_id=i,
                    direction=direction,
                    purview=purview,
                    purview_state=state,
                    phi=float(mice.phi),
                    label=_state_cased_label(purview, state, node_labels),
                )
            )
    return tuple(endpoints)


def _faces(relations, mechanism_to_id) -> tuple[RelationFaceEdge, ...]:
    by_degree = relations.faces_by_degree
    faces = []
    for degree in (2, 3):
        for face in by_degree.get(degree, ()):
            endpoint_ids = tuple(
                sorted(
                    2 * mechanism_to_id[tuple(relatum.mechanism)]
                    + (0 if relatum.direction.name == "CAUSE" else 1)
                    for relatum in face
                )
            )
            faces.append(
                RelationFaceEdge(
                    endpoints=endpoint_ids,
                    degree=degree,
                    phi=float(face.phi),
                    overlap=_unit_indices(face.overlap),
                )
            )
    faces.sort(key=lambda f: (f.degree, f.endpoints, f.phi))
    return tuple(faces)
```

In `project_phi_structure`, after `inclusion`/`sums` are computed, add:

```python
    endpoints = _endpoints(distinctions, node_labels)
    faces = _faces(ces.relations, mechanism_to_id)
```

and pass `endpoints=endpoints, faces=faces` to the `PhiStructureProjection` constructor.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_projection.py test/test_visualize_lattice.py -q`
Expected: all PASS (lattice unaffected)

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/projection/__init__.py test/test_visualize_projection.py
git -c commit.gpgsign=false commit -m "Project distinction endpoints and degree-2/3 relation faces"
```

---

### Task 3: Geometry

**Files:**
- Create: `pyphi/visualize/render/simplicial_complex.py`
- Test: `test/test_visualize_simplicial_complex.py` (new)

- [ ] **Step 1: Write the failing tests** — create `test/test_visualize_simplicial_complex.py`:

```python
"""Tests for the 3-D simplicial-complex renderer."""

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_phi_structure

    return project_phi_structure(examples.xor_system().ces())


def test_geometry_dataclass_frozen():
    import dataclasses

    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry

    geo = SimplicialComplexGeometry()
    assert geo.max_radius == 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        geo.max_radius = 2.0  # type: ignore[misc]


def test_endpoint_positions(xor_projection):
    from pyphi.visualize.render.simplicial_complex import (
        SimplicialComplexGeometry,
        _endpoint_positions,
    )

    geo = SimplicialComplexGeometry()
    pos = _endpoint_positions(xor_projection, geo)
    assert set(pos) == set(range(8))
    # Deterministic.
    assert pos == _endpoint_positions(xor_projection, geo)
    # Flat by default.
    assert all(p[2] == 0.0 for p in pos.values())
    # d3's cause/effect share the purview (0,1,2): cause sits -x, effect +x.
    assert pos[6][0] < pos[7][0]
    # Endpoints sharing (purview, direction) are jittered apart:
    # causes of d0, d1, d2, d3 all have purview (0,1,2).
    cause_ids = (0, 2, 4, 6)
    assert len({pos[i] for i in cause_ids}) == 4


def test_mechanism_positions(xor_projection):
    from pyphi.visualize.render.simplicial_complex import (
        SimplicialComplexGeometry,
        _mechanism_positions,
    )

    geo = SimplicialComplexGeometry(max_radius=2.0)
    pos = _mechanism_positions(xor_projection, geo)
    assert set(pos) == {0, 1, 2, 3}
    # Mechanisms are unique, so all positions distinct.
    assert len(set(pos.values())) == 4
    # abc (size 3) sits on the outermost shell at max_radius.
    x, y, z = pos[3]
    assert (x**2 + y**2) ** 0.5 == pytest.approx(2.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `pyphi/visualize/render/simplicial_complex.py`:

```python
"""3-D simplicial-complex renderer for phi-structure projections."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import plotly.graph_objects as go

from pyphi.visualize.projection import PhiStructureProjection
from pyphi.visualize.theme import Theme

Point = tuple[float, float, float]


@dataclass(frozen=True)
class SimplicialComplexGeometry:
    """Plot-space layout knobs for the simplicial-complex view."""

    max_radius: float = 1.0
    z_spacing: float = 0.0
    direction_offset: float = 0.5
    purview_jitter: float = 0.1


def _polygon_points(n: int, radius: float, z: float) -> list[Point]:
    """``n`` points evenly spaced on a circle of ``radius`` at height ``z``."""
    return [
        (
            radius * math.cos(2 * math.pi * k / n),
            radius * math.sin(2 * math.pi * k / n),
            z,
        )
        for k in range(n)
    ]


def _shell_positions(
    subsets: Iterable[tuple[int, ...]], geometry: SimplicialComplexGeometry
) -> dict[tuple[int, ...], Point]:
    """Place each unique subset on the shell for its size.

    Subsets of size k share a circular shell whose radius grows linearly
    with k up to ``max_radius``; within a shell, subsets sit on a regular
    polygon in sorted order. Shells stack in z by ``z_spacing``.
    """
    by_size: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for s in sorted(set(subsets)):
        by_size[len(s)].append(s)
    sizes = sorted(by_size)
    k_max = max(sizes)
    positions: dict[tuple[int, ...], Point] = {}
    for shell_index, k in enumerate(sizes):
        members = by_size[k]
        radius = geometry.max_radius * k / k_max
        z = geometry.z_spacing * shell_index
        for s, p in zip(
            members, _polygon_points(len(members), radius, z), strict=True
        ):
            positions[s] = p
    return positions


def _endpoint_positions(
    projection: PhiStructureProjection, geometry: SimplicialComplexGeometry
) -> dict[int, Point]:
    """Position each endpoint near its purview's shell point.

    Cause endpoints shift -x and effect endpoints +x by
    ``direction_offset``; endpoints sharing a purview and direction spread
    on a small polygon of radius ``purview_jitter``.
    """
    base = _shell_positions((e.purview for e in projection.endpoints), geometry)
    groups: dict[tuple[tuple[int, ...], str], list[int]] = defaultdict(list)
    for e in projection.endpoints:
        groups[(e.purview, e.direction)].append(e.id)
    positions: dict[int, Point] = {}
    for (purview, direction), ids in groups.items():
        bx, by, bz = base[purview]
        bx += (
            geometry.direction_offset
            if direction == "effect"
            else -geometry.direction_offset
        )
        jitter = geometry.purview_jitter if len(ids) > 1 else 0.0
        offsets = _polygon_points(len(ids), jitter, 0.0)
        for eid, (ox, oy, _) in zip(sorted(ids), offsets, strict=True):
            positions[eid] = (bx + ox, by + oy, bz)
    return positions


def _mechanism_positions(
    projection: PhiStructureProjection, geometry: SimplicialComplexGeometry
) -> dict[int, Point]:
    """Position each distinction's mechanism on its size shell."""
    base = _shell_positions((n.mechanism for n in projection.nodes), geometry)
    return {n.id: base[n.mechanism] for n in projection.nodes}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/render/simplicial_complex.py test/test_visualize_simplicial_complex.py
git -c commit.gpgsign=false commit -m "Add shell geometry for the simplicial-complex renderer"
```

---

### Task 4: Traces and render function

**Files:**
- Modify: `pyphi/visualize/render/simplicial_complex.py`
- Test: `test/test_visualize_simplicial_complex.py`

- [ ] **Step 1: Write the failing tests** — append to `test/test_visualize_simplicial_complex.py`:

```python
def _render(projection, **kwargs):
    from pyphi.visualize.render.simplicial_complex import render_simplicial_complex
    from pyphi.visualize.theme import DEFAULT_THEME

    return render_simplicial_complex(projection, DEFAULT_THEME, **kwargs)


def test_render_full_figure_structure(xor_projection):
    import plotly.graph_objects as go

    fig = _render(xor_projection)
    assert isinstance(fig, go.Figure)
    # One trace per element class, in declaration order.
    assert len(fig.data) == 6
    purviews, mechanisms, ce_links, mp_links, two_faces, mesh = fig.data
    assert len(purviews.x) == 8
    assert len(mechanisms.x) == 4
    # Cause-effect links: (cause, effect, None) per distinction.
    assert len(ce_links.x) == 3 * 4
    # Mechanism-purview links: (cause, mechanism, effect, None) per distinction.
    assert len(mp_links.x) == 4 * 4
    # 25 degree-2 faces, (a, b, None) each.
    assert len(two_faces.x) == 3 * 25
    # 40 degree-3 faces as one mesh.
    assert isinstance(mesh, go.Mesh3d)
    assert len(mesh.i) == 40
    # Endpoint labels present.
    assert "abc" in purviews.text and "c" in purviews.text


def test_render_show_subsetting(xor_projection):
    fig = _render(xor_projection, show=("purviews",))
    assert len(fig.data) == 1
    fig = _render(xor_projection, show=("purviews", "three_faces"))
    assert len(fig.data) == 2
    with pytest.raises(ValueError, match="show"):
        _render(xor_projection, show=("purviews", "bogus"))


def test_render_only_distinctions_filters_without_moving(xor_projection):
    full = _render(xor_projection)
    sub = _render(xor_projection, only_distinctions={0, 3})
    full_points = set(zip(full.data[0].x, full.data[0].y, full.data[0].z))
    sub_points = set(zip(sub.data[0].x, sub.data[0].y, sub.data[0].z))
    # 2 distinctions -> 4 endpoints, at unchanged coordinates.
    assert len(sub_points) == 4
    assert sub_points <= full_points
    # Faces restricted to those entirely within the subset.
    full_mesh, sub_mesh = full.data[5], sub.data[5]
    assert len(sub_mesh.i) < len(full_mesh.i)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q`
Expected: new tests FAIL (`ImportError: render_simplicial_complex`)

- [ ] **Step 3: Implement** — append to `pyphi/visualize/render/simplicial_complex.py`:

```python
_ELEMENTS = (
    "purviews",
    "mechanisms",
    "cause_effect_links",
    "mechanism_purview_links",
    "two_faces",
    "three_faces",
)


def _rescale(values: list[float], lo: float, hi: float) -> list[float]:
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [(lo + hi) / 2.0] * len(values)
    return [lo + (v - vmin) / (vmax - vmin) * (hi - lo) for v in values]


def _segments(
    paths: Iterable[tuple[Point, ...]],
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """None-separated coordinate arrays from point paths."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for path in paths:
        for x, y, z in path:
            xs.append(x)
            ys.append(y)
            zs.append(z)
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return xs, ys, zs


def _purview_trace(endpoints, pos, theme):
    hover = [
        (
            f"<b>{e.label}</b> ({e.direction})"
            f"<br>purview {e.purview} = {e.purview_state}"
            f"<br>φ = {e.phi:.4g}"
        )
        for e in endpoints
    ]
    return go.Scatter3d(
        x=[pos[e.id][0] for e in endpoints],
        y=[pos[e.id][1] for e in endpoints],
        z=[pos[e.id][2] for e in endpoints],
        mode="markers+text",
        text=[e.label for e in endpoints],
        textposition="top center",
        textfont={
            "size": theme.text_size,
            "color": [
                theme.cause_color if e.direction == "cause" else theme.effect_color
                for e in endpoints
            ],
        },
        hovertext=hover,
        hoverinfo="text",
        marker={
            "size": _rescale([e.phi for e in endpoints], *theme.node_size_range),
            "color": [e.phi for e in endpoints],
            "colorscale": theme.colorscale,
            "showscale": False,
            "line": {"width": 1, "color": "rgba(0,0,0,0.5)"},
        },
        showlegend=False,
    )


def _mechanism_trace(nodes, pos, theme):
    hover = [
        f"<b>{n.label}</b><br>mechanism {n.mechanism} = {n.mechanism_state}"
        f"<br>φ = {n.phi:.4g}"
        for n in nodes
    ]
    return go.Scatter3d(
        x=[pos[n.id][0] for n in nodes],
        y=[pos[n.id][1] for n in nodes],
        z=[pos[n.id][2] for n in nodes],
        mode="text",
        text=[n.label for n in nodes],
        textfont={"size": theme.text_size},
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )


def _link_trace(paths, theme):
    xs, ys, zs = _segments(paths)
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": theme.edge_color, "width": 2 * theme.edge_width},
        hoverinfo="skip",
        showlegend=False,
    )


def _two_face_trace(faces, endpoint_pos, theme):
    xs, ys, zs = _segments(
        tuple(endpoint_pos[i] for i in f.endpoints) for f in faces
    )
    # One color value per vertex, including the None separators.
    colors = [phi for f in faces for phi in [f.phi] * 3]
    hover = [
        f"2-face<br>overlap {f.overlap}<br>φ = {f.phi:.4g}"
        for f in faces
        for _ in range(3)
    ]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={
            "color": colors,
            "colorscale": theme.face_colorscale,
            "width": 2 * theme.edge_width,
        },
        hovertext=hover,
        hoverinfo="text",
        showlegend=False,
    )


def _three_face_trace(faces, endpoint_pos, theme):
    n = max(endpoint_pos) + 1
    xs = [endpoint_pos.get(i, (0.0, 0.0, 0.0))[0] for i in range(n)]
    ys = [endpoint_pos.get(i, (0.0, 0.0, 0.0))[1] for i in range(n)]
    zs = [endpoint_pos.get(i, (0.0, 0.0, 0.0))[2] for i in range(n)]
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=[f.endpoints[0] for f in faces],
        j=[f.endpoints[1] for f in faces],
        k=[f.endpoints[2] for f in faces],
        intensity=[f.phi for f in faces],
        intensitymode="cell",
        colorscale=theme.face_colorscale,
        opacity=theme.face_opacity,
        showscale=False,
        hoverinfo="skip",
    )


def render_simplicial_complex(
    projection: PhiStructureProjection,
    theme: Theme,
    fig: go.Figure | None = None,
    geometry: SimplicialComplexGeometry | None = None,
    show: tuple[str, ...] = _ELEMENTS,
    only_distinctions: set[int] | None = None,
) -> go.Figure:
    """Draw the phi-structure as a 3-D simplicial complex.

    Purview endpoints are vertices; degree-2 relation faces are line
    segments and degree-3 faces are triangles. Geometry is computed from
    the full projection regardless of ``only_distinctions``, so successive
    calls with different subsets align (the primitive
    ``highlight_phi_fold`` composes on).
    """
    unknown = set(show) - set(_ELEMENTS)
    if unknown:
        raise ValueError(f"unknown show element(s) {sorted(unknown)!r}")
    if geometry is None:
        geometry = SimplicialComplexGeometry()
    endpoint_pos = _endpoint_positions(projection, geometry)
    mechanism_pos = _mechanism_positions(projection, geometry)
    included = (
        set(range(len(projection.nodes)))
        if only_distinctions is None
        else set(only_distinctions)
    )
    endpoints = [e for e in projection.endpoints if e.distinction_id in included]
    nodes = [n for n in projection.nodes if n.id in included]
    faces = [
        f
        for f in projection.faces
        if all(projection.endpoints[i].distinction_id in included for i in f.endpoints)
    ]
    two_faces = [f for f in faces if f.degree == 2]
    three_faces = [f for f in faces if f.degree == 3]
    traces = []
    if "purviews" in show:
        traces.append(_purview_trace(endpoints, endpoint_pos, theme))
    if "mechanisms" in show:
        traces.append(_mechanism_trace(nodes, mechanism_pos, theme))
    if "cause_effect_links" in show:
        traces.append(
            _link_trace(
                (
                    (endpoint_pos[2 * n.id], endpoint_pos[2 * n.id + 1])
                    for n in nodes
                ),
                theme,
            )
        )
    if "mechanism_purview_links" in show:
        traces.append(
            _link_trace(
                (
                    (
                        endpoint_pos[2 * n.id],
                        mechanism_pos[n.id],
                        endpoint_pos[2 * n.id + 1],
                    )
                    for n in nodes
                ),
                theme,
            )
        )
    if "two_faces" in show and two_faces:
        traces.append(_two_face_trace(two_faces, endpoint_pos, theme))
    if "three_faces" in show and three_faces:
        traces.append(_three_face_trace(three_faces, endpoint_pos, theme))
    figure = go.Figure() if fig is None else fig
    figure.add_traces(traces)
    axis = {"visible": False}
    figure.update_layout(
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis},
        paper_bgcolor=theme.background,
        font={"family": theme.font_family},
        showlegend=False,
    )
    return figure
```

Note for the tests: with xor and default `show`, two_faces and three_faces are non-empty, so 6 traces. The `only_distinctions` test indexes `fig.data[5]` — with `{0, 3}` both face classes remain non-empty (the d0/d3 self-relation faces survive); if that assumption fails at runtime, select the Mesh3d trace by `isinstance` instead.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/render/simplicial_complex.py test/test_visualize_simplicial_complex.py
git -c commit.gpgsign=false commit -m "Add simplicial-complex trace builders and render function"
```

---

### Task 5: Public API dispatch

**Files:**
- Modify: `pyphi/visualize/__init__.py`
- Test: `test/test_visualize_simplicial_complex.py`, `test/test_visualize_lattice.py`

- [ ] **Step 1: Write the failing tests** — append to `test/test_visualize_simplicial_complex.py`:

```python
def test_plot_phi_structure_simplicial_complex_view():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_phi_structure
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry

    ces = examples.xor_system().ces()
    fig = plot_phi_structure(ces, view="simplicial_complex")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6
    fig = plot_phi_structure(
        ces,
        view="simplicial_complex",
        geometry=SimplicialComplexGeometry(z_spacing=0.3),
        show=("purviews",),
    )
    assert len(fig.data) == 1
```

In `test/test_visualize_lattice.py`, update `test_plot_phi_structure_unimplemented_views_raise`:

```python
    for view in ("scatter", "matrix"):
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_simplicial_complex.py::test_plot_phi_structure_simplicial_complex_view -q`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement** — in `pyphi/visualize/__init__.py`:

Remove the `"simplicial_complex"` entry from `_VIEWS_PENDING`. Add parameters `geometry=None, show=None` to `plot_phi_structure` after `color_by`, document them as applying to the simplicial-complex view (geometry: a `SimplicialComplexGeometry`; show: which element classes to draw), document the new `view` value, and replace the dispatch tail:

```python
    if view in _VIEWS_PENDING:
        raise NotImplementedError(
            f"view={view!r} is not implemented yet ({_VIEWS_PENDING[view]})"
        )
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
            color_by=color_by,
        )
    if view == "simplicial_complex":
        from .render.simplicial_complex import render_simplicial_complex

        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if show is not None:
            kwargs["show"] = show
        return render_simplicial_complex(projection, theme, fig=fig, **kwargs)
    raise ValueError(f"unknown view {view!r}")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_simplicial_complex.py test/test_visualize_lattice.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/__init__.py test/test_visualize_simplicial_complex.py test/test_visualize_lattice.py
git -c commit.gpgsign=false commit -m "Dispatch view=\"simplicial_complex\" in plot_phi_structure"
```

---

### Task 6: highlight_phi_fold

**Files:**
- Modify: `pyphi/visualize/__init__.py`
- Test: `test/test_visualize_simplicial_complex.py`

- [ ] **Step 1: Write the failing test** — append:

```python
def test_highlight_phi_fold_smoke():
    from types import SimpleNamespace

    from pyphi import examples
    from pyphi.visualize import highlight_phi_fold

    ces = examples.xor_system().ces()
    fold = SimpleNamespace(distinctions=list(ces.distinctions)[:2])
    fig = highlight_phi_fold(ces, fold)
    # Two passes: dimmed full structure + highlighted fold.
    assert len(fig.data) == 12
    # The overlay's endpoint coordinates are a subset of the background's.
    bg, overlay = fig.data[0], fig.data[6]
    bg_points = set(zip(bg.x, bg.y, bg.z))
    overlay_points = set(zip(overlay.x, overlay.y, overlay.z))
    assert len(overlay.x) == 4
    assert overlay_points <= bg_points
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/test_visualize_simplicial_complex.py::test_highlight_phi_fold_smoke -q`
Expected: FAIL (the legacy import is still in place at this point; the test fails on its different signature/return — or on figure structure). If the legacy symbol shadows confusingly, proceed: Step 3 removes it.

- [ ] **Step 3: Implement** — in `pyphi/visualize/__init__.py`:

Remove `from .ces import highlight_phi_fold`. Add `import dataclasses` at the top of the imports and define after `plot_phi_structure`:

```python
def highlight_phi_fold(
    ces_,
    phi_fold,
    *,
    theme=DEFAULT_THEME,
    node_labels=None,
    fig=None,
    geometry=None,
    show=None,
):
    """Plot a |CauseEffectStructure| dimmed, highlighting a phi-fold.

    Args:
        ces_ (CauseEffectStructure): The full phi-structure.
        phi_fold: An object with a ``distinctions`` attribute giving the
            distinctions to highlight; they are matched to the structure's
            by mechanism.

    Keyword Args:
        theme (Theme): Visual theme for the highlighted fold; the dimmed
            background style is derived from it.
        node_labels (NodeLabels): Labels for substrate units.
        fig: An existing plotly figure to draw into.
        geometry (SimplicialComplexGeometry): Layout knobs.
        show (tuple[str, ...]): Element classes to draw.
    """
    from .render.simplicial_complex import render_simplicial_complex

    projection = project_phi_structure(ces_, node_labels=node_labels)
    dimmed = dataclasses.replace(
        theme,
        colorscale="Greys",
        face_colorscale="Greys",
        cause_color="#999999",
        effect_color="#999999",
        edge_color="rgba(150, 150, 150, 0.2)",
        face_opacity=theme.face_opacity * 0.25,
    )
    kwargs = {}
    if geometry is not None:
        kwargs["geometry"] = geometry
    if show is not None:
        kwargs["show"] = show
    figure = render_simplicial_complex(projection, dimmed, fig=fig, **kwargs)
    fold_mechanisms = {tuple(d.mechanism) for d in phi_fold.distinctions}
    fold_ids = {
        n.id for n in projection.nodes if n.mechanism in fold_mechanisms
    }
    return render_simplicial_complex(
        projection, theme, fig=figure, only_distinctions=fold_ids, **kwargs
    )
```

Note: with `only_distinctions={0, 1}`, both face classes are non-empty for xor (d0/d1 cross- and self-relation faces exist), so the overlay also has 6 traces; if the 12-trace assertion fails at runtime, count the overlay's traces from `len(fig.data)` after the first pass instead of hard-coding.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/__init__.py test/test_visualize_simplicial_complex.py
git -c commit.gpgsign=false commit -m "Reimplement highlight_phi_fold over the simplicial-complex renderer"
```

---

### Task 7: Delete the legacy package

**Files:**
- Delete: `pyphi/visualize/ces/` (entire directory: `__init__.py`, `colors.py`, `geometry.py`, `text.py`, `theme.py`, `utils.py`)
- Modify: `pyphi/visualize/__init__.py`

- [ ] **Step 1: Remove the legacy imports** — in `pyphi/visualize/__init__.py`, delete the line `from . import ces` and remove `"ces"` from `__all__`. Also update the `_VIEWS_PENDING` messages if any still reference `pyphi.visualize.ces`.

- [ ] **Step 2: Delete the package**

```bash
git rm -r pyphi/visualize/ces
```

- [ ] **Step 3: Verify no dangling references**

Run: `grep -rn "visualize.ces\|visualize import ces\|from .ces" pyphi/ test/ docs/ --include="*.py" --include="*.rst"`
Expected: no output (the `pyphi/models/ces.py` import in `pyphi/models/__init__.py` is a different module and does not match these patterns; if it appears, leave it alone)

- [ ] **Step 4: Full verification**

Run: `uv run pytest -q` (no path argument — doctest-inclusive)
Expected: all PASS
Run: `uv run ruff check pyphi test`
Expected: clean

- [ ] **Step 5: Commit**

```bash
git add pyphi/visualize/__init__.py
git -c commit.gpgsign=false commit -m "Delete legacy 3-D phi-structure renderer (pyphi/visualize/ces/)"
```

(The `git rm` already staged the deletions.)

---

### Task 8: Changelog and visual check

**Files:**
- Modify: `changelog.d/visualize-projection-lattice.feature.md`

- [ ] **Step 1: Update the changelog fragment** — replace its final sentence ("The legacy 3-D plot remains available as `pyphi.visualize.ces.plot_phi_structure` until its rebuild lands.") with:

```
The 3-D simplicial-complex view is rebuilt on the same projection
(`view="simplicial_complex"`, with `show` element selection and a
`SimplicialComplexGeometry` layout dataclass), `highlight_phi_fold` is
reimplemented over it, and the legacy `pyphi.visualize.ces` module is
removed.
```

- [ ] **Step 2: Render to the visual companion** for maintainer reaction (not a test): xor and rule154 via `plot_phi_structure(ces, view="simplicial_complex")`, plus one `highlight_phi_fold` example, written to the companion screen dir.

- [ ] **Step 3: Commit**

```bash
git add changelog.d/visualize-projection-lattice.feature.md
git -c commit.gpgsign=false commit -m "Update visualize changelog fragment for the simplicial-complex view"
```

---

## Final verification

```bash
uv run pytest -q          # full suite + doctests
uv run ruff check pyphi test
uv run pyright pyphi      # CLI may be silenced by the local workaround; the pre-commit hook is authoritative
```
