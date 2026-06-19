# P14d A-5 — Higher-degree relation visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make degree-≥4 relation faces visible in the simplicial-complex view via a star expansion (hub + spokes per face, sized/colored by φ), carry all face degrees through the projection, add a `higher_faces` show element (default-on) + a `degrees=` knob, and add a `view="spectrum"` φ-by-degree panel.

**Architecture:** Three touch points in the decoupled `projection/` + `render/` layer: `_faces()` stops dropping degree ≥4; `render/simplicial_complex.py` gains a `_higher_face_trace` star renderer + `higher_faces` element + `degrees=` filter; a new `render/spectrum.py` aggregates the projection's relation-level edges into a per-degree bar panel, dispatched via `plot_ces(view="spectrum")`.

**Tech Stack:** Python 3.12+, plotly, pytest. Visualization requires the `visualize` extra.

**Spec:** `docs/superpowers/specs/2026-06-18-p14d-a5-higher-degree-relation-viz-design.md`

## Global Constraints

- Python 3.12+ only; no backward-compatibility shims; no new dependency.
- Work only in `pyphi/visualize/projection/` and `pyphi/visualize/render/` and `pyphi/visualize/__init__.py` — do not touch `visualize/ces/` (a separate decoupling concern) or the relation/CES formalism.
- The star expansion applies only to degree ≥4; degree-2 (lines) and degree-3 (mesh) rendering is unchanged.
- New render code imports plotly the same way the existing render modules do (the package already raises `MissingOptionalDependenciesError` at import if the extra is absent).
- Use `uv run` for all Python commands; run with the `visualize` extra present (`uv sync --all-extras`). Final verification runs `uv run pytest` **with no path argument**.
- Do not bypass pre-commit hooks. Stage only the files each task names (the tree has unrelated untracked work; never `git add -A`).

---

### Task 1: Projection carries all degrees + star-expansion renderer

**Files:**
- Modify: `pyphi/visualize/projection/__init__.py` (`_faces` — all degrees)
- Modify: `pyphi/visualize/render/simplicial_complex.py` (`_ELEMENTS`, `_higher_face_trace`, `render_simplicial_complex`)
- Test: `test/test_visualize_projection.py`, `test/test_visualize_simplicial_complex.py`

**Interfaces:**
- Consumes: `RelationFaceEdge` (already carries `endpoints`/`degree`/`phi`/`overlap`); `endpoint_pos` (dict id→(x,y,z)); `_segments`; `rescale`; `theme.face_colorscale`/`node_size_range`/`edge_color`/`edge_width`.
- Produces: `_higher_face_trace(faces, endpoint_pos, theme, show_colorbar=True) -> list[go.Scatter3d]`; `render_simplicial_complex(..., degrees: tuple[int, ...] | None = None)`; `"higher_faces"` in `_ELEMENTS`.

- [ ] **Step 1: Write the failing projection test**

In `test/test_visualize_projection.py`, replace the body of `test_project_xor_faces` (currently asserting only degrees 2/3) with one asserting all degrees are carried:

```python
def test_project_xor_faces(xor_projection):
    faces = xor_projection.faces
    by_degree = {}
    for f in faces:
        by_degree.setdefault(f.degree, []).append(f)
        assert f.degree == len(f.endpoints)
        assert all(0 <= i < 8 for i in f.endpoints)
        assert f.phi >= 0
    # All face degrees are carried now (degree >= 4 used to be dropped).
    assert {d: len(v) for d, v in sorted(by_degree.items())} == {
        2: 25,
        3: 40,
        4: 35,
        5: 16,
        6: 3,
    }
    # Known face: cause and effect of d2 (bc) related over unit a.
    assert any(
        f.endpoints == (4, 5) and f.phi == pytest.approx(1 / 6) and f.overlap == (0,)
        for f in by_degree[2]
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest test/test_visualize_projection.py::test_project_xor_faces -q`
Expected: FAIL (`by_degree` has only `{2, 3}`; degree ≥4 dropped).

- [ ] **Step 3: Make `_faces` carry all degrees**

In `pyphi/visualize/projection/__init__.py`, change `_faces` (lines ~202-223). Replace:

```python
def _faces(relations, mechanism_to_id) -> tuple[RelationFaceEdge, ...]:
    by_degree = relations.faces_by_degree
    faces = []
    for degree in (2, 3):
        for face in by_degree.get(degree, ()):
```

with:

```python
def _faces(relations, mechanism_to_id) -> tuple[RelationFaceEdge, ...]:
    by_degree = relations.faces_by_degree
    faces = []
    for degree in sorted(by_degree):
        for face in by_degree.get(degree, ()):
```

And update the `RelationFaceEdge` docstring (`pyphi/visualize/projection/__init__.py:66-67`) from `"""Plot-ready data for one degree-2 or degree-3 relation face."""` to `"""Plot-ready data for one relation face (any degree)."""`.

- [ ] **Step 4: Run the projection test to verify it passes**

Run: `uv run pytest test/test_visualize_projection.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing renderer tests**

In `test/test_visualize_simplicial_complex.py`, add (the file already has the `xor_projection` fixture and a `_render` helper calling `render_simplicial_complex(projection, DEFAULT_THEME, **kwargs)`):

```python
def test_higher_face_trace_hub_at_centroid():
    from pyphi.visualize.render.simplicial_complex import _higher_face_trace
    from pyphi.visualize.theme import DEFAULT_THEME

    class _Face:
        endpoints = (0, 1, 2, 3)
        degree = 4
        phi = 0.5
        overlap = (0,)

    endpoint_pos = {0: (0, 0, 0), 1: (2, 0, 0), 2: (0, 2, 0), 3: (2, 2, 0)}
    hub_trace, spoke_trace = _higher_face_trace([_Face()], endpoint_pos, DEFAULT_THEME)
    # Hub sits at the centroid of the four endpoints.
    assert (hub_trace.x[0], hub_trace.y[0], hub_trace.z[0]) == (1.0, 1.0, 0.0)
    # One hub marker per face.
    assert len(hub_trace.x) == 1
    # Four spokes, each (hub, endpoint, None) -> 3 coords.
    assert len(spoke_trace.x) == 3 * 4


def test_render_includes_higher_face_stars(xor_projection):
    import plotly.graph_objects as go

    fig = _render(xor_projection)
    # Default-on higher_faces adds a hub trace + a spoke trace after the six
    # base element traces.
    assert len(fig.data) == 8
    hub, spokes = fig.data[6], fig.data[7]
    assert isinstance(hub, go.Scatter3d) and hub.mode == "markers"
    assert isinstance(spokes, go.Scatter3d) and spokes.mode == "lines"
    # 35 + 16 + 3 = 54 degree->=4 faces, one hub each.
    assert len(hub.x) == 54


def test_render_degrees_filter(xor_projection):
    import plotly.graph_objects as go

    # degrees=(2, 3) drops the stars (recovers the old look): 6 base traces.
    low = _render(xor_projection, degrees=(2, 3))
    assert len(low.data) == 6
    # degrees=(4, 5, 6) keeps base elements + stars, no two/three faces.
    high = _render(xor_projection, degrees=(4, 5, 6))
    assert not any(isinstance(t, go.Mesh3d) for t in high.data)  # no degree-3 mesh
    assert any(getattr(t, "mode", None) == "markers" for t in high.data)  # hubs


def test_render_higher_faces_show_toggle(xor_projection):
    fig = _render(xor_projection, show=("purviews", "two_faces", "three_faces"))
    # No higher_faces element -> no star traces.
    assert len(fig.data) == 3
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q -k "higher or degrees or stars"`
Expected: FAIL (`_higher_face_trace` undefined; `degrees=` unknown kwarg; only 6 traces).

- [ ] **Step 7: Implement the star renderer and integration**

In `pyphi/visualize/render/simplicial_complex.py`:

(a) Add `"higher_faces"` to `_ELEMENTS` (after `"three_faces"`):

```python
_ELEMENTS = (
    "purviews",
    "mechanisms",
    "cause_effect_links",
    "mechanism_purview_links",
    "two_faces",
    "three_faces",
    "higher_faces",
)
```

(b) Add the star renderer after `_three_face_trace` (after line ~401):

```python
def _higher_face_trace(faces, endpoint_pos, theme, show_colorbar=True):
    """Star expansion for degree->=4 faces: a hub at each face's centroid,
    sized/colored by φ, with spokes to each endpoint. Two merged traces."""
    hubs = []
    spokes = []
    for face in faces:
        coords = [endpoint_pos[i] for i in face.endpoints]
        hub = (
            sum(c[0] for c in coords) / len(coords),
            sum(c[1] for c in coords) / len(coords),
            sum(c[2] for c in coords) / len(coords),
        )
        hubs.append((hub, face))
        spokes.extend((hub, c) for c in coords)
    hub_trace = go.Scatter3d(
        x=[h[0] for h, _ in hubs],
        y=[h[1] for h, _ in hubs],
        z=[h[2] for h, _ in hubs],
        mode="markers",
        marker={
            "size": rescale([f.phi for _, f in hubs], *theme.node_size_range),
            "color": [f.phi for _, f in hubs],
            "colorscale": theme.face_colorscale,
            "symbol": "diamond",
            "showscale": show_colorbar,
            "colorbar": {"title": "≥4-face φ", "x": 1.38, "len": 0.6},
        },
        hovertext=[
            f"{f.degree}-face<br>overlap {f.overlap}<br>φ = {f.phi:.4g}"
            for _, f in hubs
        ],
        hoverinfo="text",
        showlegend=False,
    )
    xs, ys, zs = _segments(spokes)
    spoke_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": theme.edge_color, "width": theme.edge_width},
        hoverinfo="skip",
        showlegend=False,
    )
    return [hub_trace, spoke_trace]
```

(c) In `render_simplicial_complex`, add the `degrees` parameter and the higher-faces handling. Change the signature to add `degrees: tuple[int, ...] | None = None` (after `show_colorbars`), then after the `faces = [...]` comprehension (line ~435-439) add the degree filter and the higher split, and after the `three_faces` trace block add the star block:

```python
    faces = [
        f
        for f in projection.faces
        if all(projection.endpoints[i].distinction_id in included for i in f.endpoints)
    ]
    if degrees is not None:
        faces = [f for f in faces if f.degree in degrees]
    two_faces = [f for f in faces if f.degree == 2]
    three_faces = [f for f in faces if f.degree == 3]
    higher_faces = [f for f in faces if f.degree >= 4]
```

```python
    if "three_faces" in show and three_faces:
        traces.append(
            _three_face_trace(three_faces, endpoint_pos, theme, show_colorbars)
        )
    if "higher_faces" in show and higher_faces:
        traces.extend(
            _higher_face_trace(higher_faces, endpoint_pos, theme, show_colorbars)
        )
```

- [ ] **Step 8: Run the renderer tests to verify they pass**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q -k "higher or degrees or stars"`
Expected: PASS.

- [ ] **Step 9: Update the full-figure structure test**

The default render now has 8 traces. In `test/test_visualize_simplicial_complex.py`, update `test_render_full_figure_structure`'s count and unpacking:

```python
def test_render_full_figure_structure(xor_projection):
    import plotly.graph_objects as go

    fig = _render(xor_projection)
    # One trace per base element class, plus two for the degree->=4 star
    # expansion (hub markers + spokes).
    assert len(fig.data) == 8
    purviews, mechanisms, ce_links, mp_links, two_faces, mesh, hub, spokes = fig.data
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
    # 54 degree->=4 faces as one hub trace + one spoke trace.
    assert not isinstance(hub, go.Mesh3d)
    assert hub.mode == "markers"
    assert spokes.mode == "lines"
    assert len(hub.x) == 54
    # Endpoint labels present.
    assert "abc" in purviews.text and "c" in purviews.text
```

- [ ] **Step 10: Run the whole simplicial-complex test file**

Run: `uv run pytest test/test_visualize_simplicial_complex.py test/test_visualize_projection.py -q`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add pyphi/visualize/projection/__init__.py pyphi/visualize/render/simplicial_complex.py test/test_visualize_projection.py test/test_visualize_simplicial_complex.py
git commit -m "Render degree->=4 relation faces as star expansions

_faces carries all face degrees (previously dropped degree >=4). The
simplicial-complex view draws each degree->=4 face as a hub at its centroid
(sized/colored by phi) with spokes to each endpoint, via a default-on
higher_faces element and a new degrees= selection knob."
```

---

### Task 2: φ-by-degree spectrum view + `plot_ces` plumbing

**Files:**
- Create: `pyphi/visualize/render/spectrum.py`
- Modify: `pyphi/visualize/__init__.py` (`plot_ces`: `degrees=` param + `view="spectrum"`)
- Test: `test/test_visualize_spectrum.py` (new)

**Interfaces:**
- Consumes: `CESProjection.edges` (each a `RelationEdge` with `degree`/`phi`); `theme.face_colorscale`/`background`/`font_family`.
- Produces: `render_relation_spectrum(projection, theme, fig=None) -> go.Figure`; `plot_ces(..., view="spectrum", degrees=...)`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_visualize_spectrum.py`:

```python
from collections import Counter

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_ces

    return project_ces(examples.xor_system().ces())


def test_spectrum_aggregates_count_and_sum_phi(xor_projection):
    import plotly.graph_objects as go

    from pyphi.visualize.render.spectrum import render_relation_spectrum
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_relation_spectrum(xor_projection, DEFAULT_THEME)
    assert isinstance(fig, go.Figure)
    (bar,) = fig.data
    assert isinstance(bar, go.Bar)
    # Degrees on the x-axis, Σφ as bar height, count in customdata.
    degrees = list(bar.x)
    expected_count = Counter(e.degree for e in xor_projection.edges)
    assert degrees == sorted(expected_count)
    for degree, height, count in zip(bar.x, bar.y, bar.customdata, strict=True):
        relevant = [e.phi for e in xor_projection.edges if e.degree == degree]
        assert height == pytest.approx(sum(relevant))
        assert count[0] == len(relevant)


def test_plot_ces_spectrum_view_runs():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_ces

    fig = plot_ces(examples.xor_system().ces(), view="spectrum")
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Bar) for t in fig.data)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest test/test_visualize_spectrum.py -q`
Expected: FAIL (`spectrum` module missing; `view="spectrum"` unknown).

- [ ] **Step 3: Create the spectrum renderer**

Create `pyphi/visualize/render/spectrum.py`:

```python
"""φ-by-degree relation spectrum panel."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from pyphi.visualize.projection import CESProjection
from pyphi.visualize.theme import Theme


def render_relation_spectrum(
    projection: CESProjection, theme: Theme, fig: go.Figure | None = None
) -> go.Figure:
    """A 2-D bar panel of relation count and Σφ per relation degree.

    Computed from the projection's relation-level ``edges`` (which carry every
    relation's degree and φ), so the high-degree structure that is hard to read
    in the 3-D simplicial-complex view is summarized at a glance.
    """
    count: dict[int, int] = defaultdict(int)
    sum_phi: dict[int, float] = defaultdict(float)
    for edge in projection.edges:
        count[edge.degree] += 1
        sum_phi[edge.degree] += edge.phi
    degrees = sorted(count)
    figure = go.Figure() if fig is None else fig
    figure.add_trace(
        go.Bar(
            x=degrees,
            y=[sum_phi[d] for d in degrees],
            customdata=[[count[d]] for d in degrees],
            marker={
                "color": [sum_phi[d] for d in degrees],
                "colorscale": theme.face_colorscale,
            },
            hovertemplate=(
                "degree %{x}<br>Σφ = %{y:.4g}<br>count = %{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        xaxis={"title": "relation degree", "dtick": 1},
        yaxis={"title": "Σφ"},
        paper_bgcolor=theme.background,
        font={"family": theme.font_family},
        showlegend=False,
    )
    return figure
```

- [ ] **Step 4: Wire `view="spectrum"` and `degrees=` into `plot_ces`**

In `pyphi/visualize/__init__.py`, add a `degrees=None` keyword to `plot_ces` (after `show=None`), document it in the docstring (a one-line `degrees` entry and a `"spectrum"` view entry), plumb it into the simplicial-complex branch, and add the spectrum branch before the final `raise`:

```python
    if view == "simplicial_complex":
        from .render.simplicial_complex import render_simplicial_complex

        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if show is not None:
            kwargs["show"] = show
        if degrees is not None:
            kwargs["degrees"] = degrees
        return render_simplicial_complex(
            projection, theme, fig=fig, layout=layout, **kwargs
        )
```

```python
    if view == "spectrum":
        from .render.spectrum import render_relation_spectrum

        return render_relation_spectrum(projection, theme, fig=fig)
    raise ValueError(f"unknown view {view!r}")
```

Docstring additions (in the `view` Keyword Arg block and a new `degrees` Keyword Arg):

```python
            ``"spectrum"``: a 2-D bar panel of relation count and Σφ per
            relation degree, summarizing the high-degree structure.
```
```python
        degrees (tuple[int, ...]): Restrict the simplicial-complex view to
            relation faces of these degrees. Defaults to all degrees present.
```

- [ ] **Step 5: Run the spectrum tests to verify they pass**

Run: `uv run pytest test/test_visualize_spectrum.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/visualize/render/spectrum.py pyphi/visualize/__init__.py test/test_visualize_spectrum.py
git commit -m "Add phi-by-degree relation spectrum view + degrees= plumbing

plot_ces(view='spectrum') renders a per-degree bar panel (count + Σφ) from the
projection's relation-level edges; plot_ces gains a degrees= knob plumbed into
the simplicial-complex view."
```

---

### Task 3: Changelog, roadmap, and full verification

**Files:**
- Create: `changelog.d/higher-degree-relation-viz.feature.md`
- Modify: `ROADMAP.md` (P14d A-5 dashboard row; Wave 2 archive note; Landed prose line)

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/higher-degree-relation-viz.feature.md`:

```markdown
Relation faces of degree ≥4 are now visible in the simplicial-complex view
(`pyphi.visualize.plot_ces`), drawn as a star expansion — a hub at each face's
centroid, sized and colored by φ, with spokes to its endpoints. Previously these
faces were silently dropped. A `higher_faces` show element (on by default) and a
`degrees=` selection knob control which face degrees render, and a new
`view="spectrum"` panel summarizes relation count and Σφ per degree.
```

- [ ] **Step 2: Update the ROADMAP dashboard row**

In `ROADMAP.md`, change the `P14d A-5` row status from `⬜ open` to `✅ landed` and update its one-line:

```markdown
| P14d A-5 | ✅ landed | 2 | Degree-≥4 relation faces are drawn as star expansions (hub at the face centroid, sized/colored by φ, spokes to each endpoint) in the simplicial-complex view — previously dropped. Projection carries all face degrees; `higher_faces` show element (default-on) + `degrees=` knob; new `view="spectrum"` φ-by-degree panel (count + Σφ). Convex-hull blobs deferred; skeleton expansion rejected. |
```

- [ ] **Step 3: Update the Wave 2 archive note and the Landed prose line**

In `ROADMAP.md`, find the Wave 2 archive bullet `**P14d — \`to_pandas\` consolidation — landed (2026-06-15); A-5 viz remains.**` and update its closing sentence (currently "A-5 ... is lower priority and remains open.") to past tense: A-5 landed (2026-06-18) — degree-≥4 faces drawn as star expansions (hub + spokes per face, sized/colored by φ), projection carrying all degrees, a default-on `higher_faces` element + `degrees=` knob, and a `view="spectrum"` φ-by-degree panel; convex-hull blobs deferred, skeleton expansion rejected. In the `### ✅ Landed` prose line near the top, append `· P14d A-5 (higher-degree relation viz)`.

- [ ] **Step 4: Run the full verification gate**

Run: `uv sync --all-extras` then `uv run pytest`
Expected: PASS with **no path argument** (the visualize tests need the extra; the full run collects `pyphi/` + `test/` doctests).

- [ ] **Step 5: Commit**

```bash
git add changelog.d/higher-degree-relation-viz.feature.md ROADMAP.md
git commit -m "Mark P14d A-5 landed: higher-degree relation viz; changelog + roadmap"
```

---

## Self-Review

**Spec coverage:**
- Projection carries all degrees (spec §4.1) → Task 1 Step 3.
- Star-expansion renderer + `higher_faces` + `degrees=` (spec §4.2) → Task 1 Step 7; `plot_ces` `degrees=` plumbing → Task 2 Step 4.
- φ-by-degree spectrum (spec §4.3) → Task 2 Step 3-4.
- Behavior change documented (spec §4.4) → Task 3 Step 1.
- Testing (spec §5): projection degrees, star renderer (centroid), show/degrees controls, spectrum aggregation, smoke → Tasks 1-2.
- Roadmap + changelog (spec §7) → Task 3.

**Placeholder scan:** none — every code step shows complete code. Task 3 Step 3 is a localized prose edit to an existing ROADMAP bullet.

**Type consistency:** `_higher_face_trace` returns `list[go.Scatter3d]` and is spread into `traces` via `extend`. `render_relation_spectrum` returns `go.Figure`. `degrees` is `tuple[int, ...] | None` in both `render_simplicial_complex` and `plot_ces`. The spectrum consumes `projection.edges` (`RelationEdge.degree`/`.phi`), matching the projection produced by `project_ces`. The renderer tests rely on the trace order purviews, mechanisms, ce_links, mp_links, two_faces, three_faces(mesh), hub, spokes — the append order in `render_simplicial_complex`.
