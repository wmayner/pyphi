# P14d A-5 — Higher-degree relation visualization: Design

**Status:** approved
**Date:** 2026-06-18
**Wave:** 2 (pre-freeze, surface-affecting — the last open Wave 2 item)
**Part of:** P14d (`to_pandas` consolidation landed 2026-06-15; this is the
remaining A-5 visualization tail).

---

## 1. Motivation

In the simplicial-complex view (`pyphi/visualize/`), relation faces of degree ≥4
are currently invisible. The projection layer drops them before rendering:
`_faces()` (`pyphi/visualize/projection/__init__.py:202`) iterates only
`for degree in (2, 3)`, so every higher face is silently discarded. Degree-2
faces render as line segments and degree-3 as a `Mesh3d` triangle; nothing
represents the rest. This is not rare — `xor` already has 54 faces of degree
4–6 (35/16/3), `grid3` has faces up to degree 8, and `rule154` has hundreds. The
plotted structure therefore understates the relational structure it claims to
show.

A k-simplex with k > 3 has no canonical faithful 3-D drawing. The honest,
degree-agnostic representation is a **star expansion**: a hub marker at the
face's centroid with spokes to each endpoint, the hub sized and colored by φ.
This makes every face visible at any degree with O(1) plot traces. A
complementary **φ-by-degree spectrum** panel conveys the aggregate high-degree
structure that is hard to read in the 3-D plot.

## 2. Goals

- Make degree-≥4 relation faces visible in the simplicial-complex view via a star
  expansion (hub at centroid, spokes to endpoints, hub sized/colored by φ).
- Carry all face degrees through the projection (stop dropping degree ≥4).
- Integrate via the existing `show=` vocabulary (a `higher_faces` element,
  default-on) and a new `degrees=` selection knob.
- A φ-by-degree spectrum panel (count and Σφ of relations per degree) as a new
  `plot_ces` view, computed from the projection's relation-level data.

## 3. Non-goals

- **Convex-hull "blob" rendering** of high-degree faces (the evocative
  translucent-hull option). Deferred — more failure modes (degenerate/coplanar
  hulls), needs its own φ-threshold/top-N filter, and is "evocative" rather than
  "faithful." The star expansion is the faithful default this item delivers.
- **Skeleton expansion** (drawing a k-relation as many 3-relations). Rejected in
  the roadmap — it conflates one high-degree relation with several low-degree
  ones.
- **Decoupling `visualize/ces/` from model internals.** A separate, larger
  engineering pass noted in the roadmap; out of scope here. This item works only
  in the already-decoupled `projection/` + `render/` layer.
- Any change to the relation/CES formalism or to how faces are computed.

## 4. Design

### 4.1 Projection carries all degrees

`pyphi/visualize/projection/__init__.py`, `_faces()`: replace the
`for degree in (2, 3)` restriction with iteration over every degree present in
`relations.faces_by_degree`. `RelationFaceEdge` already carries `endpoints`,
`degree`, `phi`, `overlap` and needs no change beyond its docstring (drop the
"degree-2 or degree-3" wording). The faces remain sorted by
`(degree, endpoints, phi)` for determinism.

### 4.2 Star expansion renderer

`pyphi/visualize/render/simplicial_complex.py`, new
`_higher_face_trace(faces, endpoint_pos, theme, show_colorbar=True) -> list`:

- For each face, compute the hub position as the centroid (mean) of
  `endpoint_pos[i]` over the face's endpoints.
- **Hub trace** — one `go.Scatter3d(mode="markers")` over all hubs, marker size
  and color driven by φ (color via `theme.face_colorscale`); hovertext gives the
  degree, overlap, and φ.
- **Spoke trace** — one `go.Scatter3d(mode="lines")` over all spokes, built with
  the existing `_segments` helper from `(hub, endpoint)` pairs (None-separated
  polylines), colored by the face φ.

Two merged traces total, independent of the number of faces. The function
returns the list of traces it produced.

`_ELEMENTS` gains `"higher_faces"` (default-on). `render_simplicial_complex`:
- New parameter `degrees: tuple[int, ...] | None = None`. When given, faces are
  filtered to those degrees before the per-class split; `None` keeps all.
- Split out `higher_faces = [f for f in faces if f.degree >= 4]` and, when
  `"higher_faces" in show` and any exist, extend `traces` with
  `_higher_face_trace(...)`.
- `two_faces` / `three_faces` rendering is unchanged (they keep their line/mesh
  representations; the star expansion applies only to degree ≥4).

`plot_ces` (`pyphi/visualize/__init__.py`) plumbs `degrees=` through to
`render_simplicial_complex` for the simplicial-complex view.

The `degrees=` knob and the `higher_faces` show element compose: `show=` toggles
the star class on/off; `degrees=` restricts which degrees draw across all face
classes (e.g. `degrees=(2, 3)` recovers the previous look; `degrees=(4, 5, 6)`
isolates the high-degree structure).

### 4.3 φ-by-degree spectrum panel

`pyphi/visualize/render/spectrum.py` (new),
`render_relation_spectrum(projection, theme, fig=None) -> go.Figure`:

- Aggregate `projection.edges` (each a `RelationEdge` with `degree` and `phi`)
  into per-degree relation **count** and **Σφ**.
- Render a 2-D `go.Bar` panel with degree on the x-axis: Σφ as the bar height,
  with the count available (second series or hovertext). Uses the theme palette.

Exposed as `view="spectrum"` in `plot_ces`, dispatched the same way as the
existing `lattice` / `scatter` / `matrix` views. It consumes only the projection
(no model internals), consistent with the decoupled `render/` layer.

### 4.4 Behavior change

`plot_ces(view="simplicial_complex")` now renders degree-≥4 faces as stars by
default (`higher_faces` is in the default `_ELEMENTS`). Previously these faces
were absent. This is the intended correction; it is recorded in the changelog.

## 5. Testing

`test/test_visualize_projection.py` and `test/test_visualize_simplicial_complex.py`
(extend; both already use the `xor` fixture, which has degree 4/5/6 faces), plus
a new `test/test_visualize_spectrum.py`:

- **Projection carries all degrees:** `project_ces(xor).faces` includes faces of
  degree 4, 5, and 6 with the expected per-degree counts (35/16/3); existing
  `test_project_xor_faces` is updated from its degree-2/3-only assertion.
- **Star renderer:** `_higher_face_trace` over the xor degree-≥4 faces returns a
  hub trace and a spoke trace; the hub count equals the number of degree-≥4
  faces; each hub sits at the centroid of its endpoints (hand-checked on one
  face).
- **`show` / `degrees` controls:** `render_simplicial_complex(..., show=(..., "higher_faces"))`
  adds the star traces; omitting `"higher_faces"` omits them; `degrees=(2, 3)`
  yields no star traces; `degrees=(4, 5, 6)` yields the star traces and no
  two/three-face traces. `test_render_full_figure_structure`'s trace-count
  assertion is updated for the default-on stars.
- **Unknown-degree safety:** passing a `degrees=` containing a degree with no
  faces is a no-op (no trace, no error).
- **Spectrum:** `render_relation_spectrum(project_ces(xor), theme)` produces a
  figure whose per-degree counts and Σφ match the relation degrees /
  `faces_by_degree` of the xor CES (cross-checked); `plot_ces(xor, view="spectrum")`
  runs.
- **Smoke:** `plot_ces(xor, view="simplicial_complex")` runs with the new default
  (stars shown) and with `degrees=`/`show=` variations.

Verification runs `uv run pytest` **with no path argument**. The visualize
package requires the `visualize` extra; the suite is run under
`uv sync --all-extras` (matplotlib/plotly/seaborn present), as the existing
visualize tests already assume.

## 6. Risks and mitigations

- **Visual density ("soup") for high-degree-rich structures** (rule154 has
  hundreds of high-degree faces). Mitigated by the `degrees=` knob (restrict to
  chosen degrees) and the lightweight star (hub+spokes, O(1) traces) rather than
  filled hulls; the spectrum panel gives the aggregate without the 3-D clutter.
  Default-on is chosen because faithful visibility is the goal; users tune via
  `degrees=` / `show=`.
- **Behavior change to the default figure.** Intended and documented; the two
  tests that pinned the old behavior are updated, not worked around.
- **Centroid degeneracy.** The centroid of distinct endpoint positions is always
  well-defined; spokes are straight segments, no hull/coplanarity math (the
  reason blobs are deferred).
- **Optional-dependency coupling.** The new render module imports plotly the same
  way the existing render modules do (the package already raises
  `MissingOptionalDependenciesError` at import if the `visualize` extra is
  absent); no new dependency.

## 7. Acceptance criteria

- `project_ces` carries relation faces of all degrees (degree ≥4 no longer
  dropped).
- The simplicial-complex view renders degree-≥4 faces as star expansions
  (hub + spokes, sized/colored by φ), default-on via a `higher_faces` show
  element, with a `degrees=` selection knob.
- A `view="spectrum"` φ-by-degree panel (count + Σφ per degree) renders from the
  projection.
- Updated tests pass (projection degrees, star traces, show/degrees controls,
  spectrum aggregation, smoke); `uv run pytest` (no path argument) green.
- ROADMAP P14d A-5 row updated to landed; changelog fragment present (noting the
  default-figure change).
