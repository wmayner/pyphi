# Evocative 3-D Renderer Rebuild — Design

Date: 2026-06-04

## Context

Sub-project 2 of the visualize refresh. Sub-project 1 established the
projection-core architecture (pure `projection/` → `render/` backends → public
API with `view=` dispatch and a frozen `Theme`) and shipped the lattice view.
This sub-project rebuilds the *evocative* 3-D simplicial-complex view — the
signature phi-structure figure — onto that seam, then deletes the legacy
implementation (`pyphi/visualize/ces/`, ~895 lines plus its private
theme/colors/geometry/text/utils modules, all self-contained with no external
references).

The legacy view draws at finer granularity than the lattice: **two points per
distinction** (the cause and effect purviews, i.e. the MICE sides) and
relation **faces** (degree-2 faces as lines, degree-3 faces as triangles;
higher degrees are not drawable in 3-D and are skipped). The current
projection carries neither endpoints nor faces, so the projection grows —
purely additively, as the parent spec requires.

Decisions taken with the maintainer:
- Behavior preservation is explicitly off the table; visual design may change.
- `highlight_phi_fold` is ported onto the new renderer (not deferred), and
  `pyphi/visualize/ces/` is deleted at the end of this sub-project.
- Semantic knobs are call-site parameters; `Theme` stays purely stylistic
  ("Theme never changes what a figure says, only how it looks").

## Goal

`plot_phi_structure(ces, view="evocative")` produces the 3-D
simplicial-complex figure from the shared projection; `highlight_phi_fold`
works on the new path; the legacy module is gone.

## Non-goals

- Scatter and matrix renderers (sub-project 3).
- Migrating connectivity/distribution/dynamics/ising (sub-project 4).
- Rendering faces of degree ≥ 4 (not drawable in 3-D; same convention as
  legacy).
- Reproducing legacy output pixel-for-pixel, including its per-face
  trace mode (see Simplifications).

## Architecture

### A. Projection additions (`pyphi/visualize/projection/`)

```
@dataclass(frozen=True)
class EndpointNode:
    id: int                      # 2 * distinction_id + (0 cause, 1 effect)
    distinction_id: int
    direction: str               # "cause" | "effect"
    purview: tuple[int, ...]
    purview_state: tuple[int, ...]
    phi: float                   # the MICE's phi
    label: str                   # state-cased purview label (ON upper, OFF lower)

@dataclass(frozen=True)
class RelationFaceEdge:
    endpoints: tuple[int, ...]   # EndpointNode ids
    degree: int                  # number of relata (2 or 3)
    phi: float
    overlap: tuple[int, ...]     # the face's shared units
```

`PhiStructureProjection` gains two fields: `endpoints:
tuple[EndpointNode, ...]` (length 2 × #distinctions, interleaved
cause/effect) and `faces: tuple[RelationFaceEdge, ...]` (degree-2 and
degree-3 faces only, from `ces.relations.faces_by_degree`; higher degrees are
not projected since no renderer consumes them).

Construction notes:
- Endpoints come from `distinction.cause` / `distinction.effect`
  (MaximallyIrreducibleCause/Effect): `.purview`, `.purview_state`, `.phi`.
- Face relata are MICE objects; they map to endpoint ids via
  `(tuple(mice.mechanism), direction)` — mechanism uniquely keys a
  distinction, direction picks the side.
- The state-cased label is computed in the projection from `node_labels`,
  the purview, and the purview state (uppercase = unit ON, lowercase = OFF).
  The projection remains the only model-coupling point; it still imports no
  plotting libraries.

### B. Evocative renderer (`pyphi/visualize/render/evocative.py`)

Pure plot-space geometry plus plotly traces; consumes only projection
dataclasses and `Theme`.

**Geometry** (module-private functions): the legacy cylindrical-shell
arrangement, cleaned up. Subsets of each size k occupy a shell of radius
`radius(k)` arranged on a regular polygon; cause/effect endpoint clouds are
offset ±x; small regular-polygon jitter separates endpoints that share a
purview. Deterministic — same projection in, same coordinates out (this
determinism is what `highlight_phi_fold` relies on for alignment).

Knobs live in a frozen dataclass passed as one parameter:

```
@dataclass(frozen=True)
class EvocativeGeometry:
    max_radius: float = 1.0
    z_spacing: float = 0.0        # 0 = flat; >0 stacks shells in z
    direction_offset: float = 0.5 # ±x separation of cause/effect clouds
    purview_jitter: float = 0.1   # radius for endpoints sharing a purview
```

**Traces** (one merged trace per element class — see Simplifications):

| Element | Trace | Driven by |
|---|---|---|
| Purview endpoints | Scatter3d markers+text | size/color = endpoint φ; text = state-cased label, colored by direction |
| Mechanism labels | Scatter3d text | label = distinction label; position = mechanism shell |
| Cause→effect links | Scatter3d lines | one segment per distinction, color by direction gradient endpoints |
| Mechanism–purview links | Scatter3d lines | cause endpoint → mechanism → effect endpoint per distinction |
| 2-face lines | Scatter3d lines | per-vertex color = face φ through theme colorscale; constant width |
| 3-face mesh | single Mesh3d | per-vertex intensity = face φ; constant opacity |

Signature:

```
def render_evocative(projection, theme, fig=None,
                     geometry=EvocativeGeometry(),
                     show=("purviews", "mechanisms", "cause_effect_links",
                           "mechanism_purview_links", "two_faces",
                           "three_faces"),
                     only_distinctions=None) -> go.Figure
```

`show` selects element classes (semantic → parameter). `only_distinctions`
is a set of distinction ids restricting what is drawn *without changing the
geometry* (computed from the full projection) — the primitive
`highlight_phi_fold` composes on. Unknown `show` entries raise `ValueError`.

Hover: endpoints show label, mechanism, direction, purview, state, φ; faces
show degree, overlap, φ (per-vertex hovertext on the merged traces).

### C. Theme additions (`pyphi/visualize/theme.py`)

New stylistic fields with defaults matching the legacy palette:

```
cause_color: str = "#8D3D00"      # brown
effect_color: str = "#006146"     # teal
face_colorscale: str = "Blues"
face_opacity: float = 0.2
text_size: int = 12
```

`highlight_phi_fold` derives its dimmed background theme via
`dataclasses.replace` (greys, low opacity) — no separate theme class.

### D. Public API (`pyphi/visualize/__init__.py`)

- `view="evocative"` dispatches to `render_evocative`;
  `_VIEWS_PENDING` drops the entry. Evocative-specific parameters
  (`geometry`, `show`) are accepted by `plot_phi_structure` and documented
  as applying to the evocative view.
- `highlight_phi_fold(ces_, phi_fold, *, theme=DEFAULT_THEME, ...)`
  reimplemented: project the full structure once, render it with the dimmed
  derived theme, then render again with `only_distinctions` = the fold's
  distinctions (matched by mechanism) into the same figure. Shared
  deterministic geometry guarantees alignment.
- Delete `pyphi/visualize/ces/` entirely; remove `from . import ces` and the
  legacy re-exports; update `__all__`.

## Simplifications relative to legacy

Recorded deliberately; all accepted by "behavior preservation off the table":

- **No per-face trace mode.** Legacy forked at `detail_threshold=100`
  between one-trace-per-face (slow, per-face width/opacity) and merged
  traces (averaged width, warning). The rebuild always merges: per-vertex
  color/intensity carries φ; width and opacity are constant from `Theme`.
- **No type-based 2-face coloring** (isotext/inclusion/paratext via legacy
  `colors.type_color`) — φ-colored only. Can return later as a `color_by`
  channel if wanted.
- **One purview arrangement** (shells). The legacy alternative
  (`arrange_by_mechanism`) is dropped; `EvocativeGeometry` can grow a
  placement mode later if needed.
- **No rotation/translation transform stack** — `EvocativeGeometry`'s four
  knobs replace the legacy `Coordinates` composition machinery.

## Testing

- **Projection (exact-value) — the coverage heart.** On the xor CES:
  8 endpoints with interleaved ids, per-side purviews/φ/states asserted
  against the known values; face counts equal
  `len(faces_by_degree[2])`/`[3]`; every face's endpoint ids valid and
  consistent with its relata's mechanisms/directions; state-cased labels
  exact.
- **Renderer (figure-structure).** Trace count = number of shown element
  classes (Mesh3d present only when 3-faces exist and are shown); merged
  2-face trace has 3 × n_faces₂ points; endpoint marker count = 2 ×
  #distinctions; `show` subsetting drops traces; unknown `show` raises;
  `only_distinctions` reduces point counts without moving shared
  coordinates.
- **highlight_phi_fold (smoke).** Highlighting a 2-distinction subset of the
  xor CES yields a figure with both passes' traces; the overlay's endpoint
  coordinates are a subset of the background's.
- **Deletion.** No references to `pyphi.visualize.ces` remain; `uv run
  pytest` (no path) green; pyright/ruff clean.

## Risks

- **Aesthetic regression** — the rebuilt figure will not look identical;
  mitigated by rendering legacy-vs-new comparisons to the visual companion
  during implementation and iterating with the maintainer.
- **Merged-trace hover fidelity** — per-vertex hovertext is coarser than
  legacy per-face traces; accepted, recorded above.
- **MICE attribute coupling** — confined to the projection builder, as
  required by the architecture.

## Success criteria

- `plot_phi_structure(ces, view="evocative")` returns the 3-D figure;
  `view="lattice"` unaffected.
- `highlight_phi_fold` works on the new path with aligned overlay.
- `pyphi/visualize/ces/` deleted; no dangling imports.
- Projection additions exact-value tested; whole suite + doctests green;
  pyright/ruff clean.
