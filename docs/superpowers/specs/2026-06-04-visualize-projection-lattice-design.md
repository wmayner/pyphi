# Visualization Projection Foundation + Lattice Renderer — Design

Date: 2026-06-04

## Context

`pyphi/visualize/` was cobbled together: an 895-line `ces/__init__.py` tangles
four concerns (extracting data from IIT result objects, computing 3-D geometry,
theming, emitting plotly traces) and reaches directly into
`PhiStructure`/`Distinction`/`Relation` internals, with zero tests. This is
**P14d sub-project A** (visualize refresh). We are not preserving the current
behavior; we are rebuilding around a clean seam.

**Agreed architecture (validated visually with the maintainer):** a pure
**projection** layer turns IIT results into plot-ready data; thin **render**
backends turn that data into figures; a consistent public API composes them
with a single `Theme`. The φ-structure figure gets **four** renderers over one
projection — *evocative* 3-D simplicial complex (today's style, cleaned up) plus
three *technical* views grounded in Haun & Tononi 2019 (Entropy 21:1160):
relational-role **scatter** (Figs 7–8), inclusion **lattice/Hasse** (Fig 9
region/location partial order), and relation **matrix/heatmap**.

This spec covers **sub-project 1 only**: the projection foundation + the
**lattice** renderer (the maintainer's headline technical view). Subsequent
sub-projects add the 3-D, scatter, and matrix renderers, then migrate the
simpler plots (connectivity/distribution/dynamics/ising) onto the same pattern.

## Goal

Establish the pure projection data model (designed to serve all four φ-structure
renderers) + a `Theme` dataclass + the public-API shape, and prove the seam
end-to-end with the lattice renderer.

## Non-goals

- The evocative 3-D, scatter, and matrix renderers (later sub-projects). The
  projection is designed *with them in mind* but only the lattice consumes it
  now; adding the others must be purely additive.
- Migrating connectivity/distribution/dynamics/ising (later sub-project).
- Reproducing the exact current figure (behavior preservation is explicitly
  off the table).
- The holistic peripheral-API namespace regroup (separate pre-P15 pass).

## Architecture

```
pyphi/visualize/
  projection/      # NEW — pure model -> plot-data. No plotting deps. Unit-tested.
  render/          # NEW — plot-data -> figure (plotly). Smoke-tested.
  theme.py         # Theme dataclass (replaces ad-hoc **theme_overrides)
  __init__.py      # public API: plot_phi_structure(..., view=..., theme=...)
  ces/, distribution.py, connectivity.py, dynamics.py, ising.py  # untouched this sub-project
```

The public entry point gains a `view` selector (this sub-project ships
`view="lattice"`; later sub-projects add `"evocative"`, `"scatter"`, `"matrix"`).

## Components

### A. Projection data model (`pyphi/visualize/projection/`)

Pure dataclasses + a builder. **Imports only stdlib + numpy** — never plotly /
matplotlib / `pyphi.models.fmt`. Built from a `CauseEffectStructure` (which
carries `.distinctions` and `.relations`) plus node labels.

```
@dataclass(frozen=True)
class DistinctionNode:
    id: int                       # stable index within the projection
    mechanism: tuple[int, ...]
    label: str                    # distinction.mechanism_label, e.g. "CDE"
    cause_purview: tuple[int, ...]
    effect_purview: tuple[int, ...]
    mechanism_state: tuple[int, ...]
    phi: float                    # distinction.phi
    sum_phi_relations: float      # Σφ_R over relations involving this distinction
    # role flags for the scatter renderer (computed now, unused by lattice):
    includes: bool                # up-includes another distinction
    included: bool                # down-included by another

@dataclass(frozen=True)
class RelationEdge:
    relata: tuple[int, ...]       # DistinctionNode ids
    degree: int                   # number of relata (2-relation, 3-relation, …)
    phi: float                    # relation.phi
    overlap: tuple[int, ...]      # relation.purview (the shared units)

@dataclass(frozen=True)
class InclusionOrder:
    # the partial order the lattice lays out: covers[a] = distinctions a directly
    # down-includes; rank[a] = a level monotonic in the order (for vertical layout)
    covers: Mapping[int, tuple[int, ...]]
    rank: Mapping[int, int]

@dataclass(frozen=True)
class PhiStructureProjection:
    nodes: tuple[DistinctionNode, ...]
    edges: tuple[RelationEdge, ...]
    inclusion: InclusionOrder
    node_labels: NodeLabels

def project_phi_structure(ces, node_labels) -> PhiStructureProjection: ...
```

Construction notes:
- `sum_phi_relations[d]` = sum of `relation.phi` over `edges` whose `relata`
  include `d`.
- Inclusion is derived from the existing `Distinctions.purview_inclusion(...)`
  helper (`pyphi/models/distinctions.py`) — the same machinery relations
  computation already uses — not re-implemented. `covers` is the transitive
  reduction; `rank` is the length of the longest down-chain (so the "whole"
  distinction is at the top, single-unit "points" at the bottom).
- The builder is the **only** code that touches `Distinction`/`Relation`
  internals.

### B. `Theme` (`pyphi/visualize/theme.py`)

A frozen dataclass collecting the knobs the renderers read (colorscale for φ,
node-size range, edge opacity, font, background, role colors). Replaces the
ad-hoc `**theme_overrides` mapping. A module-level `DEFAULT_THEME`.

### C. Public API (`pyphi/visualize/__init__.py`)

```
def plot_phi_structure(ces, *, view="lattice", theme=DEFAULT_THEME,
                       node_labels=None, fig=None): ...
```
Composes `project_phi_structure(...)` → the renderer selected by `view`. This
sub-project implements `view="lattice"`; other values raise `NotImplementedError`
with a message listing the views shipping in later sub-projects.

### D. Lattice renderer (`pyphi/visualize/render/lattice.py`)

Consumes `PhiStructureProjection` → a plotly 2-D Hasse figure:
- Nodes positioned by `inclusion.rank` (y = rank, x spread within a rank),
  drawn as markers sized by `sum_phi_relations` and colored by `phi` via the
  theme colorscale; hover text shows label, mechanism→cause/effect purviews,
  φ, Σφ_R.
- Inclusion `covers` drawn as edges between ranks.
- Returns the plotly `Figure`. (Interactive subtext/supertext highlighting is a
  nice-to-have deferred; the static lattice is the deliverable.)

Imports plotly; knows nothing about IIT (consumes only projection dataclasses).

## Testing

- **Projection (exact-value unit tests) — the coverage heart.** On a small
  `CauseEffectStructure` (a committed fixture or a fast computed example), assert
  exact `DistinctionNode` fields (labels, purviews, φ), exact `sum_phi_relations`
  per node (hand-summed from the relations), and the exact `InclusionOrder`
  (`covers`/`rank`) against a hand-derived partial order. No plotting deps.
- **Lattice renderer (figure-structure smoke test).** Assert the returned figure
  has one marker trace with N points (N = #distinctions), an edge trace with the
  expected segment count (= #cover edges), and y-coordinates equal to the ranks.
  Assert structure, not pixels.
- **Whole-suite + lint.** `uv run pytest` (no path arg), `uv run pyright pyphi`,
  `uv run ruff check`. No golden impact (visualize feeds no φ).

Fixture note: φ-structure JSON fixtures in `test/data/phi_structure/` are
version-stamped (`2.0.0a1`) and fail the loader's version guard under the
setuptools-scm dev version; the projection tests should compute a small CES
directly (e.g. a 2–3 node example) or use a fixture that loads, rather than
depend on the version-stamped JSON.

## Risks

- **Projection mis-modeled for later renderers** → rework when adding
  scatter/matrix/3-D. Mitigation: the data model above was derived from all four
  renderers' needs (positions/size/color/role/inclusion/relations); the role
  flags and relation edges are included now though only the lattice uses them.
- **Inclusion order semantics.** `rank` must be monotonic in the partial order
  or edges cross levels confusingly. Mitigation: rank = longest down-chain
  length; unit-tested against a hand-derived order.
- Low overall: no φ/golden impact; optional plotting code.

## Success criteria

- `pyphi/visualize/projection/` builds a `PhiStructureProjection` from a
  `CauseEffectStructure`, is pure (no plotting imports), and is exact-value
  unit-tested.
- `plot_phi_structure(ces, view="lattice")` returns a plotly Hasse figure;
  figure-structure smoke test passes.
- `Theme` dataclass + `DEFAULT_THEME` exist; the public entry point composes
  projection → lattice render.
- Whole suite + doctests green; pyright/ruff clean.
