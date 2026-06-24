# Scatter and Matrix Renderers — Design

Date: 2026-06-04

## Context

Sub-project 3 of the visualize refresh: the two remaining φ-structure views,
both technical 2-D figures grounded in Haun & Tononi 2019 (Entropy 21:1160).

- **Scatter** (Figs 7–8): distinctions as markers on a 2-D embedding of
  their unit composition, sized by Σφ_R, colored by relational role
  (extends up / down / both), circle vs diamond for connected vs not.
- **Matrix**: distinctions × distinctions heatmap of shared relation
  strength (not a figure in the paper; the natural tabular complement).

The projection layer needs **no changes**: the scatter consumes the
`includes`/`included` role flags and `sum_phi_relations` (added in
sub-project 1 for this purpose), positions derive from
`cause_purview ∪ effect_purview`, and the matrix aggregates `edges`.
Both renderers are pure consumers of projection dataclasses.

Decision (maintainer): positions use **PCA of composition** — a
deterministic, numpy-only stand-in for the paper's t-SNE (similar unit
composition → nearby), avoiding a scikit-learn dependency and
nondeterminism.

## Goal

`plot_phi_structure(ces, view="scatter")` and `view="matrix"` work;
`_VIEWS_PENDING` is empty and removed; all four views ship.

## Non-goals

- Focal-distinction variants of Fig 7 (roles relative to one chosen
  distinction); the structure-wide Fig 8 roles ship now. A `focus=`
  parameter can come later if wanted.
- t-SNE or other embeddings (the PCA seam makes adding them later easy).
- Sub-project 4 (connectivity/distribution/dynamics/ising migration).

## Components

### A. Scatter renderer (`pyphi/visualize/render/scatter.py`)

```
def render_scatter(projection, theme, fig=None,
                   size_by="sum_phi_relations",
                   color_by="role") -> go.Figure
```

- **Position**: PCA over each distinction's purview-union membership vector
  (n_distinctions × n_units binary matrix, column-centered, numpy SVD;
  coordinates = first two principal components). Deterministic sign
  convention: each component is flipped so its largest-magnitude loading is
  positive. Degenerate cases (fewer than 3 distinctions, or zero variance in
  a component) fall back to spreading the affected coordinate evenly by node
  id, so points never silently coincide.
- **Size**: same channel vocabulary as the lattice — `"sum_phi_relations"`
  (default, Fig 8's Σφ_R), `"phi"`, or `None` for uniform.
- **Color**: `color_by="role"` (default) colors by extendedness category
  derived from the role flags — `includes and included → "extended"`,
  `includes only → "includes"`, `included only → "included"`, neither →
  `"none"` — via `theme.role_colors`. `"phi"` / `"sum_phi_relations"` give
  continuous coloring through `theme.colorscale` instead.
- **Symbol**: circle if the distinction participates in any relation with
  another distinction, open diamond otherwise (Fig 8's connected marker
  convention).
- **Text/hover**: mechanism label on the marker; hover shows label,
  mechanism, cause/effect purviews, φ, Σφ_R, and role.
- Unknown `size_by`/`color_by` raise `ValueError`.

### B. Matrix renderer (`pyphi/visualize/render/matrix.py`)

```
def render_matrix(projection, theme, fig=None) -> go.Figure
```

- One heatmap trace, n × n over distinctions:
  - **Off-diagonal** `(i, j)`: Σ `edge.phi` over edges whose relata include
    both `i` and `j` — total strength of the relations binding the pair.
  - **Diagonal** `(i, i)`: Σ `edge.phi` over `i`'s self-relations (edges
    whose relata are exactly `{i}`) — the reflexivity strength.
- Rows/columns ordered by (mechanism size, label), so mechanism orders form
  visible blocks; axis tick labels are the distinction labels.
- Colorscale from `theme.colorscale`; hover shows the pair's labels and the
  cell value.

### C. Theme

`role_colors` gains a fourth entry for the no-role category:
`("none", "#b0b0b0")`. No other changes.

### D. Public API (`pyphi/visualize/__init__.py`)

- `view="scatter"` → `render_scatter`; `view="matrix"` → `render_matrix`.
  `_VIEWS_PENDING` and its `NotImplementedError` branch are removed —
  unknown views just raise `ValueError`.
- `color_by`'s default becomes `None`, meaning *the view's default*
  (lattice → `"phi"`, scatter → `"role"`); an explicit value is passed
  through. (`size_by` keeps its explicit default since `None` already means
  "uniform markers".) The docstring documents per-view domains.
- `size_by`/`color_by` are forwarded to the scatter view; the matrix view
  takes no channels.

## Testing

- **Scatter**: figure-structure on xor — one trace, 4 markers; marker sizes
  ordered consistently with Σφ_R; role colors match the flag-derived
  categories (xor: abc includes the pairs via purview-union inclusion? —
  assert against the projection's actual flags, not assumed values);
  symbols reflect connectedness; `color_by="phi"` produces numeric colors;
  unknown channels raise. PCA determinism: two renders give identical
  coordinates; degenerate fallback covered with a hand-crafted projection
  whose purview unions are all identical.
- **Matrix**: exact values on a hand-crafted projection (4 nodes, known
  edges incl. a self-relation and a 3-relation) — off-diagonal sums, the
  diagonal self-relation rule, and symmetry; xor smoke (shape 4×4, row
  order by (size, label)).
- **Public API**: both views return figures; `view="bogus"` raises
  `ValueError`; lattice's default coloring unchanged (`color_by=None`
  resolves to `"phi"`).
- Whole suite + doctests; ruff; pyright via hook.

## Risks

- **PCA coordinates are abstract** — axes have no direct meaning (same as
  the paper's t-SNE); hover carries the interpretable values. Recorded, not
  mitigated.
- **Role flags use purview-union inclusion**, the projection's existing
  order, where the paper's extendedness is defined via relations
  (connection/fusion/inclusion). This is the closest structure we compute;
  the mapping is documented in the renderer docstring. A relation-derived
  role classification can replace it later without API change.
- Low overall: no φ impact, additive renderers.

## Success criteria

- All four `view=` values render from one projection; `_VIEWS_PENDING` gone.
- Scatter reproduces Fig 8's encodings (composition position, Σφ_R size,
  role color, connectedness symbol) deterministically.
- Matrix exact-value tested; suite green; ruff/pyright clean.
