# Relations follow-ups: analytical-safe visualization and diff

**Date:** 2026-07-11
**Status:** approved (design)
**Branch:** `wave7-relations-followups`

## Problem

The relations query surface (merged 2026-07-11, `b3e8ec49`) added an
`AnalyticalRelations` backend, selected by
`config.formalism.iit.relation_computation = "ANALYTICAL"`. It answers moments,
degree spectrum, counts, sums, and a lazy `strongest(k)` stream in closed form,
without ever enumerating the relation set — which for large structures (Fig 6D:
~1.5M relations) cannot be materialized.

Two consumers still assume relations are **enumerable** (a `ConcreteRelations`
frozenset) and break, or silently degrade, on the analytical backend:

1. **Visualization** — `pyphi/visualize/projection/__init__.py::project_ces`
   iterates `ces.relations` to build edges (line ~269) and reads
   `ces.relations.faces_by_degree` (line ~308, via `_faces`). Neither `__iter__`
   nor `faces_by_degree` exists on `AnalyticalRelations`, so projecting an
   analytical cause-effect structure raises.

2. **Diff** — `pyphi/models/ces.py::CauseEffectStructure._changes` compares
   relations by `set(self.relations)` / `set(other.relations)`, guarded with
   `hasattr(..., "__iter__")`. For analytical relations the guard falls through
   to `set()`, so the diff silently reports **zero** relation changes even when
   Σφ_r, relation count, or the degree spectrum differ.

## Approach: Hybrid

Statistics (closed-form, available on both backends) are the always-present
relation summary. Per-relation detail (edge list, gained/lost rows) is retained
wherever the relations are enumerable, so the concrete path loses nothing.

## 1. Visualization

`project_ces(ces, node_labels=None, max_relations=None)` gains a
`max_relations: int | None` parameter, threaded from both entry points that call
it in `pyphi/visualize/__init__.py`: `plot_ces` and `highlight_phi_fold` (the
latter over `AnalyticalFoldRelations`, which also implements `strongest`).

- **Edges** come from `ces.relations.strongest(k=max_relations)` instead of
  iterating `ces.relations`. `Relation` already exposes `.mechanisms`, `.phi`,
  `.purview`, and `len()`; edge construction is otherwise unchanged.
- **Faces** are derived from the same top-k `Relation` objects via each
  relation's `.faces`, replacing the concrete-only `faces_by_degree` path. A
  single traversal of `strongest(k)` produces both edges and faces, keeping the
  two consistent (faces belong to rendered relations only).
- **`k=None`** → the base `Relations.strongest` yields every relation (now in
  φ-descending order). For `ConcreteRelations` the rendered content is unchanged.
- **Analytical + `k=None`** raises `ValueError`: the analytical relation set is
  unbounded in practice, and silently drawing the top-N of millions would
  misrepresent the structure as complete. The message directs the caller to pass
  `max_relations`.
- **Node marker size** (`DistinctionNode.sum_phi_relations`) stays faithful to
  the full structure: it is each distinction's true incident Σφ_r over *all*
  relations touching it, independent of `max_relations`. Sizing from only the
  rendered top-k would make two distinctions look equal when one carries far
  more binding outside the cap — the same misrepresentation the uncapped-error
  guards against. The value is supplied by a new backend method (below), so it
  is exact on the concrete backend and closed-form (no enumeration) on the
  analytical one.

### New method: `Relations.sum_phi_by_distinction`

`Relations.sum_phi_by_distinction(distinctions) -> tuple[float, ...]` returns
each distinction's incident Σφ_r (the Σφ_r over relations containing it),
aligned to the given distinction order.

- **Base / `ConcreteRelations`**: one pass over the relation set, adding each
  relation's φ_r to every relatum it contains. Identical to the sum the current
  projector derives from the edge list when uncapped.
- **`AnalyticalRelations`**: closed form. A distinction's incident sum is
  `total − Σφ_r(relations avoiding it)`; all n are computed in a single pass
  over the atom incidence index, without enumerating relations. (This is the
  quantity `AnalyticalFoldRelations` already exposes as a single-distinction
  fold's Σφ_r; the batch method shares one traversal instead of n folds.)

The projector calls this for node sizing, so it needs no `isinstance` branch.

### Spectrum view

`project_ces` is called for every view, including `"spectrum"`, whose renderer
(`render/spectrum.py`) currently builds its per-degree count/Σφ bars by iterating
`projection.edges`. Under a relation cap those bars would show only the top-k —
a silently truncated census, the same failure mode faithful sizing avoids. The
spectrum is a closed-form statistic, so:

- `CESProjection` gains a `degree_spectrum: dict[int, tuple[int, float]]` field,
  populated from `ces.relations.degree_spectrum()` (closed-form on both
  backends).
- `render/spectrum.py` reads `projection.degree_spectrum` instead of iterating
  edges, making the spectrum exact and independent of `max_relations`.

The lattice, scatter, matrix, and hypergraph views legitimately render the
strongest relations, so they are unchanged beyond consuming the capped edge set.

## 2. Diff

`CauseEffectStructure._changes` (`pyphi/models/ces.py`) replaces the
enumerate-or-empty relation block:

- **Always** compute closed-form relation statistics on both sides and emit one
  `Change` per statistic that differs (floats compared with `numerics.eq`):
  - `"relation_sum_phi"` — `a`/`b` = Σφ_r (`relations.sum_phi()`)
  - `"relation_count"` — `a`/`b` = total count (`relations.num_relations()`)
  - `"relation_degree"`, keyed by degree — `a`/`b` = `(count, Σφ)` per degree
    present on either side (`relations.degree_spectrum()`)

  Rows are emitted only where the value actually differs.
- **Additionally**, when both relation objects are enumerable
  (`hasattr(..., "__iter__")`), keep the existing per-relation
  `relation_gained` / `relation_lost` rows.

The new `Change.kind` values are additive. `Change` and `ResultDiff._describe` /
`to_pandas` already render arbitrary `(kind, key, a, b)` rows, so no structural
change to `models/diff.py` is required.

## Interfaces touched

- `pyphi/visualize/projection/__init__.py` — `project_ces` signature + edge/face
  construction; `_faces` reworked to take relations from `strongest(k)`;
  `CESProjection` gains `degree_spectrum`; the now-dead `_sum_phi_relations`
  helper is removed.
- `pyphi/visualize/__init__.py` — `plot_ces` and `highlight_phi_fold` gain and
  forward `max_relations`.
- `pyphi/visualize/render/spectrum.py` — reads `projection.degree_spectrum`.
- `pyphi/models/ces.py` — `_changes` relation block.
- `pyphi/relations.py` — new `Relations.sum_phi_by_distinction` (base, iterating)
  and `AnalyticalRelations.sum_phi_by_distinction` (closed-form override).

`models/diff.py` and the other renderers (lattice, scatter, matrix, hypergraph)
are unchanged; they already provide the generic `Change` rendering and consume
the capped edge set as intended.

## Testing

- **Viz:** projecting a small concrete CES is unchanged (golden on edges/faces).
  Under `relation_computation="ANALYTICAL"`, `project_ces(ces,
  max_relations=k)` renders the top-k, and `project_ces(ces)` (no cap) raises
  `ValueError`. Parity: the mechanism/φ set of `strongest(k)` edges is a subset
  of the concrete edge set for the same CES; and analytical
  `sum_phi_by_distinction` equals the concrete per-distinction incident Σφ_r,
  distinction by distinction, on a small network.
- **Spectrum:** `project_ces` carries a `degree_spectrum` equal to
  `ces.relations.degree_spectrum()`; the value is identical whether or not a cap
  is applied, and matches between the concrete and analytical backends.
- **Diff:** two cause-effect structures computed under `ANALYTICAL` config now
  produce nonzero `relation_sum_phi` / `relation_count` / `relation_degree`
  deltas where they differ (zero today). A concrete diff still lists per-relation
  `relation_gained` / `relation_lost` **and** now carries the statistic deltas.
- Changelog fragment under `changelog.d/`.

## Out of scope

- Any change to the lattice, scatter, matrix, or hypergraph renderers (only
  `render/spectrum.py` changes, to read the closed-form spectrum).
- Any change to relation computation beyond the additive
  `sum_phi_by_distinction` query.
- ROADMAP N6/N24 are already landed; this is a follow-up, not a new roadmap row.
