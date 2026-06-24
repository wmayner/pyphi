# Simplicial-complex embedding layout: Design

**Status:** approved
**Date:** 2026-06-20

---

## 1. Motivation

The simplicial-complex view (`pyphi/visualize/render/simplicial_complex.py`)
positions vertices on size-shells: each purview subset sits on a shell whose
radius and height encode its size. This is legible but imposes a structure that
is about *cardinality*, not *similarity*. The relation faces drawn between
vertices are about purview congruence, so a layout where spatial proximity
reflects compositional similarity (MICE that tend to relate sit near each
other) would make the relational structure read more directly. This is the
geometry Haun & Tononi 2019 (Figs 7-8) show via a t-SNE embedding; the scatter
view (`render/scatter.py`) already approximates it for *distinctions* in 2D
with a deterministic PCA.

This adds a global-embedding layout for the simplicial-complex view: position
each MICE by a deterministic embedding of its composition, in 3D, and draw the
existing relation faces, links, and mechanism markers among the embedded
points.

## 2. Goals

- A new positioning family `layout="embedding"` for the simplicial-complex
  view, peer to `"barycentric"` / `"sorted"`, that replaces the shell layout
  entirely.
- Two deterministic, dependency-free embedding methods, selectable via a new
  geometry knob: PCA (default) and classical MDS.
- Embedding positions each **MICE (endpoint)** as its own point, so the
  relation faces reflect MICE/relation similarity.
- Preserve the layout contract: deterministic, a pure function of the full
  projection (so `only_distinctions` subsetting and `highlight_phi_fold`
  overlay alignment keep working), collision-free.

## 3. Non-goals

- Changing the relation-face rendering, hovers, coloring, or the
  `endpoint_placement` separation of coincident purviews (that knob simply does
  not apply under embedding — there is no shell to collide on).
- Rewiring the scatter view to share the embedding helper (see Future Work).
- Adding a new runtime dependency. PCA is numpy-only; classical MDS is
  numpy-only (scipy, already a core dependency, may be used where cleaner).
- t-SNE / UMAP (would need a new heavy optional dependency and a seed; the
  deterministic PCA/MDS pair is the chosen route).

## 4. Design

### 4.1 Where it lives

- `layout="embedding"` is accepted by `render_simplicial_complex` /
  `_positions_3d` (and already flows through `plot_ces`'s `layout=` kwarg).
- `SimplicialComplexGeometry` gains `embedding_method: str = "pca"` (or
  `"mds"`); validated, with an unknown value raising `ValueError` (mirroring the
  `layout` validation).
- New module `pyphi/visualize/render/embedding.py` holds the embedding
  primitives (MICE feature vectors, the MDS distance, `pca_embed`,
  `mds_embed`, the normalization, and a shared coincident-point spreader),
  keeping `simplicial_complex.py` focused on assembling positions.
- `_positions_3d` gets a third branch: when `layout == "embedding"`, it computes
  endpoint positions from the embedding and derives mechanism positions as
  centroids (see 4.4), bypassing the ring/shell logic.

### 4.2 What gets embedded

Each MICE (endpoint) becomes one 3D point. Its feature vector concatenates,
over the sorted unit set, three blocks:

1. **purview membership** — weighted up, since the relation faces are about
   purview congruence and this is the signal proximity should most reflect;
2. **mechanism membership** — so MICE that share a purview but belong to
   different distinctions still separate and cluster by mechanism family;
3. **direction marker** — distinguishes the cause side from the effect side.

The block weights are module constants (documented), chosen so the purview
block dominates. Two MICE coincide only if they share purview, mechanism, and
direction — i.e. they are the same MICE — so collisions are not expected; a
small-circle spreader covers any numerical ties.

### 4.3 The two methods (both deterministic, numpy-only)

- **PCA** (`embedding_method="pca"`, default): SVD of the centered
  feature-vector matrix; take the first 3 principal components; fix signs by the
  scatter view's rule (the largest-magnitude loading is positive). This is
  `render/scatter.py`'s `_pca_coords` recipe lifted from 2 to 3 components and
  from distinction vectors to MICE vectors.
- **MDS** (`embedding_method="mds"`): build a MICE-by-MICE dissimilarity matrix
  from a weighted blend of Jaccard distance on purview unit-sets, Jaccard on
  mechanisms, and a direction term; run classical (Torgerson) MDS — double-center
  the squared-distance matrix, eigendecompose, take the top 3 positive
  eigenvectors scaled by the square root of their eigenvalues. Signs fixed as in
  PCA; eigenvalues sorted descending with a deterministic tie-break so component
  order is stable. Clusters congruent (relation-forming) MICE more directly than
  linear PCA.

### 4.4 Mechanisms and the other draw elements

Under embedding, a mechanism marker sits at the **centroid of its distinction's
two embedded endpoints** — consistent with how relation hubs are already
centroids — so cause-effect links and mechanism-purview links draw correctly
with no separate mechanism embedding. Relation faces / stars are unchanged; they
consume the embedded endpoint coordinates.

### 4.5 Normalization

The embedded cloud is centered at its centroid and scaled so its largest extent
fits `max_radius`, a deterministic transform that keeps figure size stable
across structures. `z` is simply the third component (it no longer encodes
purview size in this mode).

### 4.6 Determinism, subsetting, edge cases

- No RNG anywhere; positions are a pure function of the full projection plus the
  geometry/layout arguments.
- The embedding is always computed over **all** MICE in the full projection;
  `only_distinctions` then selects a subset, so retained points never move and
  `highlight_phi_fold` overlay alignment holds (the existing contract).
- Fewer than ~4 MICE, or components with zero variance (PCA) / fewer than 3
  positive eigenvalues (MDS), fall back to spreading points evenly by id along
  the dead axes (the scatter view's degenerate-case strategy, generalized to
  3D). A single MICE sits at the origin. Residual coincidences get the
  small-circle spread.

## 5. Testing

- **`embedding.py` primitives**, directly with small synthetic inputs: the MICE
  feature vectors; the Jaccard-blend distance; `pca_embed` and `mds_embed`
  (shape, centering, determinism, sign-fix, degenerate fallback).
- **On the `xor` projection** (both methods): determinism (identical positions
  on re-call); all MICE distinct; the cloud centered and within `max_radius`; a
  proximity check that two MICE sharing a purview embed closer than two with
  disjoint purviews (the composition signal drives geometry); subset stability
  (retained MICE positions unchanged under `only_distinctions`).
- **Integration**: `plot_ces(ces, view="simplicial_complex", layout="embedding")`
  runs and produces the expected element traces under each method; the geometry
  default is `embedding_method == "pca"`; an unknown `embedding_method` raises
  `ValueError`.
- **Visual smoke**: export PNGs of `xor` under both methods to eyeball.
- Final gate: `uv run pytest` with **no path argument** (collects the `pyphi/`
  doctest sweep), under the `visualize` extra (`uv sync --all-extras`).

## 6. Future work (follow-ups — record so they are not lost)

1. **Share the PCA helper with the scatter view.** `render/scatter.py`'s
   `_pca_coords` and the new `embedding.py` PCA do the same thing at different
   dimensionalities; refactor the scatter view to consume the shared helper.
2. **Expose the feature-vector block weights as user knobs.** The purview /
   mechanism / direction weights are module constants in this version; promote
   them to `SimplicialComplexGeometry` fields if users want to tune the balance.
3. **Flat 2D-in-3D toggle.** An option to embed to 2 components with `z = 0`
   (mirroring the scatter view's planar layout) for a flatter look.

## 7. Acceptance criteria

- `layout="embedding"` positions every MICE by the chosen embedding; the
  shell radius/size structure is not used in this mode.
- `embedding_method` selects PCA (default) or classical MDS; both are
  deterministic, numpy-only, and collision-free.
- Mechanisms render at the centroid of their endpoints; relation faces, links,
  and hovers work unchanged on the embedded coordinates.
- `only_distinctions` and `highlight_phi_fold` keep aligning (positions are a
  pure function of the full projection).
- New tests pass (primitives, determinism, distinctness, normalization,
  proximity, subset stability, integration, validation); `uv run pytest` with no
  path argument is green.
- Changelog fragment present; the three Future Work items are recorded.
