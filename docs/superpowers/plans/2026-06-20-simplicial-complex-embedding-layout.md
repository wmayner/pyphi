# Simplicial-complex embedding layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `layout="embedding"` positioning family to the simplicial-complex view that places each MICE by a deterministic PCA or classical-MDS embedding of its composition, so spatial proximity reflects compositional similarity instead of purview size.

**Architecture:** A new module `pyphi/visualize/render/embedding.py` holds the embedding primitives (MICE feature vectors, a Jaccard-blend distance, `pca_embed`, `mds_embed`, normalization, coincident-point spreading) and a top-level `embedding_positions(projection, geometry)` that returns endpoint and mechanism positions. `_positions_3d` in `simplicial_complex.py` gains an `"embedding"` branch that delegates to it; mechanisms sit at the centroid of their two embedded endpoints. The method is chosen by a new `SimplicialComplexGeometry.embedding_method` field.

**Tech Stack:** Python 3.12+, numpy (core dep), plotly, pytest. No new dependency. Visualization needs the `visualize` extra.

**Spec:** `docs/superpowers/specs/2026-06-20-simplicial-complex-embedding-layout-design.md`

## Global Constraints

- Deterministic, no RNG: the layout is a pure function of the full projection plus geometry/layout args (required for `highlight_phi_fold` overlay alignment and reproducibility).
- Embedding is always computed over **all** MICE in the full projection; `only_distinctions` selects a subset afterward (in `render_simplicial_complex`), so retained points never move.
- numpy-only; no new runtime dependency.
- Python 3.12+, no back-compat shims. Use `uv run` for all commands. Final verification: `uv run pytest` with **no path argument**, under `uv sync --all-extras`.
- Do not bypass pre-commit (ruff + pyright). Stage only the files each task names (the tree has unrelated untracked work; never `git add -A`).
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```

---

### Task 1: Embedding primitives module

**Files:**
- Create: `pyphi/visualize/render/embedding.py`
- Test: `test/test_visualize_embedding.py`

**Interfaces:**
- Consumes: `CESProjection` (its `.nodes` with `.mechanism`/`.cause_purview`/`.effect_purview`, and `.endpoints` with `.id`/`.distinction_id`/`.direction`/`.purview`). Endpoint `id` equals its index in `projection.endpoints`.
- Produces: `_mice_vectors(projection) -> np.ndarray`, `_mice_distance(projection) -> np.ndarray`, `pca_embed(vectors, n_components=3) -> np.ndarray`, `mds_embed(distance, n_components=3) -> np.ndarray`, `embedding_positions(projection, geometry) -> tuple[dict[int, Point], dict[int, Point]]`. Module constants `_PURVIEW_WEIGHT=1.0`, `_MECHANISM_WEIGHT=0.5`, `_DIRECTION_WEIGHT=0.5`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_visualize_embedding.py`:

```python
import math

import numpy as np
import pytest


def test_mice_vectors_blocks_set():
    from types import SimpleNamespace

    from pyphi.visualize.render.embedding import (
        _MECHANISM_WEIGHT,
        _PURVIEW_WEIGHT,
        _mice_vectors,
    )

    # One distinction (mechanism (0,)), cause purview (1, 2), effect purview (2,).
    nodes = [SimpleNamespace(id=0, mechanism=(0,), cause_purview=(1, 2), effect_purview=(2,))]
    endpoints = [
        SimpleNamespace(id=0, distinction_id=0, direction="cause", purview=(1, 2)),
        SimpleNamespace(id=1, distinction_id=0, direction="effect", purview=(2,)),
    ]
    proj = SimpleNamespace(nodes=nodes, endpoints=endpoints)
    v = _mice_vectors(proj)
    # Units present: {0, 1, 2}; width = 3; columns 0..2 purview, 3..5 mechanism,
    # 6 direction.
    assert v.shape == (2, 7)
    # Cause row: purview {1,2} -> cols 1,2; mechanism {0} -> col 3; direction -.
    assert v[0, 1] == _PURVIEW_WEIGHT and v[0, 2] == _PURVIEW_WEIGHT
    assert v[0, 0] == 0.0
    assert v[0, 3] == _MECHANISM_WEIGHT
    assert v[0, 6] < 0  # cause marker negative
    assert v[1, 6] > 0  # effect marker positive


def test_mice_distance_purview_overlap_orders_pairs():
    from types import SimpleNamespace

    from pyphi.visualize.render.embedding import _mice_distance

    nodes = [
        SimpleNamespace(id=0, mechanism=(0,), cause_purview=(0, 1), effect_purview=(0, 1)),
        SimpleNamespace(id=1, mechanism=(1,), cause_purview=(0, 1), effect_purview=(2,)),
    ]
    # ep0,ep2 share purview (0,1); ep0,ep3 are disjoint ((0,1) vs (2,)).
    endpoints = [
        SimpleNamespace(id=0, distinction_id=0, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=1, distinction_id=0, direction="effect", purview=(0, 1)),
        SimpleNamespace(id=2, distinction_id=1, direction="cause", purview=(0, 1)),
        SimpleNamespace(id=3, distinction_id=1, direction="effect", purview=(2,)),
    ]
    proj = SimpleNamespace(nodes=nodes, endpoints=endpoints)
    d = _mice_distance(proj)
    assert d.shape == (4, 4)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0.0)
    # Sharing a purview is nearer than disjoint purviews.
    assert d[0, 2] < d[0, 3]


def test_pca_embed_shape_centered_deterministic():
    from pyphi.visualize.render.embedding import pca_embed

    rng = np.arange(30, dtype=float).reshape(10, 3)
    a = pca_embed(rng)
    assert a.shape == (10, 3)
    assert np.allclose(a.mean(axis=0), 0.0, atol=1e-9)  # centered
    assert np.array_equal(a, pca_embed(rng))  # deterministic


def test_pca_embed_degenerate_fallback():
    from pyphi.visualize.render.embedding import pca_embed

    # All rows identical: zero variance -> every component falls back to a
    # spread by id, so points are still distinct.
    flat = np.ones((4, 5))
    a = pca_embed(flat)
    assert not np.any(np.isnan(a))
    assert len({tuple(row) for row in a}) == 4


def test_mds_embed_recovers_line_order():
    from pyphi.visualize.render.embedding import mds_embed

    # Four points on a line at 0,1,2,3: classical MDS recovers their order on
    # the first axis (up to sign, which the sign-fix pins).
    pts = np.array([0.0, 1.0, 2.0, 3.0])
    dist = np.abs(pts[:, None] - pts[None, :])
    a = mds_embed(dist)
    assert a.shape == (4, 3)
    first = a[:, 0]
    order = np.argsort(first)
    assert list(order) == [0, 1, 2, 3] or list(order) == [3, 2, 1, 0]


def test_mds_embed_deterministic():
    from pyphi.visualize.render.embedding import mds_embed

    dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    assert np.array_equal(mds_embed(dist), mds_embed(dist))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest test/test_visualize_embedding.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.visualize.render.embedding'`.

- [ ] **Step 3: Create the module**

Create `pyphi/visualize/render/embedding.py`:

```python
"""Global-embedding layout for the simplicial-complex view.

Positions each MICE (endpoint) by a deterministic embedding of its composition,
so spatial proximity reflects compositional similarity rather than purview size.
Two methods: PCA of a composition feature vector, and classical (Torgerson) MDS
of a purview-overlap distance. Both are deterministic and numpy-only.
"""

from __future__ import annotations

import math

import numpy as np

Point = tuple[float, float, float]

# Feature-vector block weights (PCA) and distance-blend weights (MDS). Purview
# dominates because the relation faces are about purview congruence.
_PURVIEW_WEIGHT = 1.0
_MECHANISM_WEIGHT = 0.5
_DIRECTION_WEIGHT = 0.5


def _units(projection) -> list[int]:
    return sorted(
        {
            u
            for n in projection.nodes
            for u in (*n.mechanism, *n.cause_purview, *n.effect_purview)
        }
    )


def _mice_vectors(projection) -> np.ndarray:
    """Composition feature vectors for the endpoints (row index == endpoint id).

    Each row concatenates three weighted blocks over the sorted unit set:
    purview membership, mechanism membership, and a signed direction marker.
    """
    units = _units(projection)
    column = {u: k for k, u in enumerate(units)}
    width = len(units)
    nodes_by_id = {n.id: n for n in projection.nodes}
    vectors = np.zeros((len(projection.endpoints), 2 * width + 1))
    for e in projection.endpoints:
        for u in e.purview:
            vectors[e.id, column[u]] = _PURVIEW_WEIGHT
        for u in nodes_by_id[e.distinction_id].mechanism:
            vectors[e.id, width + column[u]] = _MECHANISM_WEIGHT
        vectors[e.id, 2 * width] = (
            _DIRECTION_WEIGHT if e.direction == "effect" else -_DIRECTION_WEIGHT
        )
    return vectors


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return 1.0 - len(sa & sb) / len(union) if union else 0.0


def _mice_distance(projection) -> np.ndarray:
    """Pairwise endpoint dissimilarity for MDS: a weighted blend of Jaccard
    distance on purviews, Jaccard on mechanisms, and a direction term.
    """
    eps = projection.endpoints
    nodes_by_id = {n.id: n for n in projection.nodes}
    n = len(eps)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = (
                _PURVIEW_WEIGHT * _jaccard(eps[i].purview, eps[j].purview)
                + _MECHANISM_WEIGHT
                * _jaccard(
                    nodes_by_id[eps[i].distinction_id].mechanism,
                    nodes_by_id[eps[j].distinction_id].mechanism,
                )
                + _DIRECTION_WEIGHT * float(eps[i].direction != eps[j].direction)
            )
            dist[i, j] = dist[j, i] = d
    return dist


def _fallback_axis(n: int) -> np.ndarray:
    return np.linspace(-0.5, 0.5, n) if n > 1 else np.zeros(1)


def _sign_fix(component: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Flip so the largest-magnitude loading is positive (deterministic)."""
    if loadings[np.argmax(np.abs(loadings))] < 0:
        return -component
    return component


def pca_embed(vectors: np.ndarray, n_components: int = 3) -> np.ndarray:
    """First ``n_components`` principal components of ``vectors`` (rows = items),
    sign-fixed; zero-variance components fall back to an even spread by id.
    """
    n = len(vectors)
    centered = vectors - vectors.mean(axis=0)
    u_mat, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = np.zeros((n, n_components))
    for c in range(n_components):
        if c < len(s) and s[c] > 1e-9:
            coords[:, c] = _sign_fix(u_mat[:, c] * s[c], vt[c])
        else:
            coords[:, c] = _fallback_axis(n)
    return coords


def mds_embed(distance: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Classical (Torgerson) MDS of a dissimilarity matrix, sign-fixed; axes
    with non-positive eigenvalues fall back to an even spread by id.
    """
    n = len(distance)
    d2 = np.asarray(distance, dtype=float) ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ d2 @ centering
    vals, vecs = np.linalg.eigh(gram)  # ascending
    order = np.argsort(vals)[::-1]  # descending
    coords = np.zeros((n, n_components))
    for c in range(n_components):
        idx = int(order[c]) if c < len(order) else None
        if idx is not None and vals[idx] > 1e-9:
            coords[:, c] = _sign_fix(vecs[:, idx] * math.sqrt(vals[idx]), vecs[:, idx])
        else:
            coords[:, c] = _fallback_axis(n)
    return coords


def _normalize_cloud(coords: np.ndarray, max_radius: float) -> np.ndarray:
    """Center at the centroid and scale so the largest extent fits max_radius."""
    centered = coords - coords.mean(axis=0)
    scale = float(np.max(np.abs(centered)))
    if scale < 1e-12:
        return centered
    return centered / scale * max_radius


def _spread_coincident(coords: np.ndarray, radius: float) -> np.ndarray:
    """Spread points sharing (numerically) the same spot on a small xy circle."""
    span = float(np.max(np.abs(coords))) or 1.0
    quantum = span * 1e-6
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(coords):
        groups.setdefault(tuple(np.round(p / quantum).astype(int)), []).append(i)
    out = coords.copy()
    for ids in groups.values():
        if len(ids) > 1:
            for k, i in enumerate(ids):
                angle = 2 * math.pi * k / len(ids)
                out[i, 0] += radius * math.cos(angle)
                out[i, 1] += radius * math.sin(angle)
    return out


def embedding_positions(projection, geometry) -> tuple[dict[int, Point], dict[int, Point]]:
    """Endpoint and mechanism positions from a global composition embedding.

    ``geometry.embedding_method`` selects ``"pca"`` or ``"mds"``. Each MICE is
    one point; a mechanism sits at the centroid of its two endpoints.
    """
    method = geometry.embedding_method
    if method == "pca":
        coords = pca_embed(_mice_vectors(projection))
    elif method == "mds":
        coords = mds_embed(_mice_distance(projection))
    else:
        raise ValueError(f"unknown embedding_method {method!r}")
    coords = _normalize_cloud(coords, geometry.max_radius)
    coords = _spread_coincident(coords, geometry.max_radius * 0.02)
    endpoint_pos: dict[int, Point] = {
        e.id: (float(coords[e.id, 0]), float(coords[e.id, 1]), float(coords[e.id, 2]))
        for e in projection.endpoints
    }
    mechanism_pos: dict[int, Point] = {}
    for node in projection.nodes:
        cx, cy, cz = endpoint_pos[2 * node.id]
        ex, ey, ez = endpoint_pos[2 * node.id + 1]
        mechanism_pos[node.id] = ((cx + ex) / 2, (cy + ey) / 2, (cz + ez) / 2)
    return endpoint_pos, mechanism_pos
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_visualize_embedding.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check pyphi/visualize/render/embedding.py && uv run ruff format --check pyphi/visualize/render/embedding.py && uv run pyright pyphi/visualize/render/embedding.py`
Expected: all clean (0 errors). Fix any line-length/format issues by wrapping; re-run.

- [ ] **Step 6: Commit**

```bash
git add pyphi/visualize/render/embedding.py test/test_visualize_embedding.py
git commit -m "Add embedding primitives for the simplicial-complex layout

PCA and classical-MDS embeddings of MICE composition (purview/mechanism/
direction), with a Jaccard-blend distance, deterministic sign-fixing,
normalization, and coincident-point spreading."
```
(append the trailer)

---

### Task 2: Wire `layout="embedding"` into the positioning pipeline

**Files:**
- Modify: `pyphi/visualize/render/simplicial_complex.py` (`SimplicialComplexGeometry`, `_positions_3d`)
- Test: `test/test_visualize_simplicial_complex.py`

**Interfaces:**
- Consumes: `embedding_positions(projection, geometry)` from Task 1.
- Produces: `SimplicialComplexGeometry.embedding_method: str = "pca"`; `_positions_3d(..., layout="embedding")` returns embedded `(endpoint_pos, mechanism_pos)`.

- [ ] **Step 1: Write the failing tests**

In `test/test_visualize_simplicial_complex.py`, add:

```python
def test_embedding_layout_distinct_normalized_and_centroids(xor_projection):
    import math

    from pyphi.visualize.render.simplicial_complex import (
        SimplicialComplexGeometry,
        _positions_3d,
    )

    geo = SimplicialComplexGeometry(max_radius=1.0)  # embedding_method defaults to pca
    epos, mpos = _positions_3d(xor_projection, geo, layout="embedding")
    # One point per endpoint, all distinct.
    assert set(epos) == set(range(8))
    assert len(set(epos.values())) == 8
    # Centered and within max_radius (allowing the small coincident-spread).
    assert all(math.hypot(x, y, z) <= 1.0 + 0.05 for x, y, z in epos.values())
    # A mechanism sits at the centroid of its two endpoints.
    for d in range(4):
        cx, cy, cz = epos[2 * d]
        ex, ey, ez = epos[2 * d + 1]
        assert mpos[d] == pytest.approx(((cx + ex) / 2, (cy + ey) / 2, (cz + ez) / 2))


def test_embedding_layout_deterministic(xor_projection):
    from pyphi.visualize.render.simplicial_complex import (
        SimplicialComplexGeometry,
        _positions_3d,
    )

    for method in ("pca", "mds"):
        geo = SimplicialComplexGeometry(embedding_method=method)
        a = _positions_3d(xor_projection, geo, layout="embedding")
        b = _positions_3d(xor_projection, geo, layout="embedding")
        assert a == b


def test_embedding_method_validation(xor_projection):
    from pyphi.visualize.render.simplicial_complex import (
        SimplicialComplexGeometry,
        _positions_3d,
    )

    geo = SimplicialComplexGeometry(embedding_method="bogus")
    with pytest.raises(ValueError, match="embedding_method"):
        _positions_3d(xor_projection, geo, layout="embedding")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q -k embedding`
Expected: FAIL (`SimplicialComplexGeometry` has no `embedding_method`; `_positions_3d` rejects `layout="embedding"`).

- [ ] **Step 3: Add the geometry field**

In `pyphi/visualize/render/simplicial_complex.py`, in `SimplicialComplexGeometry` (the dataclass with `endpoint_placement`), add after `endpoint_placement`:

```python
    embedding_method: str = "pca"
```

- [ ] **Step 4: Add the `embedding` branch to `_positions_3d`**

In `_positions_3d`, change the layout validation and add the branch at the top (the existing body computes shell rings and must not run for embedding). Replace:

```python
    if layout not in ("barycentric", "sorted"):
        raise ValueError(f"unknown layout {layout!r}")
    purview_rings = _rings(e.purview for e in projection.endpoints)
```

with:

```python
    if layout not in ("barycentric", "sorted", "embedding"):
        raise ValueError(f"unknown layout {layout!r}")
    if layout == "embedding":
        from pyphi.visualize.render.embedding import embedding_positions

        return embedding_positions(projection, geometry)
    purview_rings = _rings(e.purview for e in projection.endpoints)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/test_visualize_simplicial_complex.py -q -k embedding`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint, type-check, and run the whole viz file**

Run: `uv run ruff check pyphi/visualize/render/simplicial_complex.py && uv run pyright pyphi/visualize/render/simplicial_complex.py && uv run pytest test/test_visualize_simplicial_complex.py -q`
Expected: clean; all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyphi/visualize/render/simplicial_complex.py test/test_visualize_simplicial_complex.py
git commit -m "Add layout='embedding' to the simplicial-complex positioning

_positions_3d delegates to embedding_positions for layout='embedding';
SimplicialComplexGeometry gains embedding_method ('pca' default, or 'mds')."
```
(append the trailer)

---

### Task 3: `plot_ces` integration, subset stability, docs, and closeout

**Files:**
- Modify: `pyphi/visualize/__init__.py` (`plot_ces` docstring `layout` entry)
- Modify: `test/test_visualize_simplicial_complex.py`, `test/test_visualize_projection.py`
- Create: `changelog.d/embedding-layout.feature.md`

**Interfaces:**
- Consumes: `plot_ces(ces, view="simplicial_complex", layout="embedding", geometry=...)` (the `layout=` kwarg already flows to `render_simplicial_complex`).

- [ ] **Step 1: Write the failing tests**

In `test/test_visualize_simplicial_complex.py`, add:

```python
def test_plot_ces_embedding_layout_runs():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_ces
    from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry

    ces = examples.xor_system().ces()
    for method in ("pca", "mds"):
        fig = plot_ces(
            ces,
            view="simplicial_complex",
            layout="embedding",
            geometry=SimplicialComplexGeometry(embedding_method=method),
        )
        assert isinstance(fig, go.Figure)
        # Purview dots present (the embedded MICE).
        assert any(getattr(t, "mode", None) == "markers+text" for t in fig.data)


def test_embedding_layout_subset_stability(xor_projection):
    from pyphi.visualize.render.simplicial_complex import (
        SimplicialComplexGeometry,
        render_simplicial_complex,
    )
    from pyphi.visualize.theme import DEFAULT_THEME

    geo = SimplicialComplexGeometry()
    full = render_simplicial_complex(
        xor_projection, DEFAULT_THEME, geometry=geo, layout="embedding"
    )
    sub = render_simplicial_complex(
        xor_projection,
        DEFAULT_THEME,
        geometry=geo,
        layout="embedding",
        only_distinctions={0, 3},
    )
    full_pts = set(zip(full.data[0].x, full.data[0].y, full.data[0].z, strict=True))
    sub_pts = set(zip(sub.data[0].x, sub.data[0].y, sub.data[0].z, strict=True))
    # The retained subset's purview dots keep their full-render coordinates.
    assert sub_pts <= full_pts
    assert len(sub_pts) == 4  # 2 distinctions -> 4 endpoints
```

In `test/test_visualize_projection.py`, extend `test_simplicial_complex_geometry_defaults` (the test asserting `endpoint_placement`) by adding:

```python
    assert geo.embedding_method == "pca"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest test/test_visualize_simplicial_complex.py test/test_visualize_projection.py -q -k "embedding or geometry_defaults"`
Expected: FAIL only on the new assertions if the docstring/geometry pieces are missing (the geometry default assertion fails until Task 2's field exists — it does — so this should actually pass; the new integration tests pass once Task 2 is in). If `test_plot_ces_embedding_layout_runs` fails, it indicates a real wiring gap to fix here.

(If all three already pass because the `layout=` kwarg flows and the field exists, that is expected — this task is mostly docs/closeout. Proceed to Step 3.)

- [ ] **Step 3: Document the layout in `plot_ces`**

In `pyphi/visualize/__init__.py`, in the `plot_ces` docstring `layout` Keyword Arg entry, append a sentence describing the embedding layout. Find the `layout (str):` block and add:

```
            ``"embedding"`` (simplicial-complex view) ignores the shells and
            positions each MICE by a deterministic embedding of its composition
            (``SimplicialComplexGeometry.embedding_method`` selects ``"pca"`` or
            ``"mds"``), so proximity reflects compositional similarity.
```

- [ ] **Step 4: Write the changelog fragment**

Create `changelog.d/embedding-layout.feature.md`:

```markdown
The simplicial-complex view (`pyphi.visualize.plot_ces`) gains a
`layout="embedding"` mode that positions each MICE by a deterministic embedding
of its composition (purview, mechanism, and direction) instead of the size
shells, so spatial proximity reflects compositional similarity. The method is
chosen by `SimplicialComplexGeometry.embedding_method`: `"pca"` (default, a
principal-component embedding) or `"mds"` (classical multidimensional scaling of
a purview-overlap distance). Both are deterministic and need no new dependency.
```

- [ ] **Step 5: Run the targeted tests**

Run: `uv run pytest test/test_visualize_simplicial_complex.py test/test_visualize_projection.py -q`
Expected: PASS.

- [ ] **Step 6: Visual smoke (manual, optional but recommended)**

Run a short script to export PNGs of `xor` under both methods and eyeball that each is a sensible 3-D cloud with relation faces:

```bash
uv run python -c "
import pyphi; from pyphi import examples; import pyphi.visualize as V
from pyphi.visualize.render.simplicial_complex import SimplicialComplexGeometry
pyphi.config.progress_bars = False
ces = examples.xor_system().ces()
for m in ('pca','mds'):
    fig = V.plot_ces(ces, view='simplicial_complex', layout='embedding', geometry=SimplicialComplexGeometry(embedding_method=m))
    fig.update_layout(width=760, height=620); fig.write_image(f'/tmp/embed_{m}.png', scale=2)
    print('wrote', m)
"
```
(Requires kaleido; `uv pip install kaleido` if absent.)

- [ ] **Step 7: Full gate**

Run: `uv sync --all-extras` then `uv run pytest` (NO path argument).
Expected: PASS (collects the `pyphi/` doctest sweep).

- [ ] **Step 8: Commit**

```bash
git add pyphi/visualize/__init__.py test/test_visualize_simplicial_complex.py test/test_visualize_projection.py changelog.d/embedding-layout.feature.md
git commit -m "Document and verify layout='embedding'; changelog

plot_ces documents the embedding layout; tests cover plot_ces integration,
only_distinctions subset stability, and the embedding_method geometry default."
```
(append the trailer)

---

## Self-Review

**Spec coverage:**
- `layout="embedding"` peer to barycentric/sorted (spec §4.1) → Task 2 Steps 3-4.
- `embedding_method` geometry knob, PCA default + MDS (spec §4.1, §4.3) → Task 1 (`embedding_positions`), Task 2 (field).
- MICE embedded directly; three-block vector, purview weighted up (spec §4.2) → Task 1 `_mice_vectors`.
- Mechanism = centroid of endpoints (spec §4.4) → Task 1 `embedding_positions`.
- Normalization to `max_radius` (spec §4.5) → Task 1 `_normalize_cloud`.
- Determinism / pure function of full projection / subset stability (spec §4.6) → Task 1 (no RNG, sign-fix), Task 3 `test_embedding_layout_subset_stability`.
- Fallbacks for degenerate/coincident (spec §4.6) → Task 1 `_fallback_axis`, `_spread_coincident`, `test_pca_embed_degenerate_fallback`.
- Testing items (spec §5): primitives, determinism, distinctness, normalization, proximity (distance ordering in `test_mice_distance_purview_overlap_orders_pairs` and MDS line recovery), subset stability, integration, validation, geometry default → Tasks 1-3.
- Changelog + future-work recorded (spec §6, §7) → Task 3 Step 4; future-work lives in the committed spec §6.

**Placeholder scan:** none — every code step has complete code. Task 3 Step 2's note explains that some new assertions may already pass once Task 2 lands (this is expected for a docs/closeout task, not a placeholder).

**Type consistency:** `embedding_positions(projection, geometry) -> (dict[int, Point], dict[int, Point])` matches `_positions_3d`'s return contract. `pca_embed`/`mds_embed` both take a 2-D array and return `(n, n_components)`. `embedding_method` is `str` with default `"pca"` in both the geometry field (Task 2) and the validation (Task 1 `embedding_positions`). Endpoint `id` is used as the row index consistently (Task 1 builds rows by `e.id`; `embedding_positions` reads `coords[e.id]`).

**Proximity-test note (deliberate):** the spec's "two MICE sharing a purview embed closer than disjoint" is asserted robustly via the *distance* primitive (`test_mice_distance_purview_overlap_orders_pairs`) and MDS's distance-preserving recovery (`test_mds_embed_recovers_line_order`), rather than as a fragile property of the linear PCA projection. This satisfies "the composition signal drives geometry" without a brittle assertion on PCA output.
