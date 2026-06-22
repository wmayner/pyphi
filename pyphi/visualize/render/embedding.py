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


def embedding_positions(
    projection, geometry
) -> tuple[dict[int, Point], dict[int, Point]]:
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
