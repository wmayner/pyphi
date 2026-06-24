"""Helpers shared by the render backends."""

from __future__ import annotations

import math

import numpy as np

CHANNEL_TITLES = {"phi": "φ", "sum_phi_relations": "Σφ_R"}


def rescale(values: list[float], lo: float, hi: float) -> list[float]:
    """Map values linearly onto [lo, hi]; midpoint if they are all equal."""
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [(lo + hi) / 2.0] * len(values)
    return [lo + (v - vmin) / (vmax - vmin) * (hi - lo) for v in values]


def spread_coincident(coords: np.ndarray, radius: float) -> np.ndarray:
    """Spread points sharing (numerically) the same spot onto a small xy circle.

    Points whose coordinates round to the same bucket are nudged apart on a
    circle of ``radius`` in the xy-plane, leaving any further axes untouched.
    """
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
