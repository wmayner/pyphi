"""Helpers shared by the render backends."""

from __future__ import annotations

CHANNEL_TITLES = {"phi": "φ", "sum_phi_relations": "Σφ_R"}


def rescale(values: list[float], lo: float, hi: float) -> list[float]:
    """Map values linearly onto [lo, hi]; midpoint if they are all equal."""
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [(lo + hi) / 2.0] * len(values)
    return [lo + (v - vmin) / (vmax - vmin) * (hi - lo) for v in values]
