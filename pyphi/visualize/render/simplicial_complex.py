"""3-D simplicial-complex renderer for phi-structure projections."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from pyphi.visualize.projection import PhiStructureProjection

Point = tuple[float, float, float]


@dataclass(frozen=True)
class SimplicialComplexGeometry:
    """Plot-space layout knobs for the simplicial-complex view."""

    max_radius: float = 1.0
    z_spacing: float = 0.0
    direction_offset: float = 0.5
    purview_jitter: float = 0.1


def _polygon_points(n: int, radius: float, z: float) -> list[Point]:
    """``n`` points evenly spaced on a circle of ``radius`` at height ``z``."""
    return [
        (
            radius * math.cos(2 * math.pi * k / n),
            radius * math.sin(2 * math.pi * k / n),
            z,
        )
        for k in range(n)
    ]


def _shell_positions(
    subsets: Iterable[tuple[int, ...]], geometry: SimplicialComplexGeometry
) -> dict[tuple[int, ...], Point]:
    """Place each unique subset on the shell for its size.

    Subsets of size k share a circular shell whose radius grows linearly
    with k up to ``max_radius``; within a shell, subsets sit on a regular
    polygon in sorted order. Shells stack in z by ``z_spacing``.
    """
    by_size: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for s in sorted(set(subsets)):
        by_size[len(s)].append(s)
    sizes = sorted(by_size)
    k_max = max(sizes)
    positions: dict[tuple[int, ...], Point] = {}
    for shell_index, k in enumerate(sizes):
        members = by_size[k]
        radius = geometry.max_radius * k / k_max
        z = geometry.z_spacing * shell_index
        positions.update(
            zip(members, _polygon_points(len(members), radius, z), strict=True)
        )
    return positions


def _endpoint_positions(
    projection: PhiStructureProjection, geometry: SimplicialComplexGeometry
) -> dict[int, Point]:
    """Position each endpoint near its purview's shell point.

    Cause endpoints shift -x and effect endpoints +x by
    ``direction_offset``; endpoints sharing a purview and direction spread
    on a small polygon of radius ``purview_jitter``.
    """
    base = _shell_positions((e.purview for e in projection.endpoints), geometry)
    groups: dict[tuple[tuple[int, ...], str], list[int]] = defaultdict(list)
    for e in projection.endpoints:
        groups[(e.purview, e.direction)].append(e.id)
    positions: dict[int, Point] = {}
    for (purview, direction), ids in groups.items():
        bx, by, bz = base[purview]
        bx += (
            geometry.direction_offset
            if direction == "effect"
            else -geometry.direction_offset
        )
        jitter = geometry.purview_jitter if len(ids) > 1 else 0.0
        offsets = _polygon_points(len(ids), jitter, 0.0)
        for eid, (ox, oy, _) in zip(sorted(ids), offsets, strict=True):
            positions[eid] = (bx + ox, by + oy, bz)
    return positions


def _mechanism_positions(
    projection: PhiStructureProjection, geometry: SimplicialComplexGeometry
) -> dict[int, Point]:
    """Position each distinction's mechanism on its size shell."""
    base = _shell_positions((n.mechanism for n in projection.nodes), geometry)
    return {n.id: base[n.mechanism] for n in projection.nodes}
