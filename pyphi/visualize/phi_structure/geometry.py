# visualize/phi_structure/geometry.py

"""Geometric functions for plotting."""

from itertools import combinations
from math import cos, radians, sin
from typing import Mapping, Optional

import numpy as np
import scipy.special
from numpy.typing import ArrayLike

from ...direction import Direction
from ...utils import powerset

TWOPI = 2 * np.pi


# Radius scaling function
def log_n_choose_k(N):
    if N == 1:
        return np.array([1.0])
    return np.log(scipy.special.binom(N, np.arange(1, N + 1)))


# Radius scaling function
def linear(N):
    return np.arange(N)[::-1]


SHAPES = {"linear": linear, "log_n_choose_k": log_n_choose_k}


class Coordinates:
    """Map subsets to 3D coordinates."""

    def __init__(
        self,
        mapping: Mapping[tuple[int], ArrayLike],
        direction_offset_amount: Optional[float] = None,
        subset_offset_radius: Optional[float] = None,
        offset_subsets: Optional[float] = None,
    ):
        self.mapping = mapping
        if direction_offset_amount is not None:
            self.direction_offset = _direction_offset_mapping(direction_offset_amount)
        else:
            self.direction_offset = None
        if subset_offset_radius:
            self.subset_offset_mapping = _subset_offset_mapping(
                offset_subsets, subset_offset_radius
            )
        else:
            self.subset_offset_mapping = None

    def get(
        self,
        subset: tuple[int],
        offset_subset: tuple[int] = None,
        direction: Direction = None,
    ):
        """Return coordinates for the given subset."""
        coords = self.mapping[subset].copy()
        if offset_subset is not None and self.subset_offset_mapping is not None:
            coords += self.subset_offset_mapping[offset_subset]
        if direction is not None and self.direction_offset is not None:
            coords += self.direction_offset[direction]
        return coords


def _direction_offset_mapping(direction_offset_amount: float):
    direction_offset = np.zeros(3)
    direction_offset[0] = direction_offset_amount / 2
    return {
        Direction.CAUSE: -direction_offset,
        Direction.EFFECT: direction_offset,
    }


def _subset_offset_mapping(offset_subsets, subset_offset_radius):
    offset_subsets = list(offset_subsets)
    return dict(
        zip(
            offset_subsets,
            regular_polygon(len(offset_subsets), radius=subset_offset_radius),
            strict=True,
        )
    )


def powerset_coordinates(
    nodes,
    max_radius=1.0,
    aspect_ratio=0.5,
    z_offset=0.0,
    z_spacing=1.0,
    x_offset=0.0,
    y_offset=0.0,
    radius_func=log_n_choose_k,
    purview_radius_mod=1,
):
    """Return a mapping from subsets of the nodes to coordinates."""
    radius_func = SHAPES.get(radius_func) or radius_func
    N = len(nodes)
    radii = radius_func(N)
    # Normalize overall radius
    radii = radii * max_radius / radii.max() * purview_radius_mod
    z = aspect_ratio * max_radius * np.cumsum(np.ones(N) * z_spacing) + z_offset
    mapping = dict()
    for k in range(N):
        # TODO: sort?? order determines a lot about how the shape looks
        subsets = sorted(combinations(nodes, k + 1))
        mapping.update(
            dict(
                zip(
                    subsets,
                    regular_polygon(
                        len(subsets),
                        radius=radii[k],
                        center=(x_offset, y_offset),
                        z=z[k],
                    ),
                    strict=True,
                )
            )
        )
    return mapping


def regular_polygon(n, radius=1.0, center=(0, 0), z=0, angle=0):
    angles = (TWOPI / n) * np.arange(n) - angle
    points = np.empty([n, 3])
    points[:, 0] = center[0] + radius * np.sin(angles)
    points[:, 1] = center[1] + radius * np.cos(angles)
    points[:, 2] = z
    return points


def spherical_to_cartesian(coords):
    """Convert spherical coordinates (in degrees) to Cartesian coordinates.

    Spherical coordinates are expected in the order (radius, polar, azimuth).
    """
    radius, polar, azimuth = map(radians, coords)
    return (
        radius * sin(polar) * cos(azimuth),
        radius * sin(polar) * sin(azimuth),
        radius * cos(polar),
    )
