# visualize/phi_structure/geometry.py
"""Utilities for specifying the spatial layout of |big_phi|-structures."""

from itertools import combinations
from math import cos, radians, sin
from typing import Mapping, Optional

import numpy as np
import scipy.special
from numpy.typing import ArrayLike

from ...direction import Direction

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
        direction_offset: Optional[float] = None,
        subset_multiplicities: Optional[float] = None,
        subset_offset_radius: Optional[float] = 0.0,
        state_multiplicities: Optional[float] = None,
        state_offset_radius: Optional[float] = 0.0,
        rotation: Optional[float] = 0.0,
        rotation_plane: Optional[str] = "xy",
        scale: Optional[ArrayLike] = 1.0,
        translate: Optional[ArrayLike] = 0.0,
    ):
        self.mapping = mapping

        if direction_offset is not None:
            self.direction_offset = _direction_offset_mapping(direction_offset)
        else:
            self.direction_offset = None

        self._subset_offset = (
            subset_offset_radius != 0 and subset_multiplicities is not None
        )
        if self._subset_offset:
            self.subset_offset_mapping = _multiplicity_mapping(
                subset_multiplicities, subset_offset_radius
            )

        self._state_offset = (
            state_offset_radius != 0.0 and state_multiplicities is not None
        )
        if self._state_offset:
            self.state_offset_mapping = _multiplicity_mapping(
                state_multiplicities, state_offset_radius
            )

        self.rotation_amount = rotation
        self.rotation_plane = rotation_plane

        self.scale = scale
        self.translate = translate

    def get(
        self,
        subset: tuple[int],
        direction: Direction = None,
        offset_subset: tuple[int] = None,
        offset_state: tuple[int] = None,
    ):
        """Return coordinates for the given subset."""
        coords = self.mapping[subset].copy()

        if direction is not None and self.direction_offset is not None:
            coords += self.direction_offset[direction]

        if offset_subset is not None and self._subset_offset:
            coords += self.subset_offset_mapping[subset][offset_subset]

        if offset_state is not None and self._state_offset:
            coords += self.state_offset_mapping[subset][offset_state]

        coords *= self.scale
        coords += self.translate

        if self.rotation_amount != 0:
            coords = rotate(coords, self.rotation_amount, self.rotation_plane)

        return coords


def _direction_offset_mapping(direction_offset_amount: float):
    direction_offset = np.zeros(3)
    direction_offset[0] = direction_offset_amount / 2
    return {
        Direction.CAUSE: -direction_offset,
        Direction.EFFECT: direction_offset,
    }


def _multiplicity_mapping(multiplicities, radius, **kwargs):
    return {
        subset: dict(
            zip(
                sorted(multiples),
                regular_polygon(len(multiples), radius=radius, **kwargs),
            )
        )
        for subset, multiples in multiplicities.items()
    }


# def _subset_offset_mapping(offset_subsets, subset_offset_radius):
#     offset_subsets = list(offset_subsets)
#     return dict(
#         zip(
#             offset_subsets,
#             regular_polygon(len(offset_subsets), radius=subset_offset_radius),
#             strict=True,
#         )
#     )


def rotate(coordinates, theta, plane):
    """Return the coordinates rotated theta degrees in the specified plane."""
    if plane == "xy":
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
    elif plane == "yz":
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
    elif plane == "xz":
        rotation_matrix = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
    else:
        raise ValueError("Invalid plane specified. Must be one of 'xy', 'yz', or 'xz'.")
    return np.dot(coordinates, rotation_matrix)


def arrange(
    nodes,
    max_radius=1.0,
    aspect_ratio=0.5,
    z_offset=0.0,
    z_spacing=1.0,
    x_offset=0.0,
    y_offset=0.0,
    N=None,
    radius_func=log_n_choose_k,
):
    """Return a mapping from subsets of the nodes to coordinates."""
    radius_func = SHAPES.get(radius_func, radius_func)
    if N is None:
        N = len(nodes)
    radii = radius_func(N)
    # Normalize overall radius
    radii = radii * max_radius / radii.max()
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


def center_coords(coords):
    """Center coordinates around the origin."""
    return coords - coords.mean(axis=0)
