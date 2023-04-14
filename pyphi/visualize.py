import string
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations
from math import cos, isclose, log2, radians, sin
from typing import Callable, Mapping, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import plotly.colors
import scipy.special
import seaborn as sb
from _plotly_utils.basevalidators import ColorscaleValidator
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from plotly import graph_objs as go
from toolz import partition

import pyphi

from .conf import config
from .direction import Direction
from .models.subsystem import CauseEffectStructure
from .new_big_phi import PhiStructure
from .relations import ConcreteRelations
from .utils import state_of

# TODO(viz): Update to use new relations


def two_relation_face_type(relation_face):
    if len(relation_face) != 2:
        raise ValueError(f"must be a 2-relation; got a {len(relation_face)}-relation")
    purview = list(map(set, relation_face.relata_purviews))
    # Isotext (mutual full-overlap)
    if purview[0] == purview[1] == relation_face.purview:
        return "isotext"
    # Sub/Supertext (inclusion / full-overlap)
    elif purview[0].issubset(purview[1]) or purview[0].issuperset(purview[1]):
        return "inclusion"
    # Paratext (connection / partial-overlap)
    else:
        return "paratext"


TWOPI = 2 * np.pi
FONT_FAMILY = "MesloLGS NF, Roboto Mono, Menlo"


@dataclass
class PhiPlotTheme:
    """Specifies plot attributes."""

    fontsize: int = 12
    direction_offset: float = 0.5
    purview_offset_radius: float = 0.1
    cause_color: str = "#e21a1a"
    effect_color: str = "#14b738"
    point_size_range: tuple = (5, 30)
    distinction_colorscale: str = "mint"
    distinction_opacity_range: tuple = (0.1, 0.9)
    line_width_range: tuple = (3, 10)
    cause_effect_link_color: str = "lightgrey"
    cause_effect_link_opacity: float = 0.5
    mechanism_purview_link_color: str = "lightgrey"
    mechanism_purview_link_opacity: float = 0.5
    mechanism_max_radius: float = 1.0
    mechanism_z_offset: float = 0.0
    mechanism_z_spacing: float = 0.0
    mechanism_radius_func: Union[Callable, str] = "linear"
    purview_radius_mod: float = 1.0
    two_relation_colorscale: Union[str, Callable, Mapping] = "type"
    two_relation_opacity: float = 0.2
    two_relations_hoverlabel_font_color: str = "white"
    three_relation_colorscale: str = "teal"
    three_relation_opacity: float = 0.1
    three_relation_opacity_range: Optional[tuple] = None
    three_relation_intensity_range: tuple = (0, 1)
    three_relation_showlegend: bool = True
    three_relation_showscale: bool = True
    lighting: Mapping = field(
        default_factory=lambda: dict(
            ambient=0.8, diffuse=0, roughness=0, specular=0, fresnel=0
        )
    )
    legendgroup_postfix: str = ""


GREYS = PhiPlotTheme(
    distinction_colorscale="greys",
    distinction_opacity_range=(0.1, 0.2),
    # cause_effect_link_color="grey",
    cause_effect_link_opacity=0.1,
    # mechanism_purview_link_color="grey",
    mechanism_purview_link_opacity=0.1,
    two_relation_colorscale="greys",
    two_relation_opacity=0.1,
    three_relation_colorscale="greys",
    three_relation_opacity=0.05,
    three_relation_intensity_range=(0, 0.5),
    # three_relation_showlegend=True,
    legendgroup_postfix=" (greyed)",
)


_TYPE_COLORS = {"isotext": "magenta", "inclusion": "indigo", "paratext": "cyan"}


def type_color(relation_face):
    return _TYPE_COLORS[two_relation_face_type(relation_face)]


TWO_RELATION_COLORSCHEMES = {"type": type_color}


def rgb_to_rgba(color, alpha=0):
    """Return an RGBA color string from an RGB color string."""
    channels = plotly.colors.unlabel_rgb(color)
    channels = tuple(round(c, 7) for c in channels)
    channels += (alpha,)
    return f"rgba{channels}"


def _get_color(colorscale, intermed):
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    color = plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
    return rgb_to_rgba(color)


def get_color(colorscale, loc):
    """Return the interpolated color at `loc` using the given colorscale."""
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale)

    # convert to rgb strings
    locs, colors = zip(*colorscale)
    colors, _ = plotly.colors.convert_colors_to_same_type(colors)
    colorscale = list(zip(locs, colors))

    if isinstance(loc, Iterable):
        return [_get_color(colorscale, x) for x in loc]
    return _get_color(colorscale, loc)


def indent(lines, amount=2, char="&nbsp;", newline="<br>"):
    """Indent a string."""
    lines = str(lines)
    padding = amount * char
    return padding + (newline + padding).join(lines.split(newline))


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


def regular_polygon(n, radius=1.0, center=(0, 0), z=0, angle=0):
    angles = (TWOPI / n) * np.arange(n) - angle
    points = np.empty([n, 3])
    points[:, 0] = center[0] + radius * np.sin(angles)
    points[:, 1] = center[1] + radius * np.cos(angles)
    points[:, 2] = z
    return points


def rescale(values, target_range):
    _min = values.min()
    _max = values.max()
    if isclose(_min, _max, rel_tol=1e-9, abs_tol=1e-9):
        x = np.ones(len(values)) * np.mean(target_range)
        return x
    return target_range[0] + (
        ((values - _min) * (target_range[1] - target_range[0])) / (_max - _min)
    )


# Radius scaling function
def log_n_choose_k(N):
    if N == 1:
        return np.array([1.0])
    return np.log(scipy.special.binom(N, np.arange(1, N + 1)))


# Radius scaling function
def linear(N):
    return np.arange(N)[::-1]


SHAPES = {"linear": linear, "log_n_choose_k": log_n_choose_k}


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
                )
            )
        )
    return mapping


def offset_mapping(mapping, offset):
    """Offset each coordinate in a mapping."""
    return {key: value + offset for key, value in mapping.items()}


def make_layout(width=900, aspect=1.62, eye=None, fontsize=12):
    if eye is not None:
        eye = dict(zip("xyz", spherical_to_cartesian(eye)))
    height = width / aspect
    return go.Layout(
        scene={
            name: dict(
                showbackground=False,
                showgrid=False,
                showticklabels=False,
                showspikes=False,
                title="",
            )
            for name in ["xaxis", "yaxis", "zaxis"]
        },
        scene_camera_eye=eye,
        autosize=True,
        showlegend=True,
        hovermode="x",
        hoverlabel_font=dict(family=FONT_FAMILY, size=int(0.75 * fontsize)),
        title="",
        width=width,
        height=height,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
    )


class Labeler:
    def __init__(self, subsystem):
        self.subsystem = subsystem

    def nodes(self, nodes, state=None):
        if state is None:
            state = state_of(nodes, self.subsystem.state)
        # print(
        #     "".join(
        #         f"{n.upper() if state[i] else n.lower()}, {n}"
        #         for i, n in enumerate(
        #             self.subsystem.node_labels.coerce_to_labels(nodes)
        #         )
        #     )
        # )
        return "".join(
            n.upper() if state[i] else n.lower()
            for i, n in enumerate(self.subsystem.node_labels.coerce_to_labels(nodes))
        )

    def hover(self, mice):
        return f"<br>".join(
            [
                f"Distinction ({mice.direction})",
                indent(
                    "<br>".join(
                        [
                            f"M: {self.nodes(mice.mechanism)}",
                            f"P: {self.nodes(mice.purview, state=mice.specified_state)}",
                            f"φ: {round(mice.phi, config.PRECISION)}",
                            f"S: {','.join(map(str, mice.specified_state))}",
                        ]
                    )
                ),
            ]
        )

    def mice(self, mice):
        return f"{self.nodes(mice.purview, state=mice.specified_state)}"

    def relata(self, relata):
        return "<br>".join(map(self.mice, relata))

    def relation(self, relation):
        return f"{len(relation)}-relation<br>" + indent(
            "<br>".join(
                [
                    f"P: {self.nodes(relation.purview)}",
                    f"φ: {round(relation.phi, config.PRECISION)}",
                    "Relata:",
                    indent(self.relata(relation.relata)),
                ]
            )
        )


def scatter_from_coords(coords, labels=None, fontsize=12, **kwargs):
    """Return a Scatter3d given labels and coordinates."""
    x, y, z = np.stack(coords).transpose()
    defaults = dict(
        mode="text",
        text=labels,
        textposition="middle center",
        textfont=dict(family=FONT_FAMILY, size=fontsize),
        hoverinfo="text",
        showlegend=True,
    )
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        **{**defaults, **kwargs},
    )


def _plot_distinctions(
    fig,
    distinctions,
    purview_mapping,
    label,
    theme,
):
    for direction, color in zip(
        Direction.both(), [theme.cause_color, theme.effect_color]
    ):
        coords = []
        labels = []
        _distinctions = []
        for distinction in distinctions:
            _distinctions.append(distinction.mice(direction))
            purview = distinction.purview(direction)
            coords.append(purview_mapping[direction][purview])
            labels.append(label.nodes(purview))

        _distinctions = CauseEffectStructure(_distinctions)
        hovertext = [label.hover(distinction) for distinction in _distinctions]
        text = [label.mice(distinction) for distinction in _distinctions]
        phis = np.array(list(_distinctions.phis))
        scaled_phis = rescale(phis, (0, 1))
        opacities = rescale(phis, theme.distinction_opacity_range)
        marker_size = rescale(phis, theme.point_size_range)
        marker_colors = [
            rgb_to_rgba(get_color(theme.distinction_colorscale, loc), alpha=opacity)
            for loc, opacity in zip(scaled_phis, opacities)
        ]
        fig.add_trace(
            scatter_from_coords(
                coords,
                labels=labels,
                name=f"{direction} distinctions" + theme.legendgroup_postfix,
                hovertext=hovertext,
                text=text,
                hoverlabel_bgcolor=color,
                textfont_color=color,
                opacity=0.99,
                mode="text+markers",
                marker=dict(symbol="circle", color=marker_colors, size=marker_size),
                fontsize=theme.fontsize,
            )
        )


def _plot_cause_effect_links(
    fig,
    distinctions,
    purview_mapping,
    theme,
):
    # TODO make this scaling consistent with 2-relation phi?
    name = "Cause-effect links" + theme.legendgroup_postfix
    widths = rescale(np.array(list(distinctions.phis)), theme.line_width_range)
    showlegend = True
    link_coords = []
    for distinction, width in zip(distinctions, widths):
        coords = np.stack(
            [
                purview_mapping[direction][distinction.purview(direction)]
                for direction in Direction.both()
            ]
        )
        link_coords.append(coords)
        x, y, z = coords.transpose()
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                showlegend=showlegend,
                legendgroup=name + theme.legendgroup_postfix,
                name=name,
                mode="lines",
                line_color=theme.cause_effect_link_color,
                opacity=theme.cause_effect_link_opacity,
                line_width=width,
                hoverinfo="skip",
            )
        )
        showlegend = False
    return link_coords


def _plot_mechanisms(fig, distinctions, mechanism_mapping, label, theme):
    name = "Mechanisms" + theme.legendgroup_postfix
    labels = []
    coords = []
    for mechanism in distinctions.mechanisms:
        labels.append(label.nodes(mechanism))
        coords.append(mechanism_mapping[mechanism])
    fig.add_trace(
        scatter_from_coords(
            coords,
            labels=labels,
            legendgroup=name + theme.legendgroup_postfix,
            name=name,
            hoverinfo="skip",
            fontsize=theme.fontsize,
        )
    )


def _plot_mechanism_purview_links(
    fig, distinctions, cause_effect_link_coords, mechanism_mapping, theme
):
    name = "Mechanism-purview links" + theme.legendgroup_postfix
    # TODO make this scaling consistent with 2-relation phi?
    widths = rescale(np.array(list(distinctions.phis)), theme.line_width_range)
    showlegend = True
    for distinction, width, coords in zip(
        distinctions, widths, cause_effect_link_coords
    ):
        coords = np.stack(
            [coords[0], mechanism_mapping[distinction.mechanism], coords[1]]
        )
        x, y, z = coords.transpose()
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                showlegend=showlegend,
                legendgroup=name + theme.legendgroup_postfix,
                name=name,
                mode="lines",
                line_color=theme.mechanism_purview_link_color,
                opacity=theme.mechanism_purview_link_opacity,
                line_width=width,
                hoverinfo="skip",
            )
        )
        showlegend = False


def _plot_two_relation_faces(fig, relation_to_coords, relation_faces, label, theme):
    name = "2-relations" + theme.legendgroup_postfix
    phis = np.array(list(phi for _, phi in relation_faces))
    widths = rescale(phis, theme.line_width_range)
    faces = [face for face, _ in relation_faces]

    if isinstance(theme.two_relation_colorscale, Mapping):
        # Map to relation type
        line_colors = list(
            map(
                theme.two_relation_colorscale.get,
                map(two_relation_face_type, faces),
            )
        )
    elif (
        isinstance(theme.two_relation_colorscale, str)
        and theme.two_relation_colorscale not in TWO_RELATION_COLORSCHEMES
    ):
        # Plotly colorscale
        scaled_phis = rescale(phis, (0, 1))
        line_colors = [
            get_color(theme.two_relation_colorscale, phi) for phi in scaled_phis
        ]
    elif theme.two_relation_colorscale in TWO_RELATION_COLORSCHEMES:
        # Library function
        line_colors = list(
            map(TWO_RELATION_COLORSCHEMES[theme.two_relation_colorscale], faces)
        )
    else:
        # Callable
        line_colors = list(map(theme.two_relation_colorscale, faces))

    showlegend = True
    for face, width, line_color in zip(
        faces,
        widths,
        line_colors,
    ):
        x, y, z = relation_to_coords(face).transpose()
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                showlegend=showlegend,
                legendgroup=name + theme.legendgroup_postfix,
                name=name,
                mode="lines",
                line_color=line_color,
                opacity=theme.two_relation_opacity,
                line_width=width,
                hoverinfo="text",
                # hovertext=label.relation(faces),
                # hoverlabel_font_color=theme.two_relations_hoverlabel_font_color,
            )
        )
        # Only show the first trace in the legend
        showlegend = False


def _plot_three_relation_faces(fig, relation_to_coords, relation_faces, label, theme):
    name = "3-relations" + theme.legendgroup_postfix
    # Build vertices:
    # Stack the [relation, relata] axes together and tranpose to put the 3D axis
    # first to get lists of x, y, z coordinates
    x, y, z = np.vstack(
        list(map(relation_to_coords, [face for face, _ in relation_faces]))
    ).transpose()
    # Build triangles:
    # The vertices are stacked triples, so we want each (i, j, k) = [0, 1, 2], [3, 4, 5], ...
    relata_indices = np.arange(len(relation_faces) * 3, step=3)
    i, j, k = np.tile(relata_indices, (3, 1)) + np.arange(3).reshape(3, 1)
    phis = np.array(list(phi for _, phi in relation_faces))
    intensities = rescale(phis, theme.three_relation_intensity_range)
    # hovertext = list(map(label.relation, relation_faces))
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            showlegend=theme.three_relation_showlegend,
            legendgroup=name + theme.legendgroup_postfix,
            name=name,
            intensity=intensities,
            intensitymode="cell",
            colorscale=theme.three_relation_colorscale,
            showscale=theme.three_relation_showscale,
            colorbar=dict(
                title=dict(text="φ", font_size=2 * theme.fontsize),
                x=0.0,
                len=1.0,
            ),
            opacity=theme.three_relation_opacity,
            lighting=theme.lighting,
            hoverinfo="text",
            # hovertext=hovertext,
        )
    )


def _plot_three_relation_faces_with_opacity(
    fig, relation_to_coords, relation_faces, label, theme
):
    name = "3-relations" + theme.legendgroup_postfix
    # Build vertices:
    # Stack the [relation, relata] axes together and tranpose to put the 3D axis
    # first to get lists of x, y, z coordinates
    x, y, z = np.vstack(
        list(map(relation_to_coords, [face for face, _ in relation_faces]))
    ).transpose()
    phis = np.array(list(phi for _, phi in relation_faces))
    intensities = rescale(phis, theme.three_relation_intensity_range)
    opacities = rescale(phis, theme.three_relation_opacity_range)
    # hovertexts = list(map(label.relation, relation_faces))
    showlegend = theme.three_relation_showlegend
    showscale = theme.three_relation_showscale
    for _x, _y, _z, intensity, opacity in zip(
        partition(3, x),
        partition(3, y),
        partition(3, z),
        intensities,
        opacities,
        # hovertexts,
    ):
        fig.add_trace(
            go.Mesh3d(
                x=_x,
                y=_y,
                z=_z,
                i=[0],
                j=[1],
                k=[2],
                showlegend=showlegend,
                legendgroup=name + theme.legendgroup_postfix,
                name=name,
                intensity=[intensity],
                intensitymode="cell",
                colorscale=theme.three_relation_colorscale,
                showscale=showscale,
                colorbar=dict(
                    title=dict(text="φ", font_size=2 * theme.fontsize),
                    x=0.0,
                    len=1.0,
                ),
                opacity=opacity,
                hoverinfo="text",
                # hovertext=hovertext,
                lighting=theme.lighting,
            )
        )
        showlegend = False
        showscale = False


# TODO
# - 4-relations?
# - think about configuration for visual attributes; seems it can be easily
#   done post-hoc by user on the figure object
def plot_phi_structure(
    phi_structure,
    node_indices=None,
    fig=None,
    shape="log_n_choose_k",
    theme=None,
    system_size=None,
):
    """Plot a PhiStructure.

    Keyword Arguments:
        fig (plotly.graph_objects.Figure): The figure to use. Defaults to None,
            which creates a new figure.
        shape (string | Callable): Function determining the shape of the
            structure; specifically, how the radii scale with purview order. Can
            be the name of an existing function in the SHAPES dictionary, or a
            user-supplied function.
        plot_mechanisms (bool): Whether to plot mechanisms.
    """
    if not isinstance(phi_structure, PhiStructure):
        raise ValueError(
            f"phi_structure must be a PhiStructure; got {type(phi_structure)}"
        )
    if not phi_structure.distinctions:
        raise ValueError("No distinctions; cannot plot")

    if theme is None:
        theme = PhiPlotTheme()

    if fig is None:
        fig = go.Figure()
    fig.update_layout(make_layout(fontsize=theme.fontsize))

    # Use named shape function if available; otherwise assume `shape` is a function and use it
    radius_func = SHAPES.get(shape, shape)

    distinctions = phi_structure.distinctions
    subsystem = distinctions.subsystem
    if node_indices is None:
        node_indices = subsystem.node_indices

    label = Labeler(subsystem)

    # x offsets for causes and effects
    direction_offset = dict(
        zip(Direction.both(), [-theme.direction_offset / 2, theme.direction_offset / 2])
    )

    # Base purview coordinates
    purview_mapping_base = {
        direction: powerset_coordinates(
            node_indices,
            x_offset=direction_offset[direction],
            radius_func=radius_func,
            purview_radius_mod=theme.purview_radius_mod,
        )
        for direction in Direction.both()
    }

    # Since there can be different distinctions that have the same purview on
    # one side, and there can be relation faces among those copies of the same
    # purview, we offset each distinction's purview so they don't overlap.
    purview_offset_mapping = dict(
        zip(
            phi_structure.distinctions.mechanisms,
            regular_polygon(
                len(phi_structure.distinctions), radius=theme.purview_offset_radius
            ),
        ),
        strict=True,
    )
    purview_multiplicities = {
        direction: Counter(distinctions.purviews(direction))
        for direction in Direction.both()
    }
    purview_mapping = {
        direction: {
            distinction.mechanism: offset_mapping(
                purview_mapping_base[direction],
                (
                    purview_offset_mapping[distinction.mechanism]
                    if purview_multiplicities[direction][distinction.purview(direction)]
                    > 1
                    else 0
                ),
            )
            for distinction in phi_structure.distinctions
        }
        for direction in Direction.both()
    }

    # Distinctions
    _plot_distinctions(
        fig,
        distinctions,
        purview_mapping_base,
        label,
        theme,
    )

    # Cause-effect links
    cause_effect_link_coords = _plot_cause_effect_links(
        fig,
        distinctions,
        purview_mapping_base,
        theme,
    )

    mechanism_mapping = powerset_coordinates(
        node_indices,
        max_radius=theme.mechanism_max_radius,
        z_offset=theme.mechanism_z_offset,
        z_spacing=theme.mechanism_z_spacing,
        radius_func=theme.mechanism_radius_func,
        # purview_radius_mod=1,
    )
    # Mechanisms
    _plot_mechanisms(fig, distinctions, mechanism_mapping, label, theme)
    # Mechanism-purview links
    _plot_mechanism_purview_links(
        fig, distinctions, cause_effect_link_coords, mechanism_mapping, theme
    )

    # Group relations by degree
    relations = defaultdict(set)
    for relation in phi_structure.relations:
        for face in relation.faces:
            relations[len(face)].add((face, relation.phi))

    def face_to_coords(face):
        return np.array(
            [
                purview_mapping[relatum.direction][relatum.mechanism][relatum.purview]
                for relatum in face
            ]
        )

    # 2-relations
    if relations[2]:
        _plot_two_relation_faces(
            fig,
            face_to_coords,
            relations[2],
            label,
            theme,
        )

    # 3-relations
    if relations[3]:
        if theme.three_relation_opacity_range is None:
            _plot_three_relation_faces(fig, face_to_coords, relations[3], label, theme)
        else:
            _plot_three_relation_faces_with_opacity(
                fig, face_to_coords, relations[3], label, theme
            )

    return fig


###############################################################################
# Connectivity
###############################################################################


NODE_COLORS = {
    # (in subsystem, state)
    (False, 0): "lightgrey",
    (False, 1): "darkgrey",
    (True, 0): "lightblue",
    (True, 1): "darkblue",
}


def plot_graph(g, **kwargs):
    kwargs = {
        **dict(
            with_labels=True,
            arrowsize=20,
            node_size=600,
            font_color="white",
        ),
        **kwargs,
    }
    nx.draw(
        g,
        **kwargs,
    )


def plot_subsystem(subsystem, **kwargs):
    g = nx.from_numpy_matrix(subsystem.cm, create_using=nx.DiGraph)
    nx.relabel_nodes(
        g, dict(zip(range(subsystem.network.size), subsystem.node_labels)), copy=False
    )
    if "node_color" not in kwargs:
        kwargs["node_color"] = [
            NODE_COLORS[(i in subsystem.node_indices, subsystem.state[i])]
            for i in range(subsystem.network.size)
        ]
    plot_graph(g, **kwargs)
    return g


###############################################################################
# Distributions
###############################################################################


def all_states_str(*args, **kwargs):
    """Return all states as bit strings."""
    for state in pyphi.utils.all_states(*args, **kwargs):
        yield "".join(map(str, state))


def _plot_distribution_bar(data, ax, label, **kwargs):
    sb.barplot(data=data, x="state", y="probability", ax=ax, **kwargs)

    plt.xticks(rotation=90, ha="center", va="top")
    # Add state label
    xtick_pad = 6
    xtick_length = 6
    ax.tick_params(axis="x", pad=xtick_pad, length=xtick_length)
    ax.annotate(
        str(label) if label is not None else "",
        xy=(-0.5, 0),
        xycoords="data",
        xytext=(0, -(xtick_pad + xtick_length)),
        textcoords="offset points",
        annotation_clip=False,
        rotation=90,
        ha="right",
        va="top",
    )

    return ax


def _plot_distribution_line(data, ax, **kwargs):
    sb.lineplot(data=data, x="state", y="probability", ax=ax, **kwargs)
    return ax


def plot_distribution(
    *distributions,
    states=None,
    label=None,
    figsize=(9, 3),
    fig=None,
    ax=None,
    lineplot_threshold=64,
    title="State distribution",
    y_label="Pr(state)",
    validate=True,
    labels=None,
    **kwargs,
):
    """Plot a distribution over states.

    Arguments:
        d (array_like): The distribution. If no states are provided, must
            have length equal to a power of 2. Multidimensional distributions
            are flattened with ``pyphi.distribution.flatten()``.

    Keyword Arguments:
        states (Iterable | None): The states corresponding to the
            probabilities in the distribution; if ``None``, infers states from
            the length of the distribution and assumes little-endian ordering.
        **kwargs: Passed to ``sb.barplot()``.
    """
    if validate and not all(np.allclose(d.sum(), 1, rtol=1e-4) for d in distributions):
        raise ValueError("a distribution does not sum to 1!")

    defaults = dict()
    # Overrride defaults with keyword arguments
    kwargs = {**defaults, **kwargs}

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    distributions = [pd.Series(pyphi.distribution.flatten(d)) for d in distributions]
    d = distributions[0]

    if validate and not all(
        (distributions[0].index == d.index).all() for d in distributions
    ):
        raise ValueError("distribution indices do not match")

    N = log2(np.prod(d.shape))
    if states is None:
        if N.is_integer() and len(d) <= lineplot_threshold:
            N = int(N)
            states = list(all_states_str(N))
            if label is None:
                label = string.ascii_uppercase[:N]
        else:
            states = np.arange(len(d))

    if labels is None:
        labels = list(map(str, range(len(distributions))))

    data = pd.concat(
        [
            pd.DataFrame(dict(probability=d, state=states, hue=[label] * len(d)))
            for d, label in zip(distributions, labels)
        ]
    ).reset_index(drop=True)

    if len(d) > lineplot_threshold:
        ax = _plot_distribution_line(data, ax, hue="hue", **kwargs)
    else:
        ax = _plot_distribution_bar(data, ax, label, hue="hue", **kwargs)

    ax.set_title(title)
    ax.set_ylabel(y_label, labelpad=12)
    ax.set_xlabel("state", labelpad=12)
    ax.legend(bbox_to_anchor=(1.1, 1.05))

    return fig, ax


def plot_repertoires(subsystem, sia, **kwargs):
    if config.REPERTOIRE_DISTANCE != "GENERALIZED_INTRINSIC_DIFFERENCE":
        raise NotImplementedError(
            "Only REPERTOIRE_DISTANCE = "
            "GENERALIZED_INTRINSIC_DIFFERENCE is supported"
        )
    cut_subsystem = subsystem.apply_cut(sia.partition)

    labels = ["unpartitioned", "partitioned"]
    subsystems = dict(zip(labels, [subsystem, cut_subsystem]))
    repertoires = {
        direction: {
            label: s.forward_repertoire(direction, s.node_indices, s.node_indices)
            for label, s in subsystems.items()
        }
        for direction in Direction.both()
    }

    fig = plt.figure(figsize=(12, 9))
    axes = fig.subplots(2, 1)
    for ax, direction in zip(axes, Direction.both()):
        plot_distribution(
            repertoires[direction][labels[0]],
            repertoires[direction][labels[1]],
            validate=False,
            title=str(direction),
            labels=labels,
            ax=ax,
            **kwargs,
        )
    fig.tight_layout(h_pad=0.5)
    for ax in axes:
        ax.legend(bbox_to_anchor=(1.1, 1.1))
    return fig, axes, repertoires


def plot_dynamics(data: ArrayLike, node_labels=None, title=""):
    """Plot an array of states over time.

    Arguments:
        data (ArrayLike): An array of states with shape (timesteps, units).
    """
    # Plot time horizontally
    data = np.transpose(data)
    fig = plt.figure(figsize=(25, 5))
    ax = plt.imshow(data, aspect="auto", interpolation="none", vmin=0, vmax=1)
    plt.grid(False)
    plt.title(title)
    plt.ylabel("Substrate state")
    plt.xlabel("Time")
    if node_labels is not None:
        plt.yticks(range(len(node_labels)), node_labels)
    plt.colorbar()
    plt.show()
    return fig, ax
