# visualize/phi_structure/__init__.py

from typing import Callable, Iterable

import numpy as np
from plotly import graph_objs as go
from plotly.colors import find_intermediate_color, validate_colors
from toolz import partition

from ...direction import Direction
from . import colors, geometry, text, utils
from .colors import get_color
from .theme import DefaultTheme, Grey, Theme

DEFAULT_THEME = DefaultTheme()
GREY_THEME = Grey()


def highlight_phi_fold(
    phi_fold,
    phi_structure,
    node_indices=None,
    state=None,
    node_labels=None,
    fig=None,
    highlight_theme=DEFAULT_THEME,
    background_theme=GREY_THEME,
    **theme_overrides,
):
    """Plot a PhiStructure with a PhiFold highlighted."""
    fig, purview_coords, mechanism_coords = plot_phi_structure(
        phi_structure,
        fig=fig,
        theme=background_theme,
        return_coords=True,
        node_indices=node_indices,
        state=state,
        node_labels=node_labels,
        **theme_overrides,
    )
    fig = plot_phi_structure(
        phi_fold,
        fig=fig,
        theme=highlight_theme,
        purview_coords=purview_coords,
        mechanism_coords=mechanism_coords,
        node_indices=node_indices,
        state=state,
        node_labels=node_labels,
        **theme_overrides,
    )
    return fig


def plot_phi_structure(
    phi_structure=None,
    distinctions=None,
    relations=None,
    subsystem=None,
    state=None,
    node_indices=None,
    node_labels=None,
    fig=None,
    theme=DEFAULT_THEME,
    mechanism_coords=None,
    purview_coords=None,
    return_coords=False,
    **theme_overrides,
):
    """Plot a PhiStructure.

    Arguments:
        phi_structure (PhiStructure): The PhiStructure to plot.

    Keyword Arguments:
        fig (plotly.graph_objects.Figure): The figure to use. Defaults to None,
            which creates a new figure.
        theme (Theme): The visual theme to use.
        purview_coords (Coordinates): Coordinates to use when arranging
            purviews. Defaults to generating coordinates according to the theme.
        mechanism_coords (Coordinates): Coordinates to use when arranging
            mechanisms. Defaults to generating coordinates according to the theme.
        node_indices (tuple[int]): The node indices to use when arranging
            purviews. Defaults to the subsystem's node indices.
        **theme_overrides (Mapping): Overrides for the theme.
    """
    if phi_structure is None and (distinctions is None or relations is None):
        raise ValueError(
            "Either phi_structure or both distinctions and relations are required"
        )
    if distinctions is None:
        distinctions = phi_structure.distinctions
    if not distinctions:
        raise ValueError("No distinctions; cannot plot")
    if relations is None:
        relations = phi_structure.relations

    if subsystem is None and any(
        variable is None for variable in [state, node_indices, node_labels]
    ):
        raise ValueError(
            "Either subsystem or each of state, node_indices, and node_labels are required"
        )
    if state is None:
        state = subsystem.state
    if node_indices is None:
        node_indices = subsystem.node_indices
    if node_labels is None:
        node_labels = subsystem.node_labels

    # Need to convert to native dict because Plotly has overly strict type
    # checking; see https://github.com/plotly/plotly.py/issues/4212
    theme = Theme(theme, **theme_overrides).to_dict()

    if fig is None:
        fig = go.Figure()
    fig.update_layout(theme["layout"])

    label = text.Labeler(state, node_labels)

    mechanism_mapping = theme["geometry"]["mechanisms"].get("mapping")
    if mechanism_mapping is None:
        mechanism_mapping = geometry.arrange(
            node_indices,
            **theme["geometry"]["mechanisms"]["arrange"],
        )
    offset_subsets = list(mechanism_mapping)
    mechanism_coords = mechanism_coords or theme["mechanisms"].get("coords")
    if mechanism_coords is None:
        mechanism_coords = geometry.Coordinates(
            mechanism_mapping,
            **theme["geometry"]["mechanisms"].get("coordinate_kwargs", dict()),
        )

    purview_mapping = theme["geometry"]["purviews"].get("mapping")
    if purview_mapping is None:
        purview_mapping = geometry.arrange(
            node_indices,
            **theme["geometry"]["purviews"].get("arrange", dict()),
        )
    purview_coords = purview_coords or theme["geometry"]["purviews"].get("coords")
    if purview_coords is None:
        purview_coords = geometry.Coordinates(
            purview_mapping,
            offset_subsets=offset_subsets,
            **theme["geometry"]["purviews"].get("coordinate_kwargs", dict()),
        )

    # Relations
    if theme["show"].get("two_relations") or theme["show"].get("three_relations"):
        # Sort relations for deterministic traversal
        faces_by_degree = dict()
        for degree, faces in relations.faces_by_degree.items():
            faces_by_degree[degree] = sorted(faces)

        def face_to_coords(face):
            return np.array(
                [
                    purview_coords.get(
                        relatum.purview,
                        direction=relatum.direction,
                        offset_subset=relatum.mechanism,
                        offset_state=relatum.specified_state.state,
                    )
                    for relatum in face
                ]
            )

        # 2-relations
        if theme["show"].get("two_relations") and faces_by_degree[2]:
            fig = _plot_two_relation_faces(
                fig=fig,
                face_to_coords=face_to_coords,
                relation_faces=faces_by_degree[2],
                label=label,
                theme=theme,
            )

        # 3-relations
        if theme["show"].get("three_relations") and faces_by_degree[3]:
            fig = _plot_three_relation_faces(
                fig=fig,
                face_to_coords=face_to_coords,
                relation_faces=faces_by_degree[3],
                label=label,
                theme=theme,
            )

    # Cause-effect links
    if theme["show"].get("cause_effect_links"):
        fig = _plot_cause_effect_links(
            fig,
            distinctions,
            purview_coords,
            theme,
        )

    # Mechanism-purview links
    if theme["show"].get("mechanism_purview_links"):
        fig = _plot_mechanism_purview_links(
            fig,
            distinctions,
            purview_coords,
            mechanism_coords,
            theme,
        )

    # Mechanisms
    if theme["show"].get("mechanisms"):
        fig = _plot_mechanisms(fig, distinctions, mechanism_coords, label, theme)

    # Distinctions
    if theme["show"].get("purviews"):
        fig = _plot_purviews(
            fig,
            distinctions,
            purview_coords,
            label,
            theme,
        )

    if return_coords:
        return fig, purview_coords, mechanism_coords
    return fig


def scatter_from_coords(coords, theme=DEFAULT_THEME, **kwargs):
    """Return a Scatter3d given labels and coordinates."""
    x, y, z = np.stack(coords).transpose()
    defaults = dict(
        mode="text",
        textposition="middle center",
        textfont=dict(family=theme["fontfamily"], size=theme["fontsize"]),
        hoverinfo="text",
        showlegend=True,
    )
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        **{**defaults, **kwargs},
    )


def lines_from_coords(coords, **kwargs):
    """Return a Scatter3d line plot given labels and coordinates.

    Assumes ``coords`` has shape (<num_lines>, 2, 3), where the second dimension
    indexes start and end, and the third dimension indexes x, y, and z
    coordinates.
    """
    x, y, z = [
        _individual_lines_from_one_dimensional_coords(coords[:, :, i]) for i in range(3)
    ]
    defaults = dict(
        mode="lines",
        hoverinfo="text",
        showlegend=True,
    )
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        **{**defaults, **kwargs},
    )


def _individual_lines_from_one_dimensional_coords(one_dimensional_coords):
    """Return a single coordinate list with gaps to plot unconnected lines with
    Scatter3d.

    ``one_dimensional_coords`` assumed to have shape (<num_lines>, 2), where the
    second dimension indexes start and end points.
    """
    x = np.empty(len(one_dimensional_coords) * 3)
    x[0::3] = one_dimensional_coords[:, 0]
    x[1::3] = one_dimensional_coords[:, 1]
    x[2::3] = np.nan
    return x


def _plot_purviews(
    fig,
    distinctions,
    purview_coords,
    label,
    theme,
):
    marker_size = get_values(
        distinctions,
        theme["purviews"]["marker"].pop("size", None),
        rescale=theme["point_size_range"],
    )
    marker_color = get_values(
        distinctions, theme["purviews"]["marker"].pop("color", None)
    )
    textfont_color = theme["purviews"]["textfont"].pop("color", None)

    showscale = theme["purviews"]["marker"].pop("showscale", True)
    traces = []
    for direction, direction_color in zip(
        Direction.both(),
        [theme["direction"]["cause_color"], theme["direction"]["effect_color"]],
        strict=True,
    ):
        coords = [
            purview_coords.get(
                distinction.purview(direction),
                direction=direction,
                offset_subset=distinction.mechanism,
                offset_state=distinction.mice(direction).specified_state.state,
            )
            for distinction in distinctions
        ]
        labels = [
            label.nodes(
                distinction.purview(direction),
                state=distinction.mice(direction).specified_state.state,
            )
            for distinction in distinctions
        ]
        hovertext = [
            label.hover(distinction.mice(direction)) for distinction in distinctions
        ]

        current_marker_color = marker_color
        if not isinstance(marker_color, np.ndarray) and marker_color == "direction":
            current_marker_color = direction_color
        current_textfont_color = textfont_color
        if current_textfont_color == "direction":
            current_textfont_color = direction_color

        traces.append(
            scatter_from_coords(
                coords,
                theme=theme,
                marker_color=current_marker_color,
                marker_size=marker_size,
                marker_showscale=showscale,
                name=f"{direction} distinctions" + theme["legendgroup_postfix"],
                text=labels,
                hovertext=hovertext,
                hoverlabel_bgcolor=direction_color,
                textfont_color=current_textfont_color,
                **theme["purviews"],
            )
        )
        showscale = False
    return fig.add_traces(traces)


def _plot_cause_effect_links(
    fig,
    distinctions,
    purview_coords,
    theme,
):
    # TODO make this scaling consistent with 2-relation phi?
    name = "Cause-effect links" + theme["legendgroup_postfix"]
    widths = get_values(
        distinctions,
        theme["cause_effect_links"]["line"].pop("width", None),
        rescale=theme["line_width_range"],
    )
    if not isinstance(widths, np.ndarray):
        widths = [widths] * len(distinctions)

    color = theme["cause_effect_links"]["line"].pop("color", "direction")
    if color == "direction":
        colors = [theme["direction"]["cause_color"], theme["direction"]["effect_color"]]
    else:
        colors = [color] * 2

    showlegend = theme["cause_effect_links"].pop("showlegend", True)
    traces = []
    for distinction, width in zip(distinctions, widths, strict=True):
        coords = np.stack(
            [
                purview_coords.get(
                    distinction.purview(direction),
                    direction=direction,
                    offset_subset=distinction.mechanism,
                    offset_state=distinction.mice(direction).specified_state.state,
                )
                for direction in Direction.both()
            ]
        )
        x, y, z = coords.transpose()
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                showlegend=showlegend,
                legendgroup=name + theme["legendgroup_postfix"],
                name=name,
                line_width=width,
                line_color=colors,
                **theme["cause_effect_links"],
            )
        )
        showlegend = False
    return fig.add_traces(traces)


def _plot_mechanisms(fig, distinctions, mechanism_coords, label, theme):
    name = "Mechanisms" + theme["legendgroup_postfix"]
    labels = []
    coords = []
    for distinction in distinctions:
        labels.append(
            label.nodes(distinction.mechanism, state=distinction.mechanism_state)
        )
        coords.append(
            mechanism_coords.get(
                distinction.mechanism, offset_state=distinction.mechanism_state
            )
        )

    marker_size = get_values(
        distinctions,
        theme["mechanisms"]["marker"].pop("size", None),
        rescale=theme["point_size_range"],
    )
    marker_color = get_values(
        distinctions, theme["mechanisms"]["marker"].pop("color", None)
    )

    return fig.add_trace(
        scatter_from_coords(
            coords,
            theme=theme,
            legendgroup=name + theme["legendgroup_postfix"],
            name=name,
            text=labels,
            marker_size=marker_size,
            marker_color=marker_color,
            **theme["mechanisms"],
        )
    )


def _plot_mechanism_purview_links(
    fig,
    distinctions,
    purview_coords,
    mechanism_coords,
    theme,
):
    name = "Mechanism-purview links" + theme["legendgroup_postfix"]
    # TODO make this scaling consistent with 2-relation phi?
    showlegend = theme["mechanism_purview_links"].get("showlegend", True)
    traces = []

    widths = get_values(
        distinctions,
        theme["mechanism_purview_links"]["line"].pop("width", None),
        rescale=theme["line_width_range"],
    )
    if not isinstance(widths, np.ndarray):
        widths = [widths] * len(distinctions)

    color = theme["mechanism_purview_links"]["line"].pop("color", "direction")
    if color == "direction":
        cause_color, effect_color = validate_colors(
            [theme["direction"]["cause_color"], theme["direction"]["effect_color"]],
            colortype="tuple",
        )
        colors = [
            cause_color,
            find_intermediate_color(cause_color, effect_color, 0.5),
            effect_color,
        ]
    else:
        colors = [color] * 3

    for distinction, width in zip(distinctions, widths, strict=True):
        coords = np.stack(
            [
                purview_coords.get(
                    distinction.purview(Direction.CAUSE),
                    direction=Direction.CAUSE,
                    offset_subset=distinction.mechanism,
                    offset_state=distinction.mice(
                        Direction.CAUSE
                    ).specified_state.state,
                ),
                mechanism_coords.get(
                    distinction.mechanism, offset_state=distinction.mechanism_state
                ),
                purview_coords.get(
                    distinction.purview(Direction.EFFECT),
                    direction=Direction.EFFECT,
                    offset_subset=distinction.mechanism,
                    offset_state=distinction.mice(
                        Direction.EFFECT
                    ).specified_state.state,
                ),
            ]
        )
        x, y, z = coords.transpose()
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                showlegend=showlegend,
                legendgroup=name + theme["legendgroup_postfix"],
                name=name,
                line_color=colors,
                line_width=width,
                **theme["mechanism_purview_links"],
            )
        )

        showlegend = False
    return fig.add_traces(traces)


def _plot_two_relation_faces(fig, face_to_coords, relation_faces, label, theme):
    name = "2-relations" + theme["legendgroup_postfix"]

    color_spec = theme["two_relations"]["line"].pop("color", None)
    colors = _two_relation_line_colors(relation_faces, color_spec)
    if not isinstance(colors, np.ndarray):
        colors = [colors] * len(relation_faces)

    widths = get_values(
        relation_faces,
        theme["two_relations"]["line"].pop("width", None),
        rescale=theme["line_width_range"],
    )
    if not isinstance(widths, np.ndarray):
        widths = [widths] * len(relation_faces)

    detail_threshold = theme["two_relations"].pop("detail_threshold", 100)
    if len(relation_faces) >= detail_threshold:
        if color_spec == "type":
            raise NotImplementedError(
                'Cannot use the "type" color scheme for more than '
                f"`detail_threshold` 2-faces (currently set to {detail_threshold})"
            )
        return _plot_two_relation_faces_single_trace(
            fig, face_to_coords, label, theme, relation_faces, name, colors, widths
        )
    else:
        return _plot_two_relation_faces_multiple_traces(
            fig, face_to_coords, label, theme, relation_faces, name, colors, widths
        )


def _two_relation_line_colors(faces, color_spec):
    """Get the line colors for a list of faces.

    Attempts to use the function in TWO_RELATION_COLORSCHEMES specified by
    `color_spec`, falling back to the normal method of specifying colors.
    """
    return get_values(
        faces, colors.TWO_RELATION_COLORSCHEMES.get(color_spec, color_spec)
    )


def _plot_two_relation_faces_multiple_traces(
    fig,
    face_to_coords,
    label,
    theme,
    faces,
    name,
    colors,
    widths,
):
    # if not isinstance(widths, np.ndarray):
    #     widths = cycle([widths])
    showlegend = theme["two_relations"].pop("showlegend", True)
    showscale = theme["two_relations"].pop("showscale", True)
    traces = []
    colors = [
        get_color(theme["two_relations"]["line"]["colorscale"], value)
        for value in colors
    ]
    for face, width, color in zip(
        faces,
        widths,
        colors,
        strict=True,
    ):
        x, y, z = face_to_coords(face).transpose()
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                name=name,
                legendgroup=name + theme["legendgroup_postfix"],
                showlegend=showlegend,
                line_showscale=showscale,
                line_color=color,
                line_width=width,
                **theme["two_relations"],
                # hovertext=label.relation(faces),
                # hoverlabel_font_color=theme['two_relations']['hoverlabel_font_color'],
            )
        )
        # Only show the first trace in the legend
        showlegend = False
        showscale = False
    return fig.add_traces(traces)


def _plot_two_relation_faces_single_trace(
    fig,
    face_to_coords,
    label,
    theme,
    faces,
    name,
    colors,
    widths,
):
    coords = np.array([face_to_coords(face) for face in faces])
    colors = _line_color_values(colors)
    # Cannot plot different widths with a single trace
    width = np.array(widths).mean()
    return fig.add_trace(
        lines_from_coords(
            coords,
            legendgroup=name + theme["legendgroup_postfix"],
            name=name,
            line_color=colors,
            line_width=width,
            **theme["two_relations"],
            # hovertext=label.relation(faces),
            # hoverlabel_font_color=theme['two_relations']['hoverlabel_font_color'],
        )
    )


def _line_color_values(values):
    """Convert an iterable of floats to an array of colors suitable for a single
    Scatter3D line trace that will color each line as a single solid color,
    rather than interpolating colors from one marker to the next.
    """
    values = np.array(values)
    colors = np.repeat(values, 3)
    colors[2::3] = np.nan
    return colors


def _plot_three_relation_faces(fig, face_to_coords, relation_faces, label, theme):
    name = "3-relations" + theme["legendgroup_postfix"]
    # Build vertices:
    # Stack the [relation, relata] axes together and tranpose to put the 3D axis
    # first to get lists of x, y, z coordinates
    x, y, z = np.vstack(list(map(face_to_coords, relation_faces))).transpose()
    intensities = get_values(
        relation_faces,
        theme["three_relations"].pop("intensity", None),
        rescale=theme["three_relations"].pop("intensity_range", None),
    )
    opacities = get_values(
        relation_faces,
        theme["three_relations"].pop("opacity", None),
        rescale=theme["three_relations"].pop("opacity_range", None),
    )
    if not isinstance(opacities, np.ndarray):
        opacities = [opacities] * len(relation_faces)
    if len(relation_faces) >= theme["three_relations"].pop("detail_threshold", 100):
        return _plot_three_relation_faces_single_trace(
            x=x,
            y=y,
            z=z,
            fig=fig,
            label=label,
            theme=theme,
            name=name,
            intensities=intensities,
            opacity=opacities,
        )
    return _plot_three_relation_faces_multiple_traces(
        x=x,
        y=y,
        z=z,
        fig=fig,
        label=label,
        theme=theme,
        name=name,
        intensities=intensities,
        opacities=opacities,
    )


def _plot_three_relation_faces_single_trace(
    x,
    y,
    z,
    fig,
    label,
    theme,
    name,
    intensities,
    opacities,
):
    # Build triangles:
    # The vertices are stacked triples, so we want each
    # (i, j, k) = [0, 1, 2], [3, 4, 5], ...
    relata_indices = np.arange(len(intensities) * 3, step=3)
    i, j, k = np.tile(relata_indices, (3, 1)) + np.arange(3).reshape(3, 1)
    opacity = np.array(opacities).mean()

    # hovertext = list(map(label.relation, relation_faces))
    return fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            legendgroup=name + theme["legendgroup_postfix"],
            name=name,
            intensity=intensities,
            opacity=opacity,
            # hovertext=hovertext,
            **theme["three_relations"],
        )
    )


def _plot_three_relation_faces_multiple_traces(
    x,
    y,
    z,
    fig,
    label,
    theme,
    name,
    intensities,
    opacities,
):
    # hovertexts = list(map(label.relation, relation_faces))
    showlegend = theme["three_relations"].pop("showlegend", True)
    showscale = theme["three_relations"].pop("showscale", True)
    traces = []
    for _x, _y, _z, intensity, opacity in zip(
        partition(3, x),
        partition(3, y),
        partition(3, z),
        intensities,
        opacities,
        # hovertexts,
        strict=True,
    ):
        traces.append(
            go.Mesh3d(
                x=_x,
                y=_y,
                z=_z,
                i=[0],
                j=[1],
                k=[2],
                legendgroup=name + theme["legendgroup_postfix"],
                name=name,
                showlegend=showlegend,
                showscale=showscale,
                intensity=[intensity],
                opacity=opacity,
                # hovertext=hovertext,
                **theme["three_relations"],
            )
        )
        showlegend = False
        showscale = False
    return fig.add_traces(traces)


def get_values(objects, attr_or_func, rescale=None):
    if isinstance(attr_or_func, Callable):
        values = np.array(list(map(attr_or_func, objects)))
    else:
        try:
            values = np.array([getattr(obj, attr_or_func) for obj in objects])
        except (AttributeError, TypeError):
            values = attr_or_func
    if rescale is not None and isinstance(values, Iterable):
        values = utils.rescale(values, rescale)
    return values
