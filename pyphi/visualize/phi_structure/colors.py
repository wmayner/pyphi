# visualize/colors.py
"""Handle color computations."""

from typing import Iterable

import plotly
from _plotly_utils.basevalidators import ColorscaleValidator


_TYPE_COLORS = {"isotext": "magenta", "inclusion": "indigo", "paratext": "cyan"}


def type_color(relation_face):
    return _TYPE_COLORS[two_relation_face_type(relation_face)]


TWO_RELATION_COLORSCHEMES = {"type": type_color}


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


def get_color(colorscale, loc):
    """Return the interpolated color at `loc` using the given colorscale."""
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale)

    # convert to rgb strings
    locs, colors = zip(*colorscale)
    colors = standardize_colors(colors, colortype="rgb")
    colorscale = list(zip(locs, colors))

    if isinstance(loc, Iterable):
        return [_get_color(colorscale, x) for x in loc]
    return _get_color(colorscale, loc)


def standardize_colors(colors, colortype="tuple", **kwargs):
    colors, _ = plotly.colors.convert_colors_to_same_type(
        colors, colortype=colortype, **kwargs
    )
    return colors


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

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
