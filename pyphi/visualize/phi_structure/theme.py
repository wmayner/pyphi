# visualize/theme.py
"""Provides visual themes."""

from inspect import getmro
from pprint import pformat

from pyphi.data_structures import AttrDeepChainMap
from pyphi.models import fmt


class Theme(AttrDeepChainMap):
    """Specifies plot attributes."""

    def __init__(self, *maps, **kwargs) -> None:
        # Combine defaults from all base classes to allow easily overriding
        # certain defaults by subclassing
        base_classes = getmro(self.__class__)
        defaults = [cls.DEFAULTS for cls in base_classes if hasattr(cls, "DEFAULTS")]
        super().__init__(kwargs, *maps, *defaults)

    def __repr__(self) -> str:
        body = pformat(self.to_dict())
        return "\n".join(
            [
                f"{self.__class__.__name__}(",
                fmt.indent(body, amount=2),
                ")",
            ]
        )


Theme.SUBMAPPING_TYPE = Theme


class DefaultTheme(Theme):
    DEFAULTS = {
        "show": {
            "purviews": True,
            "mechanisms": True,
            "cause_effect_links": True,
            "mechanism_purview_links": True,
            "two_faces": True,
            "three_faces": True,
        },
        "labels": {
            "postprocessor": None,
        },
        "layout": {
            **{
                "autosize": True,
                "showlegend": True,
                "title": "",
                "width": 1000,
                "height": 800,
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
            },
            **{
                ("scene" + (str(i) if i != 1 else "")): {
                    name: {
                        "showbackground": False,
                        "showgrid": False,
                        "showticklabels": False,
                        "showspikes": False,
                        "title": "",
                    }
                    for name in ["xaxis", "yaxis", "zaxis"]
                }
                for i in range(1, 9)
            },
            **{
                "coloraxis" + (str(i) if i != 1 else ""): {
                    "colorscale": "turbo",
                    "cmin": 0,
                    "colorbar": {
                        "x": 1.1,
                        "len": 0.25,
                        "ticks": "outside",
                        "ticklen": 5,
                    },
                }
                for i in range(1, 9)
            },
        },
        "fontfamily": "Roboto Mono, MesloLGS NF, Menlo",
        "fontsize": 20,
        "pointsizerange": (5, 30),
        "linewidthrange": (1, 20),
        "geometry": {
            "purviews": {
                "arrange": {
                    "radius_func": "log_n_choose_k",
                },
                "coordinate_kwargs": {
                    "direction_offset": 0.5,
                    "subset_offset_radius": 0.1,
                    "state_offset_radius": 0.05,
                    "rotation": 0.0,
                    "rotation_plane": "xy",
                },
            },
            "mechanisms": {
                "arrange": {
                    "max_radius": 1.0,
                    "z_offset": 0.0,
                    "z_spacing": 0.0,
                    "radius_func": "linear",
                },
                "coordinate_kwargs": {
                    "rotation": 0.0,
                    "rotation_plane": "xy",
                },
            },
        },
        "direction": {
            "cause_color": "#8D3D00",
            "effect_color": "#006146",
        },
        "mechanisms": {
            "mode": "text",
            "textposition": "middle center",
            "hoverinfo": "skip",
            "showlegend": True,
            "opacity": 1,
            "marker": {
                "opacity": 0.75,
                "size": "phi",
            },
        },
        "purviews": {
            "mode": "text+markers",
            "textposition": "middle center",
            "hoverinfo": "text",
            "showlegend": True,
            "textfont": {
                "color": "direction",
            },
            "marker": {
                "opacity": 0.75,
                "color": "phi",
                "size": "phi",
                "symbol": "circle",
                "colorscale": "blues",
                "cmin": 0,
                "colorbar": {
                    "title": {"text": "φ_d"},
                    "x": 1.1,
                    "y": 0.5,
                    "len": 0.25,
                    "ticks": "outside",
                    "ticklen": 5,
                },
            },
        },
        "cause_effect_links": {
            "mode": "lines",
            "showlegend": True,
            "hoverinfo": "skip",
            "opacity": 0.25,
            "line": {
                "color": "direction",
                "width": 5,
            },
        },
        "mechanism_purview_links": {
            "mode": "lines",
            "hoverinfo": "skip",
            "showlegend": True,
            "opacity": 0.25,
            "line": {
                "color": "direction",
                "width": "phi",
            },
        },
        "two_faces": {
            "detail_threshold": 100,
            "opacity": 0.75,
            "mode": "lines",
            "hoverinfo": "text",
            "showlegend": True,
            "line": {
                "width": 5,
                "color": "phi",
                "colorscale": "blues",
                "showscale": True,
                "cmin": 0,
                "colorbar": {
                    "title": {"text": "2-face φ_r"},
                    "x": 1.1,
                    "y": 0.25,
                    "len": 0.25,
                    "ticks": "outside",
                    "ticklen": 5,
                },
            },
        },
        "three_faces": {
            "detail_threshold": 100,
            "intensity": "phi",
            "intensitymode": "cell",
            "intensity_range": (0, 1),
            "colorscale": "blues",
            "opacity": 0.1,
            "opacity_range": None,
            "cmin": 0,
            "colorbar": {
                "title": {"text": "3-face φ_r"},
                "x": 1.1,
                "y": 0.0,
                "len": 0.25,
                "ticks": "outside",
                "ticklen": 5,
            },
            "lighting": {
                "ambient": 0.8,
                "diffuse": 0,
                "roughness": 0,
                "specular": 0,
                "fresnel": 0,
            },
            "showscale": True,
            "showlegend": True,
        },
        "legendgroup_postfix": "",
    }


class Grey(DefaultTheme):
    DEFAULTS = {
        "legendgroup_postfix": " (greyed)",
        "direction": {
            "cause_color": "black",
            "effect_color": "black",
        },
        "purviews": {
            "marker": {
                "opacity": 0.75,
                "colorscale": "greys",
            },
        },
        "cause_effect_links": {
            "opacity": 0.2,
            "line": {
                "color": "grey",
            },
        },
        "mechanism_purview_links": {
            "opacity": 0.2,
            "line": {
                "color": "grey",
            },
        },
        "two_faces": {
            "opacity": 0.1,
            "line": {
                "colorscale": "greys",
            },
        },
        "three_faces": {
            "colorscale": "greys",
            "opacity": 0.05,
            "intensity_range": (0, 0.5),
        },
    }
