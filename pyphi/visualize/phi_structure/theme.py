# visualize/theme.py

"""Provides visualization themes for plotting phi structures."""

from inspect import getmro
from pprint import pformat

from ...data_structures import AttrDeepChainMap
from ...models import fmt


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

    DEFAULTS = dict(
        fontfamily="MesloLGS NF, Roboto Mono, Menlo",
        fontsize=12,
        cause_color="#e21a1a",
        effect_color="#14b738",
        point_size_range=(5, 30),
        line_width_range=(3, 10),
        direction=dict(
            offset=0.5,
        ),
        cause_effect_link=dict(
            color="lightgrey",
            opacity=0.5,
        ),
        mechanism=dict(
            max_radius=1.0,
            z_offset=0.0,
            z_spacing=0.0,
            shape="linear",
        ),
        purview=dict(
            shape="log_n_choose_k",
            offset_radius=0.1,
        ),
        mechanism_purview_link=dict(
            color="lightgrey",
            opacity=0.5,
        ),
        distinction=dict(
            mode="text+markers",
            opacity=0.75,
            colorscale="viridis",
            color_range=(0, 0.8),
            opacity_range=(0.1, 0.9),
        ),
        two_relation=dict(
            detail_threshold=1000,
            opacity=0.1,
            line_width=1,
            color=None,
            colorscale="type",
            showscale=True,
            reversescale=False,
            hoverlabel_font_color="white",
        ),
        three_relation=dict(
            colorscale="teal",
            reversescale=False,
            showscale=True,
            opacity=0.1,
            opacity_range=None,
            intensity_range=(0, 1),
            showlegend=True,
        ),
        lighting=dict(
            ambient=0.8,
            diffuse=0,
            roughness=0,
            specular=0,
            fresnel=0,
        ),
        legendgroup_postfix="",
        layout=dict(
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
            autosize=True,
            showlegend=True,
            hovermode="x",
            title="",
            width=1000,
            height=800,
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
        ),
    )


class Grey(Theme):
    DEFAULTS = dict(
        legendgroup_postfix=" (greyed)",
        cause_color="grey",
        effect_color="grey",
        distinction=dict(
            colorscale="greys",
            opacity_range=(0.1, 0.2),
        ),
        cause_effect_link=dict(
            color="grey",
            opacity=0.1,
        ),
        mechanism_purview_link=dict(
            color="grey",
            link_opacity=0.1,
        ),
        two_relation=dict(
            colorscale="greys",
            opacity=0.1,
            showscale=False,
        ),
        three_relation=dict(
            colorscale="greys",
            opacity=0.05,
            intensity_range=(0, 0.5),
        ),
    )
