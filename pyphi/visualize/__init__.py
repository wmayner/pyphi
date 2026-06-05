# visualize/__init__.py
"""Provides visualization capabilities for PyPhi objects."""

from pyphi.exceptions import MissingOptionalDependenciesError

try:
    import matplotlib  # noqa: F401
    import plotly  # noqa: F401
    import seaborn  # noqa: F401
except ImportError as exc:
    raise MissingOptionalDependenciesError(
        MissingOptionalDependenciesError.MSG.format(dependencies="visualize")
    ) from exc

import dataclasses

from . import ising
from .connectivity import plot_graph
from .connectivity import plot_system
from .connectivity import plot_tpm
from .distribution import plot_distribution
from .distribution import plot_repertoires
from .dynamics import plot_dynamics
from .projection import project_ces
from .theme import DEFAULT_THEME
from .theme import Theme

__all__ = [
    "DEFAULT_THEME",
    "Theme",
    "highlight_phi_fold",
    "ising",
    "plot_ces",
    "plot_distribution",
    "plot_dynamics",
    "plot_graph",
    "plot_repertoires",
    "plot_system",
    "plot_tpm",
    "project_ces",
]


def plot_ces(
    ces_,
    *,
    view="lattice",
    theme=DEFAULT_THEME,
    node_labels=None,
    fig=None,
    layout="barycentric",
    order="mechanism",
    rank="chain",
    size_by="sum_phi_relations",
    color_by=None,
    geometry=None,
    show=None,
):
    """Plot a |CauseEffectStructure|.

    Args:
        ces_ (CauseEffectStructure): The cause-effect structure to plot
            (distinctions and relations).

    Keyword Args:
        view (str): Which rendering of the structure to produce.
            ``"lattice"``: an inclusion partial order over the distinctions
            drawn as a 2-D Hasse diagram, with marker size given by each
            distinction's total relation phi and color by its phi.
            ``"simplicial_complex"``: the 3-D view with cause/effect purviews
            as vertices, degree-2 relation faces as line segments, and
            degree-3 faces as triangles.
            ``"scatter"``: distinctions on a deterministic PCA embedding of
            their unit composition, sized by total relation phi and colored
            by relational role.
            ``"matrix"``: a distinctions-by-distinctions heatmap of shared
            relation phi, with self-relation strength on the diagonal.
        theme (Theme): Visual theme.
        node_labels (NodeLabels): Labels for substrate units. Defaults to the
            labels carried by the distinctions.
        fig: An existing plotly figure to draw into.
        layout (str): Within-level ordering. In the lattice view,
            ``"barycentric"`` orders each rank by the mean position of its
            cover neighbors to reduce edge crossings; in the
            simplicial-complex view it orders each shell ring so subsets
            connected by drawn elements sit at nearby angles, shortening
            edges. ``"sorted"`` orders by label in both views.
        order (str): Which partial order the lattice view shows:
            ``"mechanism"`` orders distinctions by subset relation on their
            mechanisms; ``"purview_union"`` by subset relation on the unions
            of their cause and effect purviews.
        rank (str): Vertical placement in the lattice view: ``"chain"``
            places each distinction at its longest-down-chain rank
            (compact); ``"size"`` at the cardinality of its mechanism or
            purview union, leaving gaps at sizes with no distinctions.
        size_by (str): Marker size encoding (lattice and scatter views):
            ``"sum_phi_relations"``, ``"phi"``, or ``None`` for uniform
            markers.
        color_by (str): Marker color encoding (lattice and scatter views).
            ``None`` (the default) uses the view's default — ``"phi"`` for
            the lattice, ``"role"`` for the scatter. Both views accept
            ``"phi"`` and ``"sum_phi_relations"``; the scatter additionally
            accepts ``"role"``.
        geometry (SimplicialComplexGeometry): Layout knobs for the
            simplicial-complex view.
        show (tuple[str, ...]): Element classes the simplicial-complex view
            draws. Defaults to all of them.
    """
    projection = project_ces(ces_, node_labels=node_labels)
    if view == "lattice":
        from .render.lattice import render_lattice

        return render_lattice(
            projection,
            theme,
            fig=fig,
            layout=layout,
            order=order,
            rank=rank,
            size_by=size_by,
            color_by="phi" if color_by is None else color_by,
        )
    if view == "simplicial_complex":
        from .render.simplicial_complex import render_simplicial_complex

        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if show is not None:
            kwargs["show"] = show
        return render_simplicial_complex(
            projection, theme, fig=fig, layout=layout, **kwargs
        )
    if view == "scatter":
        from .render.scatter import render_scatter

        return render_scatter(
            projection,
            theme,
            fig=fig,
            size_by=size_by,
            color_by="role" if color_by is None else color_by,
        )
    if view == "matrix":
        from .render.matrix import render_matrix

        return render_matrix(projection, theme, fig=fig)
    raise ValueError(f"unknown view {view!r}")


def highlight_phi_fold(
    ces_,
    phi_fold,
    *,
    theme=DEFAULT_THEME,
    node_labels=None,
    fig=None,
    geometry=None,
    show=None,
):
    """Plot a |CauseEffectStructure| dimmed, highlighting a phi-fold.

    Args:
        ces_ (CauseEffectStructure): The full cause-effect structure.
        phi_fold: An object with a ``distinctions`` attribute giving the
            distinctions to highlight; they are matched to the structure's
            by mechanism.

    Keyword Args:
        theme (Theme): Visual theme for the highlighted fold; the dimmed
            background style is derived from it.
        node_labels (NodeLabels): Labels for substrate units.
        fig: An existing plotly figure to draw into.
        geometry (SimplicialComplexGeometry): Layout knobs.
        show (tuple[str, ...]): Element classes to draw.
    """
    from .render.simplicial_complex import render_simplicial_complex

    projection = project_ces(ces_, node_labels=node_labels)
    dimmed = dataclasses.replace(
        theme,
        colorscale="Greys",
        face_colorscale="Greys",
        cause_color="#999999",
        effect_color="#999999",
        edge_color="rgba(150, 150, 150, 0.2)",
        face_opacity=theme.face_opacity * 0.25,
    )
    kwargs = {}
    if geometry is not None:
        kwargs["geometry"] = geometry
    if show is not None:
        kwargs["show"] = show
    figure = render_simplicial_complex(projection, dimmed, fig=fig, **kwargs)
    fold_mechanisms = {tuple(d.mechanism) for d in phi_fold.distinctions}
    fold_ids = {n.id for n in projection.nodes if n.mechanism in fold_mechanisms}
    return render_simplicial_complex(
        projection, theme, fig=figure, only_distinctions=fold_ids, **kwargs
    )
