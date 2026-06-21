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
from .render.simplicial_complex import SimplicialComplexGeometry
from .theme import DEFAULT_THEME
from .theme import Theme

__all__ = [
    "DEFAULT_THEME",
    "SimplicialComplexGeometry",
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
    degrees=None,
    star_min_degree=None,
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
            ``"spectrum"``: a 2-D bar panel of relation count and sum of phi
            per relation degree, summarizing the high-degree structure.
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
        degrees (tuple[int, ...]): Restrict the simplicial-complex view to
            relation faces of these degrees. Defaults to all degrees present.
        star_min_degree (int): In the simplicial-complex view, the lowest
            relation-face degree drawn as a star expansion (hub + spokes);
            lower degrees keep their geometric form (degree-2 lines, degree-3
            triangles). Must be 2, 3, or 4. ``2`` (the default) draws every
            face as a star; ``4`` keeps degree-2 lines and degree-3 triangles.
    """
    from pyphi.models.ces import PhiFold

    if isinstance(ces_, PhiFold):
        raise TypeError(
            "cannot plot a PhiFold directly; use highlight_phi_fold(fold) to "
            "render a fold against its parent structure"
        )
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
        if degrees is not None:
            kwargs["degrees"] = degrees
        if star_min_degree is not None:
            kwargs["star_min_degree"] = star_min_degree
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
    if view == "spectrum":
        from .render.spectrum import render_relation_spectrum

        return render_relation_spectrum(projection, theme, fig=fig)
    raise ValueError(f"unknown view {view!r}")


def highlight_phi_fold(
    ces_,
    phi_fold=None,
    *,
    theme=DEFAULT_THEME,
    node_labels=None,
    fig=None,
    geometry=None,
    show=None,
):
    """Plot a |CauseEffectStructure| dimmed, highlighting a phi-fold.

    Call with a single :class:`PhiFold` to highlight it against its own
    ``parent``, or with ``(ces_, phi_fold)`` to highlight any object with a
    ``distinctions`` attribute against an explicit structure.

    Args:
        ces_ (CauseEffectStructure | PhiFold): The full cause-effect structure,
            or a fold (whose ``parent`` supplies the background).
        phi_fold: An object with a ``distinctions`` attribute giving the
            distinctions to highlight; they are matched to the structure's
            by mechanism. Omit when ``ces_`` is a fold.

    Keyword Args:
        theme (Theme): Visual theme for the highlighted fold; the dimmed
            background style is derived from it.
        node_labels (NodeLabels): Labels for substrate units.
        fig: An existing plotly figure to draw into.
        geometry (SimplicialComplexGeometry): Layout knobs.
        show (tuple[str, ...]): Element classes to draw.
    """
    from pyphi.models.ces import PhiFold

    from .render.simplicial_complex import render_simplicial_complex

    if phi_fold is None:
        if not isinstance(ces_, PhiFold):
            raise TypeError(
                "single-argument highlight_phi_fold requires a PhiFold; pass "
                "(ces, phi_fold) otherwise"
            )
        phi_fold = ces_
        ces_ = phi_fold.parent

    projection = project_ces(ces_, node_labels=node_labels)
    dimmed = dataclasses.replace(
        theme,
        colorscale="Greys",
        face_colorscale="Greys",
        cause_color="#999999",
        effect_color="#999999",
        edge_color="rgba(150, 150, 150, 0.2)",
        face_opacity=theme.face_opacity * 0.25,
        relation_rgb=(170, 170, 170),
        relation_alpha_range=(0.0, theme.relation_alpha_range[1] * 0.3),
        spoke_color="rgba(170, 170, 170, 0.15)",
    )
    kwargs = {}
    if geometry is not None:
        kwargs["geometry"] = geometry
    if show is not None:
        kwargs["show"] = show
    figure = render_simplicial_complex(
        projection, dimmed, fig=fig, show_colorbars=False, **kwargs
    )
    fold_mechanisms = {tuple(d.mechanism) for d in phi_fold.distinctions}
    fold_ids = {n.id for n in projection.nodes if n.mechanism in fold_mechanisms}
    return render_simplicial_complex(
        projection, theme, fig=figure, only_distinctions=fold_ids, **kwargs
    )
