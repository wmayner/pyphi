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

from . import ces
from . import ising
from .ces import highlight_phi_fold
from .connectivity import plot_graph
from .connectivity import plot_system
from .distribution import plot_distribution
from .distribution import plot_repertoires
from .dynamics import plot_dynamics
from .projection import project_phi_structure
from .theme import DEFAULT_THEME
from .theme import Theme

__all__ = [
    "DEFAULT_THEME",
    "Theme",
    "ces",
    "highlight_phi_fold",
    "ising",
    "plot_distribution",
    "plot_dynamics",
    "plot_graph",
    "plot_phi_structure",
    "plot_repertoires",
    "plot_system",
    "project_phi_structure",
]

_VIEWS_PENDING = {
    "scatter": "the relational-role scatter view",
    "matrix": "the relation matrix/heatmap view",
}


def plot_phi_structure(
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
    color_by="phi",
    geometry=None,
    show=None,
):
    """Plot a |CauseEffectStructure|.

    Args:
        ces_ (CauseEffectStructure): The phi-structure to plot (distinctions
            and relations).

    Keyword Args:
        view (str): Which rendering of the structure to produce.
            ``"lattice"``: an inclusion partial order over the distinctions
            drawn as a 2-D Hasse diagram, with marker size given by each
            distinction's total relation phi and color by its phi.
            ``"simplicial_complex"``: the 3-D view with cause/effect purviews
            as vertices, degree-2 relation faces as line segments, and
            degree-3 faces as triangles.
        theme (Theme): Visual theme.
        node_labels (NodeLabels): Labels for substrate units. Defaults to the
            labels carried by the distinctions.
        fig: An existing plotly figure to draw into.
        layout (str): Horizontal placement within each rank of the lattice
            view: ``"barycentric"`` orders each rank by the mean position of
            its cover neighbors to reduce edge crossings; ``"sorted"`` orders
            by label.
        order (str): Which partial order the lattice view shows:
            ``"mechanism"`` orders distinctions by subset relation on their
            mechanisms; ``"purview_union"`` by subset relation on the unions
            of their cause and effect purviews.
        rank (str): Vertical placement in the lattice view: ``"chain"``
            places each distinction at its longest-down-chain rank
            (compact); ``"size"`` at the cardinality of its mechanism or
            purview union, leaving gaps at sizes with no distinctions.
        size_by (str): Marker size encoding: ``"sum_phi_relations"``,
            ``"phi"``, or ``None`` for uniform markers.
        color_by (str): Marker color encoding: ``"phi"`` or
            ``"sum_phi_relations"``.
        geometry (SimplicialComplexGeometry): Layout knobs for the
            simplicial-complex view.
        show (tuple[str, ...]): Element classes the simplicial-complex view
            draws. Defaults to all of them.
    """
    if view in _VIEWS_PENDING:
        raise NotImplementedError(
            f"view={view!r} is not implemented yet ({_VIEWS_PENDING[view]})"
        )
    projection = project_phi_structure(ces_, node_labels=node_labels)
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
            color_by=color_by,
        )
    if view == "simplicial_complex":
        from .render.simplicial_complex import render_simplicial_complex

        kwargs = {}
        if geometry is not None:
            kwargs["geometry"] = geometry
        if show is not None:
            kwargs["show"] = show
        return render_simplicial_complex(projection, theme, fig=fig, **kwargs)
    raise ValueError(f"unknown view {view!r}")
