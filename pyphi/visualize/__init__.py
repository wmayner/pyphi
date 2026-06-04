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
    "evocative": "the rebuilt 3-D simplicial-complex view (legacy version: "
    "pyphi.visualize.ces.plot_phi_structure)",
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
):
    """Plot a |CauseEffectStructure|.

    Args:
        ces_ (CauseEffectStructure): The phi-structure to plot (distinctions
            and relations).

    Keyword Args:
        view (str): Which rendering of the structure to produce. Currently
            ``"lattice"``: the purview-inclusion partial order drawn as a 2-D
            Hasse diagram, with marker size given by each distinction's total
            relation phi and color by its phi.
        theme (Theme): Visual theme.
        node_labels (NodeLabels): Labels for substrate units. Defaults to the
            labels carried by the distinctions.
        fig: An existing plotly figure to draw into.
        layout (str): Horizontal placement within each rank of the lattice
            view: ``"barycentric"`` orders each rank by the mean position of
            its cover neighbors to reduce edge crossings; ``"sorted"`` orders
            by label.
    """
    if view in _VIEWS_PENDING:
        raise NotImplementedError(
            f"view={view!r} is not implemented yet ({_VIEWS_PENDING[view]})"
        )
    if view != "lattice":
        raise ValueError(f"unknown view {view!r}")
    from .render.lattice import render_lattice

    projection = project_phi_structure(ces_, node_labels=node_labels)
    return render_lattice(projection, theme, fig=fig, layout=layout)
