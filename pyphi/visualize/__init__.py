# visualize/__init__.py
"""Provides visualization capabilities for PyPhi objects."""

from ..exceptions import MissingOptionalDependenciesError

try:
    import matplotlib
    import plotly
    import seaborn
except ImportError as exc:
    raise MissingOptionalDependenciesError(
        MissingOptionalDependenciesError.MSG.format(dependencies="visualize")
    ) from exc

from .distribution import plot_distribution, plot_repertoires
from .dynamics import plot_dynamics
from .connectivity import plot_graph, plot_subsystem
from .phi_structure import plot_phi_structure, highlight_phi_fold
from . import phi_structure
from . import ising
