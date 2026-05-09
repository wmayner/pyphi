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

from . import ising
from . import phi_structure
from .ces import highlight_phi_fold
from .ces import plot_phi_structure
from .connectivity import plot_graph
from .connectivity import plot_system
from .distribution import plot_distribution
from .distribution import plot_repertoires
from .dynamics import plot_dynamics

__all__ = [
    "highlight_phi_fold",
    "ising",
    "phi_structure",
    "plot_distribution",
    "plot_dynamics",
    "plot_graph",
    "plot_phi_structure",
    "plot_repertoires",
    "plot_system",
]
