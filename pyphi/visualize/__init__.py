# visualize/__init__.py
"""Provides visualization capabilities for PyPhi objects."""

from pyphi.exceptions import MissingOptionalDependenciesError

try:
    import matplotlib
    import plotly
    import seaborn
except ImportError as exc:
    raise MissingOptionalDependenciesError(
        MissingOptionalDependenciesError.MSG.format(dependencies="visualize")
    ) from exc

from . import ising
from . import phi_structure
from .connectivity import plot_graph
from .connectivity import plot_subsystem
from .distribution import plot_distribution
from .distribution import plot_repertoires
from .dynamics import plot_dynamics
from .phi_structure import highlight_phi_fold
from .phi_structure import plot_phi_structure
