# warnings.py
"""PyPhi warnings."""

from . import exceptions


class PyPhiWarning(UserWarning):
    """Class for PyPhi warnings."""


class MissingOptionalDependenciesWarning(PyPhiWarning):
    """Warn about missing dependencies."""

    MSG = exceptions.MissingOptionalDependenciesError.MSG
