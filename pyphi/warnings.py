# warnings.py
"""PyPhi warnings."""

import warnings

from . import exceptions


class PyPhiWarning(UserWarning):
    """Class for PyPhi warnings."""


def warn_about_tie_serialization(
    name, serialize=False, deserialize=False, stacklevel=3
):
    # XOR; exactly one of serialize or deserialize must be True
    if not serialize ^ deserialize:
        raise ValueError("Exactly one of ``serialize``, ``deserialize`` must be True")
    if serialize:
        msg = (
            "Serializing ties in {name} is not currently supported; tie "
            "information will be lost."
        )
    if deserialize:
        msg = (
            "Deserializing ties in {name} is not currently supported; tie "
            "information was lost during serialization."
        )
    warnings.warn(msg.format(name=name), PyPhiWarning, stacklevel=stacklevel)


class MissingOptionalDependenciesWarning(PyPhiWarning):
    """Warn about missing dependencies."""

    MSG = exceptions.MissingOptionalDependenciesError.MSG
