"""Numerics layer of the PyPhi config.

Holds knobs that govern numerical comparison (precision).
Frozen dataclass — replace via :func:`dataclasses.replace` or top-level
write on the global config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyphi.conf._helpers import read_only
from pyphi.conf._helpers import yaml_repr


@read_only
@dataclass(frozen=True)
class NumericsConfig:
    """Numerical-comparison settings."""

    precision: int = 13
    """Decimal places of agreement required when φ, Φ, and α values are
    compared through :func:`pyphi.numerics.eq` and the other
    :mod:`pyphi.numerics` predicates; values smaller than
    ``10**-precision`` are treated as zero (default ``13``; the ``iit3``
    preset sets ``6``)."""

    __repr__ = yaml_repr

    def __post_init__(self) -> None:
        # NumPy integers are accepted wherever Python ones are.
        if isinstance(self.precision, np.integer):
            object.__setattr__(self, "precision", self.precision.item())
        if not isinstance(self.precision, int) or isinstance(self.precision, bool):
            raise ValueError(
                f"precision must be int; got {type(self.precision).__name__}"
            )
        if self.precision < 0:
            raise ValueError(f"precision must be >= 0; got {self.precision}")
