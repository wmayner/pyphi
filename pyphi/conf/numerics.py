"""Numerics layer of the PyPhi config.

Holds knobs that govern numerical comparison (precision, future tolerances).
Frozen dataclass — replace via :func:`dataclasses.replace` or top-level
write on the global config.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyphi.conf._helpers import yaml_repr


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical-comparison settings.

    Attributes:
        precision: Decimal places of agreement required when comparing phi
            values via :func:`pyphi.utils.eq` and friends. Values smaller
            than ``10**-precision`` are treated as zero. ``PyPhiFloat``
            snapshots this at construction so its hash is stable across
            config writes.
    """

    precision: int = 13

    __repr__ = yaml_repr

    def __post_init__(self) -> None:
        if not isinstance(self.precision, int) or isinstance(self.precision, bool):
            raise ValueError(
                f"precision must be int; got {type(self.precision).__name__}"
            )
        if self.precision < 0:
            raise ValueError(f"precision must be >= 0; got {self.precision}")
