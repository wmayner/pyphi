"""Numerics layer of the PyPhi config.

Holds knobs that govern numerical comparison (precision, future tolerances).
Frozen dataclass — replace via :func:`dataclasses.replace` or top-level
write on the global config.
"""

from __future__ import annotations

from dataclasses import dataclass


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
