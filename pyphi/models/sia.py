# models/sia.py
"""IIT 3.0 ``SystemIrreducibilityAnalysis`` (system-level Φ result).

The IIT 4.0 system-level analysis lives in
:mod:`pyphi.formalism.iit4` (under the same name); the class here is the
IIT 3.0 result type that ``compute.system.sia`` produces.
"""

from __future__ import annotations

from typing import ClassVar

from pyphi import utils

from . import cmp
from . import fmt
from .distinctions import _null_ces

_sia_attributes = ["phi", "ces", "partitioned_ces", "system", "partitioned_system"]


class SystemIrreducibilityAnalysis(cmp.OrderableByPhi):
    """An analysis of system irreducibility (|big_phi|).

    Contains the |big_phi| value of the |System|, the cause-effect
    structure, and all the intermediate results obtained in the course of
    computing them.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |big_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the one with the larger system is greater.

    Attributes:
        phi (float): The |big_phi| value for the system when taken against
            this analysis, *i.e.* the difference between the cause-effect
            structure and the partitioned cause-effect structure for this
            analysis.
        ces (Distinctions): The cause-effect structure of
            the whole system.
        partitioned_ces (Distinctions): The cause-effect structure when
            the system is partitioned.
        system (System): The system this analysis was calculated for.
        partitioned_system (System): The system with the minimal partition applied.
        time (float): The number of seconds it took to calculate.
    """

    phi: float  # Override parent to allow None during init

    def __init__(
        self,
        phi=None,
        ces=None,
        partitioned_ces=None,
        system=None,
        partitioned_system=None,
        config=None,
    ):
        # Preserve DistanceResult type if possible, otherwise convert to PyPhiFloat
        if phi is None:
            self.phi = phi  # type: ignore[assignment]
        else:
            from pyphi.data_structures.pyphi_float import PyPhiFloat
            from pyphi.metrics.distribution import DistanceResult

            if isinstance(phi, DistanceResult):
                self.phi = phi  # type: ignore[assignment]
            else:
                self.phi = PyPhiFloat(phi)  # type: ignore[assignment]
        self.ces = ces
        self.partitioned_ces = partitioned_ces
        self.system = system
        self.partitioned_system = partitioned_system
        # ConfigSnapshot of the layered config at construction time.
        # Lazy-snapshot if None: takes a snapshot of the current global, so
        # callers that don't pass one still get a recorded config. Setting
        # to None explicitly is rare; mostly used for back-construction in
        # tests/fixtures.
        if config is None:
            from pyphi.conf import config as _global

            config = _global.snapshot()
        self.config = config

    def __repr__(self):
        return fmt.make_repr(self, _sia_attributes)

    def __str__(self, ces=True):
        return fmt.fmt_sia(self, ces=ces)

    def print(self, ces=True):
        """Print this |SystemIrreducibilityAnalysis|, optionally without
        cause-effect structures.
        """

    @property
    def partition(self):
        """The partition that makes the least difference to the system."""
        assert self.partitioned_system is not None
        return self.partitioned_system.partition

    @property
    def substrate(self):
        """The substrate the system belongs to."""
        assert self.system is not None
        return self.system.substrate

    unorderable_unless_eq: ClassVar[list[str]] = ["substrate"]

    def __eq__(self, other):
        return cmp.general_eq(self, other, _sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has
        |big_phi > 0|.
        """
        return not utils.eq(self.phi, 0)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.ces,
                self.partitioned_ces,
                self.system,
                self.partitioned_system,
            )
        )

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            attr: getattr(self, attr) for attr in [*_sia_attributes, "small_phi_time"]
        }

    @classmethod
    def from_json(cls, dct):
        del dct["small_phi_time"]
        return cls(**dct)


def _null_sia(system, phi=0.0):
    """Return a |SystemIrreducibilityAnalysis| with zero |big_phi| and empty
    cause-effect structures.

    This is the analysis result for a reducible system.
    """
    return SystemIrreducibilityAnalysis(
        system=system,
        partitioned_system=system,
        phi=phi,
        ces=_null_ces(),
        partitioned_ces=_null_ces(),
    )
