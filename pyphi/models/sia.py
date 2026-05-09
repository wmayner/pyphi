# models/sia.py
"""IIT 3.0 ``SystemIrreducibilityAnalysis`` (system-level Φ result).

The IIT 4.0 system-level analysis lives in
:mod:`pyphi.formalism.iit4` (under the same name); the class here is the
IIT 3.0 result type that ``compute.subsystem.sia`` produces.
"""

from __future__ import annotations

from typing import ClassVar

from pyphi import utils

from . import cmp
from . import fmt
from .ces import _null_ces

_sia_attributes = ["phi", "ces", "partitioned_ces", "subsystem", "cut_subsystem"]


class SystemIrreducibilityAnalysis(cmp.OrderableByPhi):
    """An analysis of system irreducibility (|big_phi|).

    Contains the |big_phi| value of the |Subsystem|, the cause-effect
    structure, and all the intermediate results obtained in the course of
    computing them.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |big_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the one with the larger subsystem is greater.

    Attributes:
        phi (float): The |big_phi| value for the subsystem when taken against
            this analysis, *i.e.* the difference between the cause-effect
            structure and the partitioned cause-effect structure for this
            analysis.
        ces (CauseEffectStructure): The cause-effect structure of
            the whole subsystem.
        partitioned_ces (CauseEffectStructure): The cause-effect structure when
            the subsystem is cut.
        subsystem (Subsystem): The subsystem this analysis was calculated for.
        cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
        time (float): The number of seconds it took to calculate.
    """

    phi: float  # Override parent to allow None during init

    def __init__(
        self,
        phi=None,
        ces=None,
        partitioned_ces=None,
        subsystem=None,
        cut_subsystem=None,
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
        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem
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
    def cut(self):
        """The unidirectional cut that makes the least difference to the
        subsystem.
        """
        assert self.cut_subsystem is not None
        return self.cut_subsystem.cut

    @property
    def network(self):
        """The network the subsystem belongs to."""
        assert self.subsystem is not None
        return self.subsystem.network

    unorderable_unless_eq: ClassVar[list[str]] = ["network"]

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
                self.subsystem,
                self.cut_subsystem,
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


def _null_sia(subsystem, phi=0.0):
    """Return a |SystemIrreducibilityAnalysis| with zero |big_phi| and empty
    cause-effect structures.

    This is the analysis result for a reducible subsystem.
    """
    return SystemIrreducibilityAnalysis(
        subsystem=subsystem,
        cut_subsystem=subsystem,
        phi=phi,
        ces=_null_ces(),
        partitioned_ces=_null_ces(),
    )
