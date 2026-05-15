# pyright: strict
"""Protocols and registry for phi formalisms.

A *formalism* is a strategy for computing integrated information. Each
formalism bundles a partition scheme, the distance measures it accepts, and
algorithms that combine them into mechanism-level RIAs, system-level SIAs,
and Φ-structures.

This module declares the abstract Protocol shape and the global registry.
Concrete implementations live in ``pyphi.formalism.iit3`` and
``pyphi.formalism.iit4``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Protocol
from typing import runtime_checkable

from pyphi.registry import Registry

if TYPE_CHECKING:
    from pyphi.conf.formalism import FormalismConfig

__all__ = [
    "FORMALISM_REGISTRY",
    "ApproximateFormalism",
    "ErrorInfo",
    "ExactFormalism",
    "FormalismRegistry",
    "MeasureNotCompatibleError",
    "PhiFormalism",
    "check_measure_compatible",
]


@runtime_checkable
class PhiFormalism(Protocol):
    """The minimum shape every formalism satisfies.

    Concrete formalisms also declare:

    - ``name``: stable string identifier used in ``config.formalism.iit.version`` and
      registered in :data:`FORMALISM_REGISTRY`.
    - ``compatible_measures``: frozenset of measure names that this
      formalism accepts.
    - ``partition_scheme``: name (string) of the partition scheme
      registered in ``pyphi.partition.partition_types`` to use by default.
      May be ``None`` for approximation methods that bypass partitions.
    - ``config``: the :class:`FormalismConfig` snapshot the formalism
      operates against.

    Signatures are intentionally permissive (``Any``) until the measure and
    partition Protocols tighten in a future cleanup pass.
    """

    name: ClassVar[str]
    compatible_measures: ClassVar[frozenset[str]]
    partition_scheme: ClassVar[str | None]

    @property
    def config(self) -> FormalismConfig: ...

    def evaluate_mechanism(
        self, system: Any, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any: ...

    def evaluate_mechanism_partition(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kwargs: Any,
    ) -> Any: ...

    def evaluate_system(self, system: Any, **kwargs: Any) -> Any: ...

    def build_ces(self, system: Any, **kwargs: Any) -> Any: ...


class MeasureNotCompatibleError(Exception):
    """Raised when a configured measure isn't compatible with the active formalism.

    Each :class:`PhiFormalism` declares ``compatible_measures`` as a frozenset
    of measure names it accepts. Combinations outside that set (e.g., the IIT
    4.0 formalism with the EMD distribution measure) compute a different
    mathematical object than the formalism's ``φ`` definition; rejecting them
    early prevents silently misleading results.
    """


def check_measure_compatible(formalism: PhiFormalism, measure: str) -> None:
    """Raise :class:`MeasureNotCompatibleError` if ``measure`` isn't accepted
    by ``formalism``.

    Called from each formalism's ``evaluate_*`` methods so the failure
    surface is at the dispatch boundary, not deep inside the math.
    """
    if measure not in formalism.compatible_measures:
        raise MeasureNotCompatibleError(
            f"Measure {measure!r} is not compatible with "
            f"formalism {formalism.name!r}. Compatible measures for "
            f"this formalism: {sorted(formalism.compatible_measures)}. "
            "If you want a different measure, set "
            "config.formalism.iit.version to one whose compatible_measures "
            "set contains it."
        )


class ErrorInfo(Protocol):
    """Error characterization for an approximate formalism's output.

    Discriminates the three flavors of approximation:

    - ``upper_bound``: result is a guaranteed upper bound on the true value
      (e.g., Zaeemzadeh-style certified pruning).
    - ``approximation_error``: result approximates the true value with a
      bounded error.
    - ``different_quantity``: result computes a related but distinct
      quantity (e.g., φ* vs Φ).
    """

    kind: Literal["upper_bound", "approximation_error", "different_quantity"]
    bound: float | None
    notes: str


@runtime_checkable
class ExactFormalism(PhiFormalism, Protocol):
    """Formalism that computes exact values via exhaustive enumeration."""

    exact: Literal[True]


@runtime_checkable
class ApproximateFormalism(PhiFormalism, Protocol):
    """Formalism that computes approximate values with error characterization.

    Reserved for P16's approximation framework (φ\\*, φ_G, geometric
    integrated information, certified Zaeemzadeh pruning). Declared here so
    downstream code can branch on ``exact`` cleanly.
    """

    exact: Literal[False]

    def error_characterization(self, system: Any) -> ErrorInfo: ...


class FormalismRegistry(Registry[PhiFormalism]):
    """Storage for phi formalisms.

    Validates registered objects against the :class:`PhiFormalism` Protocol
    so wrong-shape registrations fail at import. Concrete formalisms
    register themselves at the bottom of their module file:

    .. code-block:: python

        FORMALISM_REGISTRY.register("IIT_4_0_2023", IIT4_2023Formalism())

    Lookup uses the same string identifier ``config.formalism.iit.version`` holds.
    """

    desc = "phi formalisms"

    def register(self, name: str, formalism: object) -> PhiFormalism:  # type: ignore[override]
        """Register a formalism instance under ``name``.

        Unlike the parent :meth:`Registry.register`, which is a decorator,
        formalism registration is direct because formalisms are class
        instances rather than free functions. The instance is validated
        against the :class:`PhiFormalism` Protocol; ``object`` is the
        argument type so the runtime check is meaningful even for callers
        who don't run a type-checker.
        """
        if not isinstance(formalism, PhiFormalism):
            raise TypeError(
                f"Cannot register {formalism!r} as formalism {name!r}: "
                "object does not satisfy the PhiFormalism Protocol."
            )
        self.store[name] = formalism  # type: ignore[assignment]
        return formalism


FORMALISM_REGISTRY: FormalismRegistry = FormalismRegistry()
"""Global registry of phi formalisms. Looked up by string name (the value
held in ``config.formalism.iit.version``)."""
