"""Protocols declaring the cross-formalism surface of analysis results.

These Protocols use ``runtime_checkable`` so ``isinstance()`` works at
runtime; the declared attributes are the shared surface, not the full
field set of any concrete class. Formalism-specific extras
(IIT 4.0's ``normalized_phi`` / ``cause`` / ``effect`` / ``system_state``,
IIT 3.0's ``partitioned_distinctions``) live on the concrete classes and
are accessible via direct attribute access or ``isinstance()`` dispatch.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class SIAInterface(Protocol):
    """The system-irreducibility analysis surface common to all formalisms.

    Implementations: :class:`pyphi.models.sia.IIT3SystemIrreducibilityAnalysis`,
    :class:`pyphi.formalism.iit4.SystemIrreducibilityAnalysis`.
    """

    phi: Any
    partition: Any
    current_state: tuple[int, ...] | None
    node_indices: tuple[int, ...] | None
    node_labels: Any
    config: Any

    def order_by(self) -> Any: ...
    def __bool__(self) -> bool: ...


@runtime_checkable
class CauseEffectStructureInterface(Protocol):
    """The cause-effect structure surface common to all formalisms.

    Implementations: :class:`pyphi.models.ces.CauseEffectStructure` (used
    by both IIT 3.0 and IIT 4.0; the relations field is empty for IIT 3.0).
    """

    sia: SIAInterface
    distinctions: Any
    relations: Any
    config: Any


@runtime_checkable
class AcSIAInterface(Protocol):
    """The actual-causation system-irreducibility analysis surface.

    Implementations:
    :class:`pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`.

    Uses ``alpha`` rather than ``phi`` per the actual-causation paper
    (Albantakis et al. 2019).
    """

    alpha: Any
    direction: Any
    account: Any
    partitioned_account: Any
    partition: Any
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    node_indices: tuple[int, ...] | None
    cause_indices: tuple[int, ...] | None
    effect_indices: tuple[int, ...] | None
    node_labels: Any
    config: Any
