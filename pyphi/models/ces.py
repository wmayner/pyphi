# models/ces.py
"""Cause-effect structure: distinctions + relations (Albantakis et al. 2023).

The IIT 4.0 paper distinguishes two terms:

- *Cause-effect structure* — the distinctions plus relations specified by
  *any* candidate system (reducible or not).
- *Φ-structure* — the cause-effect structure of a *complex* (a maximally
  irreducible substrate). The IIT 4.0 paper (p11) reserves the Greek-Φ
  spelling for that complex-specific reading.

PyPhi exposes only :class:`CauseEffectStructure` as a runtime type; the
"this is a Φ-structure" reading is communicated by context (a SIA
result's :attr:`SystemIrreducibilityAnalysis.phi_structure` attribute
holds the cause-effect structure of what won the SIA competition, i.e.
the complex). The bag-of-distinctions side (without relations) is
:class:`pyphi.models.distinctions.Distinctions` — see that module's
docstring.

The algorithms that compute cause-effect structures live in
:mod:`pyphi.formalism.iit4` (which still exposes ``phi_structure()``
as the canonical entry point — paper-faithful for the SIA-winner case).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from . import cmp
from . import fmt
from .distinctions import Distinctions

if TYPE_CHECKING:
    from pyphi.data_structures import PyPhiFloat
    from pyphi.relations import Relations


@dataclass(frozen=True, eq=False)
class CauseEffectStructure(cmp.Orderable):
    """A Φ-structure: SIA + distinctions + relations.

    Access the system-level integrated information value via
    ``ps.sia.phi``; the system partition via ``ps.sia.partition``; the
    specified system state via ``ps.sia.system_state``. (Earlier versions
    of this class proxied those attributes at the top level via
    ``__getattr__`` — that proxy was removed when the class moved into
    ``pyphi.models``; access them explicitly through ``.sia``.)
    """

    sia: Any  # SystemIrreducibilityAnalysis from formalism.iit4
    distinctions: Distinctions
    relations: Relations
    config: Any = None  # ConfigSnapshot from pyphi.conf.snapshot

    def __post_init__(self) -> None:
        if self.config is None:
            from pyphi.conf import config as _global

            object.__setattr__(self, "config", _global.snapshot())

    @property
    def components(self) -> Iterable[Any]:
        yield from self.distinctions
        # Relations is not iterable in base class but subclasses (ConcreteRelations) are
        yield from list(self.relations)  # pyright: ignore[reportArgumentType]

    def order_by(self) -> PyPhiFloat:
        return self.sia.phi

    def __hash__(self) -> int:
        return hash((self.distinctions, self.relations))

    def __bool__(self) -> bool:
        return bool(self.sia)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CauseEffectStructure):
            return False
        return (
            self.sia == other.sia
            and self.distinctions == other.distinctions
            and self.relations == other.relations
        )

    def _repr_columns(self) -> list[tuple[str, Any]]:
        # Relations may not have __len__ in base class — use num_relations()
        num_relations = (
            self.relations.num_relations()
            if hasattr(self.relations, "num_relations")
            else 0
        )
        return [
            ("Φ", self.big_phi),
            ("#(distinctions)", len(self.distinctions)),
            ("Σ φ_d", self.sum_phi_distinctions),
            ("#(relations)", num_relations),
            ("Σ φ_r", self.sum_phi_relations),
        ]

    def __repr__(self) -> str:
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_1)
        body += "\n" + str(self.sia)
        return fmt.box(fmt.center(body))

    @property
    def sum_phi_relations(self):
        return self.relations.sum_phi()

    @property
    def sum_phi_distinctions(self):
        return self.distinctions.sum_phi()

    @property
    def big_phi(self):
        return self.sum_phi_distinctions + self.sum_phi_relations

    def to_json(self) -> dict[str, Any]:
        return {
            "sia": self.sia,
            "distinctions": self.distinctions,
            "relations": self.relations,
        }
