# models/ces.py
"""Cause-effect structure: distinctions + relations (Albantakis et al. 2023).

The IIT 4.0 paper distinguishes two terms:

- *Cause-effect structure* — the distinctions plus relations specified by
  *any* candidate system (reducible or not).
- *Φ-structure* — the cause-effect structure of a *complex* (a maximally
  irreducible substrate). The IIT 4.0 paper (p11) reserves the Greek-Φ
  spelling for that complex-specific reading.

PyPhi exposes only :class:`CauseEffectStructure` as a runtime type;
the "this is a Φ-structure" reading is communicated by context (the
substrate that specified the CES is a maximal substrate, i.e. a
complex). The bag-of-distinctions side (without relations) is
:class:`pyphi.models.distinctions.Distinctions` — see that module's
docstring.

The algorithms that compute cause-effect structures live in
:mod:`pyphi.formalism.iit4` as ``ces()`` (full CES with relations) and
:mod:`pyphi.formalism.iit3` as ``ces()`` (distinctions only — IIT 3.0
has no relations).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from . import cmp
from . import fmt
from .distinctions import ResolvedDistinctions

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
    distinctions: ResolvedDistinctions
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
            return NotImplemented
        if self.sia != other.sia:
            return False
        if self.distinctions != other.distinctions:
            return False
        return self.relations == other.relations

    def _repr_columns(self) -> list[tuple[str, Any]]:
        return fmt.fmt_ces_columns(self)

    def _repr_html_(self) -> str:
        return fmt.html_columns(self._repr_columns(), title=self.__class__.__name__)

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

    def fold(self, distinctions) -> PhiFold:
        """Return the Φ-fold seeded by the given distinctions.

        ``distinctions`` is an iterable of :class:`Distinction` objects or
        mechanism index-tuples drawn from this structure. The fold contains
        those distinctions and every relation incident to at least one of
        them.
        """
        from pyphi.relations import AnalyticalRelations
        from pyphi.relations import ConcreteRelations
        from pyphi.relations import NullRelations

        from .distinction import Distinction

        by_mechanism = {tuple(d.mechanism): d for d in self.distinctions}
        seeds = []
        for item in distinctions:
            mechanism = (
                tuple(item.mechanism)  # pyright: ignore[reportArgumentType]  # Distinction.mechanism is an index tuple
                if isinstance(item, Distinction)
                else tuple(item)
            )
            if mechanism not in by_mechanism:
                raise ValueError(
                    f"mechanism {mechanism} not in this cause-effect structure"
                )
            seeds.append(by_mechanism[mechanism])

        if isinstance(self.relations, NullRelations):
            raise ValueError(
                "folding requires relations; this cause-effect structure has "
                "none (e.g. IIT 3.0)"
            )
        seed_set = set(seeds)
        if isinstance(self.relations, ConcreteRelations):
            incident = ConcreteRelations(
                r for r in self.relations if not seed_set.isdisjoint(r)
            )
        elif isinstance(self.relations, AnalyticalRelations):
            from pyphi.relations import AnalyticalFoldRelations

            incident = AnalyticalFoldRelations(
                self.distinctions, ResolvedDistinctions(seeds)
            )
        else:
            raise TypeError(
                f"cannot fold a structure with {type(self.relations).__name__} relations"
            )
        return PhiFold(
            sia=self.sia,
            distinctions=ResolvedDistinctions(seeds),
            relations=incident,
            config=self.config,
            parent=self,
        )

    def distinction_folds(self):
        """Yield the single-distinction Φ-fold of each distinction, in order."""
        for distinction in self.distinctions:
            yield self.fold([distinction])


@dataclass(frozen=True, eq=False)
class PhiFold(CauseEffectStructure):
    """A slice of a cause-effect structure: a set of seed distinctions and
    the relations incident to them.

    ``distinctions`` holds the seeds; ``relations`` holds every relation that
    binds at least one seed; ``sia`` and ``config`` come from the structure the
    fold was taken from, available as ``parent``. A fold is not a self-contained
    cause-effect structure — its relations may reference distinctions outside
    ``distinctions`` — so it is not accepted by ``plot_ces``/``project_ces``;
    use ``highlight_phi_fold`` to visualize it.
    """

    parent: CauseEffectStructure = field(kw_only=True)

    @property
    def sum_phi_relations_contribution(self):
        """Σ over incident relations of ``φ_r / |r|`` — the relations' share of
        the fold's contribution to the structure's Φ.
        """
        return self.relations.apportioned_sum_phi()

    @property
    def big_phi_contribution(self):
        """The fold's additive contribution to the structure's Φ (the paper's
        Φ_d): the seed distinctions' full φ plus each incident relation's φ
        apportioned across the distinctions it binds. Summing this over a
        structure's single-distinction folds recovers its ``big_phi``.
        """
        return self.sum_phi_distinctions + self.sum_phi_relations_contribution
