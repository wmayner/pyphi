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

from pyphi.display import FULL
from pyphi.display import PROVENANCE
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display.numbers import format_value
from pyphi.display.tables import capped_table

from . import cmp
from .diff import Change
from .diff import ResultDiff
from .diff import _diff_common
from .distinctions import DISTINCTION_HEADER_TONES
from .distinctions import DISTINCTION_HEADERS
from .distinctions import ResolvedDistinctions
from .distinctions import distinction_table_row

if TYPE_CHECKING:
    from pyphi.data_structures import PyPhiFloat
    from pyphi.relations import Relations


@dataclass(frozen=True, eq=False, repr=False)
class CauseEffectStructure(Displayable, cmp.Orderable):
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
    provenance: Any = None  # Provenance from pyphi.provenance

    def __post_init__(self) -> None:
        if self.config is None:
            from pyphi.conf import config as _global

            object.__setattr__(self, "config", _global.snapshot())
        if self.provenance is None:
            from pyphi.provenance import Provenance

            object.__setattr__(self, "provenance", Provenance.capture())

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

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        num_d = len(self.distinctions)
        sum_phi_d = self.sum_phi_distinctions
        num_r = (
            self.relations.num_relations()
            if hasattr(self.relations, "num_relations")
            else None
        )
        sum_phi_r = (
            self.relations.sum_phi() if hasattr(self.relations, "sum_phi") else None
        )
        sia_phi = getattr(self.sia, "phi", None)

        summary_rows = [
            Row("Φ", sia_phi),
            Row("Distinctions", num_d),
            Row("Σφ_d", sum_phi_d),
            Row("Relations", num_r),
            Row("Σφ_r", sum_phi_r),
        ]

        distinctions_body = (
            capped_table(
                DISTINCTION_HEADERS,
                self.distinctions,
                distinction_table_row,
                total=num_d,
                header_tones=DISTINCTION_HEADER_TONES,
            ),
        )

        sections = [
            Section(rows=tuple(summary_rows)),
            Section(label="Distinctions", body=distinctions_body),
        ]
        if num_r:
            from pyphi.relations import relations_table

            table = relations_table(self.relations)
            if table is not None:
                sections.append(Section(label="Relations", body=(table,)))

        if verbosity >= FULL and self.sia is not None:
            # Embed the SIA's sections flat (the unlabeled summary becomes a
            # "System irreducibility" section), matching how every other card
            # embeds a sub-object — no nested boxes. Cap the embedded SIA at
            # FULL so the CES card carries a single Provenance section (its
            # own) rather than also surfacing the embedded SIA's.
            sections.extend(
                Section(
                    label=sec.label or "System irreducibility",
                    rows=sec.rows,
                    body=sec.body,
                    tone=sec.tone,
                )
                for sec in self.sia._describe(min(verbosity, FULL)).sections
            )

        if verbosity >= PROVENANCE and self.provenance is not None:
            from pyphi.display.provenance import provenance_section

            sections.append(provenance_section(self.provenance))

        return Description(
            title=cls,
            sections=tuple(sections),
            compact=f"{cls}(Φ={format_value(sia_phi)})",
        )

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

    def _changes(self, other) -> tuple[Change, ...]:
        from pyphi import utils

        changes: list[Change] = []
        a_by_mech = {d.mechanism: d for d in self.distinctions}
        b_by_mech = {d.mechanism: d for d in other.distinctions}
        changes.extend(
            Change("distinction_lost", mech, a_value=a_by_mech[mech].phi)
            for mech in a_by_mech.keys() - b_by_mech.keys()
        )
        changes.extend(
            Change("distinction_gained", mech, b_value=b_by_mech[mech].phi)
            for mech in b_by_mech.keys() - a_by_mech.keys()
        )
        for mech in a_by_mech.keys() & b_by_mech.keys():
            da, db = a_by_mech[mech], b_by_mech[mech]
            changed = (
                not utils.eq(float(da.phi), float(db.phi))
                or da.cause.purview != db.cause.purview
                or da.effect.purview != db.effect.purview
            )
            if changed:
                changes.append(
                    Change("distinction_changed", mech, a_value=da.phi, b_value=db.phi)
                )
        a_rels = (
            set(self.relations)  # pyright: ignore[reportArgumentType]
            if hasattr(self.relations, "__iter__")
            else set()
        )
        b_rels = set(other.relations) if hasattr(other.relations, "__iter__") else set()
        changes.extend(
            Change("relation_lost", tuple(r.mechanisms()), a_value=r.phi)
            for r in a_rels - b_rels
        )
        changes.extend(
            Change("relation_gained", tuple(r.mechanisms()), b_value=r.phi)
            for r in b_rels - a_rels
        )
        return tuple(changes)

    def diff(self, other) -> ResultDiff:
        """Structured delta from this cause-effect structure to ``other``."""
        if not isinstance(other, CauseEffectStructure):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self.sia, other.sia)
        return ResultDiff(
            subject=f"ΔΦ = {format_value(common['delta_phi'])}",
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            changes=self._changes(other),
            config_diff=(
                self.config.diff(other.config) if self.config and other.config else {}
            ),
            substrate_note=common["substrate_note"],
        )


@dataclass(frozen=True, eq=False, repr=False)
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

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        num_d = len(self.distinctions)
        sum_phi_d = self.sum_phi_distinctions
        num_r = (
            self.relations.num_relations()
            if hasattr(self.relations, "num_relations")
            else None
        )
        sum_phi_r_contrib = self.sum_phi_relations_contribution
        big_phi_contrib = self.big_phi_contribution
        sia_phi = getattr(self.sia, "phi", None)

        summary_rows = [
            Row("Φ_s", sia_phi),
            Row("Seed distinctions", num_d),
            Row("Σφ_d", sum_phi_d),
            Row("Incident relations", num_r),
            Row("Σφ_r (apportioned)", sum_phi_r_contrib),
            Row("Φ_d (contribution)", big_phi_contrib),
        ]

        distinctions_body = (
            capped_table(
                DISTINCTION_HEADERS,
                self.distinctions,
                distinction_table_row,
                total=num_d,
                header_tones=DISTINCTION_HEADER_TONES,
            ),
        )

        sections = [
            Section(rows=tuple(summary_rows)),
            Section(label="Seed distinctions", body=distinctions_body),
        ]
        if num_r:
            from pyphi.relations import relations_table

            table = relations_table(self.relations)
            if table is not None:
                sections.append(Section(label="Incident relations", body=(table,)))

        return Description(
            title=cls,
            sections=tuple(sections),
            compact=f"{cls}(Φ_d={format_value(big_phi_contrib)})",
        )

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
