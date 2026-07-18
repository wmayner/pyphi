# relations.py
"""Implements the formalism for computing relations."""

from __future__ import annotations

import heapq
import itertools
import math
import random
import statistics
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Iterator
from functools import cached_property
from functools import total_ordering
from itertools import product
from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

import pandas as pd
from tqdm.auto import tqdm

from . import combinatorics
from . import conf
from . import numerics
from .conf import config
from .conf import fallback
from .direction import Direction
from .display import Description
from .display import Displayable
from .display import Row
from .display import Section
from .display import Table
from .display.numbers import format_value
from .display.tables import capped_table
from .models import cmp
from .models.distinctions import ResolvedDistinctions
from .models.pandas import ToPandasMixin
from .models.pandas import records_to_frame
from .parallel import map_reduce
from .registry import Registry
from .serializable import Serializable

if TYPE_CHECKING:
    from .formalism.iit4 import Distinction  # type: ignore[attr-defined]


class RelationFace(Displayable, ToPandasMixin, frozenset):
    """A set of (potentially) related causes/effects."""

    phi: float  # Set in __new__

    def __new__(cls, *args, phi=None):
        self = super().__new__(cls, *args)
        if phi is None:
            raise ValueError("phi keyword argument is required")

        # Preserve DistanceResult type if possible, otherwise convert to float
        from pyphi.measures.distribution import DistanceResult

        if isinstance(phi, DistanceResult):
            self.phi = phi  # type: ignore[misc]  # frozenset is immutable but we set this in __new__
        else:
            self.phi = float(phi)  # type: ignore[misc]  # frozenset is immutable but we set this in __new__
        return self

    @total_ordering  # type: ignore[arg-type]  # total_ordering expects a class not instance
    def __lt__(self, other):
        # Exact total order for deterministic sorted(); selection among
        # relations goes through resolve_ties.
        # numerics: exact — total order for sorting, not a selection.
        return self.phi < other.phi  # type: ignore[attr-defined]  # phi is set in __new__

    @cached_property
    def overlap(self):
        """The set of elements that are in the purview of every relatum."""
        return set.intersection(*map(set, self.relata_purviews))

    @cached_property
    def congruent_overlap(self):
        """Return the congruent overlap(s) among the relata.

        These are the common purview elements among the relata whose specified
        states are consistent; that is, the largest subset of the union of the
        purviews such that each relatum specifies the same state for each
        element.
        """
        return set.intersection(*self.relata_units)

    # Alias
    @property
    def purview(self):
        """The purview of the relation face. Alias for ``congruent_overlap``."""
        return self.congruent_overlap

    @property
    def relata_units(self):
        """The Units in the purview of each cause/effect in this face."""
        return (set(relatum.purview_units) for relatum in self)

    @property
    def relata_purviews(self):
        """The purview of each cause/effect in this face."""
        return (relatum.purview for relatum in self)

    @property
    def distinctions(self):
        """The distinctions whose causes/effects are in this face."""
        return (relatum.parent for relatum in self)

    @property
    def num_distinctions(self):
        """The number of distinctions whose causes/effects are in this face."""
        return len(set(self.distinctions))

    def __bool__(self):
        return bool(self.congruent_overlap)

    def _pandas_record(self):
        return {
            "purview": tuple(sorted(self.purview)),
            "phi": float(self.phi),
            "degree": len(self),
        }

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("Purview", str(sorted(self.purview))),
                        Row("Relata", len(self)),
                    ),
                ),
            ),
            compact=f"{cls}(purview={sorted(self.purview)}, relata={len(self)})",
        )

    # frozenset.__repr__ takes priority in the MRO; delegate to Displayable.
    def __repr__(self) -> str:
        return Displayable.__repr__(self)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        return Displayable._repr_html_(self)


class Relation(Displayable, ToPandasMixin, frozenset, cmp.OrderableByPhi):
    """A set of relation faces forming the relation among a set of distinctions."""

    @property
    def is_self_relation(self):
        return len(self) == 1

    @property
    def _ordered_relata(self):
        """The relatum distinctions, ordered by mechanism index."""
        return sorted(self, key=lambda d: tuple(d.mechanism))

    @property
    def labeled_mechanisms(self):
        """The state-labeled mechanism of each relatum, ordered by mechanism
        index — for display.

        Each relatum is rendered as its distinction's ``mechanism_label`` (node
        labels cased by the mechanism state), falling back to the raw index
        tuple when a relatum carries no labels.
        """
        return tuple(
            getattr(distinction, "mechanism_label", None)
            or str(tuple(distinction.mechanism))
            for distinction in self._ordered_relata
        )

    def _relatum_labels(self, distinction):
        node_labels = getattr(distinction, "node_labels", None)
        mechanism = distinction.mechanism
        if node_labels is None:
            return tuple(mechanism)
        return tuple(node_labels.coerce_to_labels(mechanism))

    def _pandas_record(self):
        # Structured data for analysis: each relatum is a plain label tuple,
        # not a display string. Card formatting lives in ``relations_table``.
        return {
            "relata": tuple(self._relatum_labels(d) for d in self._ordered_relata),
            "phi": float(self.phi),
            "degree": len(self),
            "purview": tuple(sorted(self.purview)),
        }

    def _faces(self):
        """Yield faces of the relation."""
        # Exclude single-relatum faces for self-relations as a special case
        if self.is_self_relation:
            direction_set = [Direction.BIDIRECTIONAL]
        else:
            direction_set = Direction.all()

        distinctions = list(self)
        for directions in product(direction_set, repeat=len(self)):
            mice = []
            for direction, distinction in zip(directions, distinctions, strict=False):
                if direction is Direction.BIDIRECTIONAL:
                    mice.extend([distinction.cause, distinction.effect])
                else:
                    mice.append(distinction.mice(direction))
            face = RelationFace(mice, phi=self.phi)
            if face:
                yield face

    @cached_property
    def faces(self):
        return frozenset(self._faces())

    @property
    def num_faces(self):
        return len(self.faces)

    @cached_property
    def purview(self):
        # Special case for self-relations
        if self.is_self_relation:
            distinction = next(iter(self))
            return distinction.cause.purview_units & distinction.effect.purview_units

        return set.intersection(*(distinction.purview_union for distinction in self))

    @cached_property
    def phi(self) -> float:  # type: ignore[override]  # Overrides OrderableByPhi.phi with cached_property
        return float(
            len(self.purview) * min(self.distinction_phi_per_unique_purview_unit())
        )

    def distinction_phi_per_unique_purview_unit(self):
        return (relatum.phi / len(relatum.purview_union) for relatum in self)

    def __bool__(self):
        return numerics.is_positive(self.phi)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return NotImplemented
        return frozenset.__eq__(self, other)

    def __hash__(self) -> int:
        return frozenset.__hash__(self)

    @cached_property
    def mechanisms(self):
        return {distinction.mechanism for distinction in self}

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("φ_r", self.phi),
                        Row("Purview", str(sorted(self.purview))),
                        Row("Degree", len(self)),
                        Row("Faces", self.num_faces),
                    ),
                ),
            ),
            compact=f"{cls}(φ_r={format_value(self.phi)}, degree={len(self)})",
        )


def _relation_size_func(purview_unions):
    """Build a parent-side relation cost estimate over a combination.

    The cost of a relation rises with its overlap (the relation is computed
    over the intersection of the relata's purview unions) and its degree. Only
    the relative ordering matters, so this cheap proxy suffices for chunking.
    """

    def cost(combination):
        overlap = set.intersection(*(purview_unions[i] for i in combination))
        return len(overlap) * len(combination)

    return cost


def _passes(relation, max_degree, min_phi):
    """Filter predicate shared by ``materialize`` and ``strongest``."""
    if max_degree is not None and len(relation) > max_degree:
        return False
    if min_phi is not None:
        phi = float(relation.phi)
        if not (phi > min_phi or numerics.eq(phi, min_phi)):
            return False
    return True


def all_relations(distinctions, min_degree=2, max_degree=None, **kwargs):
    """Yield causal relations among a set of distinctions."""
    # Self relations
    yield from _self_relations(distinctions)
    # Non-self relations
    combinations = _combinations_with_nonempty_congruent_overlap(
        distinctions, min_degree=min_degree, max_degree=max_degree
    )

    # ``Relation`` is lazy (phi/faces are cached properties), so each mapped
    # item is nearly free (~µs) and parallel dispatch cost is dominated by
    # pickling the relations back — measured never to pay at any size
    # (benchmarks/b18_dispatch_gate.py), which is what the high
    # ``parallel_relation_evaluation`` sequential_threshold default encodes.
    # If relation evaluation gains real per-item cost (e.g. an eager or
    # expensive phi), force that work inside this worker and remeasure;
    # note eager phi also caches ~1 kB/relation (phi + purview) on objects
    # whose count grows combinatorially with the number of distinctions.
    def worker(combination):
        return Relation(distinctions[i] for i in combination)

    pkwargs = conf.parallel_kwargs(
        config.infrastructure.parallel_relation_evaluation, **kwargs
    )
    result = map_reduce(
        worker,
        combinations,
        desc="Evaluating relations",
        size_func=_relation_size_func([d.purview_union for d in distinctions]),
        **pkwargs,  # type: ignore[arg-type]  # parallel_kwargs contains map_reduce params
    )
    if result is not None:
        yield from result


def _self_relations(distinctions):
    return filter(None, (Relation([distinction]) for distinction in distinctions))


def _combinations_with_nonempty_congruent_overlap(
    components, min_degree=2, max_degree=None
):
    """Return combinations of distinctions with nonempty congruent overlap.

    Two distinctions can relate only if their purview-unions share a unit; a
    combination can relate only if all its members share a common unit, i.e.
    the intersection of their purview-unions is nonempty. Because the
    intersection compares :class:`UnitState` values — ``(index, state)``
    pairs — congruence of the shared state is enforced here at candidate
    generation, so the family is exactly the Eq. 49/56 congruent overlaps
    (not an over-approximation later filtered down).

    Parameters
    ----------
    components : Distinctions
        The distinctions to find overlaps among.
    """
    purview_unions = [frozenset(component.purview_union) for component in components]
    return combinatorics.combinations_with_nonempty_intersection(
        purview_unions, min_size=min_degree, max_size=max_degree
    )


def _atom_groups(distinctions, atoms=None):
    """Map each atom (a state-tagged unit) to the distinctions whose
    purview-union contains it.

    This is the incidence Z(n) of the S3 Appendix. When ``atoms`` is given,
    only those atoms are indexed.
    """
    groups = defaultdict(set)
    for distinction in distinctions:
        for atom in distinction.purview_union:
            if atoms is None or atom in atoms:
                groups[atom].add(distinction)
    return groups


def _maximal_sets(sets):
    """Return the maximal elements of a family of sets under inclusion."""
    distinct = sorted(set(sets), key=len, reverse=True)
    maximal = []
    for candidate in distinct:
        if not any(candidate < kept for kept in maximal):
            maximal.append(candidate)
    return maximal


def maximal_relations(distinctions, atoms=None):
    """Return the relations maximal under set inclusion of their relata.

    These are the facets of the relation complex: the relations (degree
    ≥ 2) form a downward-closed family, so every relation's relata are a
    subset of some maximal relation's. A set of distinctions is a relation
    exactly when it is contained in some Z(n) — the distinctions whose
    purview-union contains the state-tagged unit n [1]_ — and each Z(n) is
    itself a relation, so the maximal relations are the inclusion-maximal
    elements of {Z(n)}. No relations are enumerated; cost is quadratic in
    the number of atoms. Self-relations are excluded: the family is not
    downward-closed into degree 1 (a self-relation's overlap is the
    congruent intersection of one distinction's cause and effect purviews,
    which can be empty even when the distinction relates strongly to
    others). For φ_r-ranked relations see :meth:`Relations.strongest`.

    Parameters
    ----------
    distinctions : Iterable
        The distinctions generating the relation complex.
    atoms : collection, optional
        If given, only relations whose overlap contains one of these atoms
        are considered. If None, all atoms count.

    Returns
    -------
    ConcreteRelations
        The maximal relations, as lazy :class:`Relation` objects.

    References
    ----------
    .. [1] Albantakis L, Barbosa L, Findlay G, Grasso M, et al. (2023).
       Integrated information theory (IIT) 4.0. PLoS Computational Biology
       19(10): e1011465, S3 Appendix.
    """
    groups = _atom_groups(distinctions, atoms)
    candidates = [frozenset(group) for group in groups.values() if len(group) >= 2]
    return ConcreteRelations(Relation(group) for group in _maximal_sets(candidates))


def maximal_faces(distinctions, atoms=None):
    """Return the relation faces maximal under set inclusion of their
    relata (causes/effects), across all relations.

    The face at atom n is M(n), the causes and effects whose purview
    contains n; every face of every relation is contained in some M(n),
    and M(n) is itself a face of the relation Z(n) (the distinctions whose
    purview-union contains n [1]_), so the maximal faces are the
    inclusion-maximal elements of {M(n)}. Each face carries the φ_r of the
    relation it is a face of. A maximal face's parent relation need not be
    a maximal relation, so the maximal faces are not obtainable from
    :func:`maximal_relations`.

    Parameters
    ----------
    distinctions : Iterable
        The distinctions generating the relation complex.
    atoms : collection, optional
        If given, only faces whose overlap contains one of these atoms are
        considered. If None, all atoms count.

    Returns
    -------
    frozenset[RelationFace]
        The maximal faces.

    References
    ----------
    .. [1] Albantakis L, Barbosa L, Findlay G, Grasso M, et al. (2023).
       Integrated information theory (IIT) 4.0. PLoS Computational Biology
       19(10): e1011465, S3 Appendix.
    """
    groups = _atom_groups(distinctions, atoms)
    candidates = {}
    for atom, group in groups.items():
        if len(group) < 2:
            continue
        sides = frozenset(
            side
            for distinction in group
            for side in (distinction.cause, distinction.effect)
            if atom in side.purview_units
        )
        candidates.setdefault(sides, frozenset(group))
    return frozenset(
        RelationFace(sides, phi=Relation(candidates[sides]).phi)
        for sides in _maximal_sets(candidates)
    )


def relations_table(relations: Relations) -> Table | None:
    """Capped display table of relations (relata, ``φ_r``, degree).

    Returns ``None`` for relation sets that are not row-enumerable (e.g.
    :class:`AnalyticalRelations`). The cap
    (``config.infrastructure.repr_max_table_rows``) bounds how many rows are
    materialized, so a huge relation set is not fully realized to display.
    """
    try:
        iter(relations)  # type: ignore[arg-type]
    except TypeError:
        return None
    return capped_table(
        ("Relata (mechanisms)", "φ_r", "Degree"),
        relations,  # type: ignore[arg-type]  # iterability guarded above
        lambda r: (", ".join(r.labeled_mechanisms), r.phi, len(r)),
        total=relations.num_relations(),
    )


class Relations(Displayable, ToPandasMixin, Serializable):
    """A set of relations among distinctions."""

    def __init__(self, *args, **kwargs):
        self._num_relations_cached = None
        self._sum_phi_cached = None
        self._apportioned_sum_phi_cached = None

    def _to_pandas(self):
        rows = [
            r._pandas_record()
            for r in self  # type: ignore[attr-defined]  # iterable in subclasses
        ]
        return records_to_frame(rows, columns=["relata", "phi", "degree", "purview"])

    def sum_phi(self):
        if self._sum_phi_cached is None:
            self._sum_phi_cached = self._sum_phi()  # type: ignore[attr-defined]  # Defined in subclass
        return self._sum_phi_cached

    def apportioned_sum_phi(self):
        if self._apportioned_sum_phi_cached is None:
            self._apportioned_sum_phi_cached = self._apportioned_sum_phi()  # type: ignore[attr-defined]  # Defined in subclass
        return self._apportioned_sum_phi_cached

    def num_relations(self):
        if self._num_relations_cached is None:
            self._num_relations_cached = self._num_relations()  # type: ignore[attr-defined]  # Defined in subclass
        return self._num_relations_cached

    def sum_phi_moment(self, k: int = 2) -> float:
        """Return Σφ_r^k over all relations, including self-relations."""
        if k < 1:
            raise ValueError(f"moment order must be a positive integer: {k}")
        return math.fsum(float(relation.phi) ** k for relation in self)  # type: ignore[attr-defined]  # iterable in subclasses

    def phi_mean_std(self) -> tuple[float, float]:
        """Return the population mean and standard deviation of φ_r.

        Derived from the count, Σφ_r, and Σφ_r², so it is exact on any
        backend that answers those queries without enumeration.

        Raises
        ------
        ValueError
            If there are no relations.
        """
        n = self.num_relations()
        if n == 0:
            raise ValueError("no relations to summarize")
        mean = self.sum_phi() / n
        variance = self.sum_phi_moment(2) / n - mean**2
        return mean, math.sqrt(max(variance, 0.0))

    def num_relations_of_degree(self, degree: int) -> int:
        """Return the number of relations with exactly ``degree`` relata.

        Degree 1 counts the self-relations.
        """
        return sum(1 for relation in self if len(relation) == degree)  # type: ignore[attr-defined]  # iterable in subclasses

    def sum_phi_of_degree(self, degree: int) -> float:
        """Return Σφ_r over relations with exactly ``degree`` relata."""
        return math.fsum(
            float(relation.phi)
            for relation in self  # type: ignore[attr-defined]  # iterable in subclasses
            if len(relation) == degree
        )

    def degree_spectrum(self) -> dict[int, tuple[int, float]]:
        """Return ``{degree: (count, Σφ_r)}`` over all relations.

        Degrees with no relations are omitted. The counts sum to
        ``num_relations()`` and the φ sums to ``sum_phi()``.
        """
        counts: Counter[int] = Counter()
        sums: defaultdict[int, list[float]] = defaultdict(list)
        for relation in self:  # type: ignore[attr-defined]  # iterable in subclasses
            counts[len(relation)] += 1
            sums[len(relation)].append(float(relation.phi))
        return {
            degree: (counts[degree], math.fsum(sums[degree]))
            for degree in sorted(counts)
        }

    def sum_phi_by_distinction(self, distinctions) -> tuple[float, ...]:
        """Return each distinction's incident Σφ_r, aligned to ``distinctions``.

        A distinction's incident Σφ_r is the sum of φ_r over every relation
        that contains it, including its self-relation. The result is a tuple
        parallel to ``distinctions``; a distinction that no relation reaches
        contributes ``0.0``.
        """
        position = {tuple(d.mechanism): i for i, d in enumerate(distinctions)}
        sums = [0.0] * len(position)
        for relation in self:  # type: ignore[attr-defined]  # iterable in subclasses
            phi = float(relation.phi)
            for mechanism in relation.mechanisms:
                index = position.get(tuple(mechanism))
                if index is not None:
                    sums[index] += phi
        return tuple(sums)

    def max_phi(self) -> float:
        """Return the maximum φ_r over all relations, or ``0.0`` if empty."""
        # numerics: exact — the reported maximum, not a tolerant selection.
        return max(
            (float(relation.phi) for relation in self),  # type: ignore[attr-defined]  # iterable in subclasses
            default=0.0,
        )

    def phi_histogram(self) -> dict[float, int]:
        """Return ``{φ_r: count}`` over all relations.

        Keys are grouped at the configured precision
        (:func:`pyphi.numerics.round_to_precision`), so mathematically equal
        values that differ by float noise share a bucket. Counts sum to
        ``num_relations()``.
        """
        histogram: Counter[float] = Counter(
            numerics.round_to_precision(float(relation.phi))
            for relation in self  # type: ignore[attr-defined]  # iterable in subclasses
        )
        return dict(histogram)

    def num_faces(self) -> int:
        """Return the total number of faces across all relations."""
        return sum(relation.num_faces for relation in self)  # type: ignore[attr-defined]  # iterable in subclasses

    def binding_matrix(self) -> pd.DataFrame:
        """Return the atom-pair binding matrix of the relational structure.

        Entry ``(a, b)`` is the total minimum density (``φ_r / |O|``) of the
        non-self relations whose congruent overlap contains both atoms — the
        strength with which the two unit-states are jointly bound by
        relations. The diagonal decomposes the apportioned relation strength
        per atom. Index and columns are the atoms (state-tagged units)
        incident to at least one non-self relation, sorted. Self-relations
        are excluded: the matrix measures binding between distinctions.
        """
        weights: defaultdict[tuple, float] = defaultdict(float)
        atoms = set()
        for relation in self:  # type: ignore[attr-defined]  # iterable in subclasses
            if relation.is_self_relation:
                continue
            purview = sorted(relation.purview)
            atoms.update(purview)
            weight = float(relation.phi) / len(purview)
            for a in purview:
                for b in purview:
                    weights[a, b] += weight
        ordered = sorted(atoms)
        matrix = pd.DataFrame(0.0, index=pd.Index(ordered), columns=pd.Index(ordered))
        for (a, b), weight in weights.items():
            matrix.loc[a, b] = weight
        return matrix

    def strongest(
        self,
        k: int | None = None,
        min_phi: float | None = None,
        max_degree: int | None = None,
    ) -> Iterator[Relation]:
        """Yield relations in descending φ_r order.

        Ties in φ_r yield in an unspecified but deterministic order.

        Parameters
        ----------
        k : int, optional
            Yield at most this many relations. If None, yield all.
        min_phi : float, optional
            Stop once φ_r falls below this threshold (compared tolerantly
            at the configured precision).
        max_degree : int, optional
            Skip relations with more than this many relata.
        """
        if k is not None and k <= 0:
            return
        # Descending sort for a stream; the min_phi threshold below is tolerant.
        # numerics: exact — total order for streaming, not a tolerant selection.
        candidates = sorted(self, key=lambda r: float(r.phi), reverse=True)  # type: ignore[attr-defined]  # iterable in subclasses
        yielded = 0
        for relation in candidates:
            if min_phi is not None:
                phi = float(relation.phi)
                if not (phi > min_phi or numerics.eq(phi, min_phi)):
                    return
            if max_degree is not None and len(relation) > max_degree:
                continue
            yield relation
            yielded += 1
            if k is not None and yielded >= k:
                return

    def materialize(
        self, max_degree: int | None = None, min_phi: float | None = None
    ) -> ConcreteRelations:
        """Return the relations as an explicit :class:`ConcreteRelations`.

        The one deliberately loud way to obtain enumerable relation objects
        from a non-enumerating backend. ``max_degree`` and ``min_phi``
        (tolerant ``≥``) bound what is materialized.
        """
        return ConcreteRelations(
            relation
            for relation in self  # type: ignore[attr-defined]  # iterable in subclasses
            if _passes(relation, max_degree, min_phi)
        )

    def _facet_context(self) -> tuple[Iterable, frozenset | None]:
        """Return ``(distinctions, atoms)`` generating this set's relation
        complex: the participating distinctions and the atom filter (None
        means all atoms)."""
        return ({d for relation in self for d in relation}, None)  # type: ignore[attr-defined]  # iterable in subclasses

    def maximal_relations(self) -> ConcreteRelations:
        """Return the relations maximal under set inclusion of their relata.

        The facets of the relation complex: every relation's relata are a
        subset of some maximal relation's. Degree ≥ 2 only; self-relations
        are excluded. Computed in closed form from the distinctions — on a
        filtered set (e.g. from :meth:`materialize` with bounds) the result
        is the facets of the complex generated by the participating
        distinctions, not of the filtered subset. For φ_r-ranked relations
        see :meth:`strongest`.
        """
        distinctions, atoms = self._facet_context()
        return maximal_relations(distinctions, atoms=atoms)

    def maximal_faces(self) -> frozenset:
        """Return the relation faces maximal under set inclusion of their
        relata (causes/effects), across all relations; see
        :func:`maximal_faces`."""
        distinctions, atoms = self._facet_context()
        return maximal_faces(distinctions, atoms=atoms)

    def maximal_relations_by_distinction(self, distinctions) -> tuple:
        """Return, for each distinction, the maximal relations containing it.

        Parallel to ``distinctions``; a distinction contained in no maximal
        relation (an isolated distinction) gets an empty tuple. Within each
        tuple, facets are ordered by their sorted mechanism tuples.
        """
        facets = sorted(
            self.maximal_relations(),
            key=lambda r: tuple(sorted(tuple(d.mechanism) for d in r)),
        )
        return tuple(
            tuple(facet for facet in facets if distinction in facet)
            for distinction in distinctions
        )

    def sample(self, n: int, *, seed: int) -> RelationSample:
        """Draw a coverage-weighted sample of relations.

        Implemented on backends that hold the distinction set; see
        :meth:`AnalyticalRelations.sample`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support sampling; use AnalyticalRelations"
        )

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        num_r = self.num_relations()
        sum_phi_r = self.sum_phi()
        table = relations_table(self)
        relations_section = (
            (Section(label="Relations", body=(table,)),) if table is not None else ()
        )
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("Relations", num_r),
                        Row("Σφ_r", sum_phi_r),
                    ),
                ),
                *relations_section,
            ),
            compact=f"{cls}({num_r} relations, Σφ_r={format_value(sum_phi_r)})",
        )


class NullRelations(Relations):
    """An empty set of relations specified by a substrate whose formalism
    does not define relations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __iter__(self):
        return iter(())

    def _sum_phi(self):
        return 0

    def _apportioned_sum_phi(self):
        return 0

    def _num_relations(self):
        return 0

    def __len__(self):
        return 0


class ConcreteRelations(frozenset, Relations):
    def _sum_phi(self):
        return sum(relation.phi for relation in self)

    def _apportioned_sum_phi(self):
        return sum(relation.phi / len(relation) for relation in self)

    def _num_relations(self):
        return len(self)

    # frozenset.__repr__ and __str__ take priority over Displayable in the MRO;
    # delegate explicitly so the unified display card is used instead.
    def __repr__(self) -> str:
        return Displayable.__repr__(self)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        return Displayable._repr_html_(self)

    @cached_property
    def faces_by_degree(self):
        """Return a dictionary mapping degree to relation faces of that degree."""
        faces = defaultdict(list)
        for relation in tqdm(
            self,
            desc="Grouping relation faces by degree",
            leave=False,
        ):
            for face in relation.faces:
                faces[len(face)].append(face)
        return dict(faces)


class RelationSample:
    """An i.i.d., coverage-weighted sample of non-self relations.

    Relations are drawn with probability proportional to the size of their
    congruent overlap ``|O(S)|`` — the number of atoms covering them — which
    is known exactly per sample, so any sum over non-self relations is
    estimable without bias by Horvitz-Thompson reweighting (the union-of-sets
    sampling scheme of Karp & Luby (1983)). Self-relations are never sampled:
    there are at most ``|D|`` of them, and their exact totals are carried on
    the sample so the convenience estimators cover all relations.

    Attributes
    ----------
    relations : tuple[Relation, ...]
        The sampled relations, drawn with replacement.
    normalization : int
        The exact coverage-weighted total ``Σ_S |O(S)|`` over all non-self
        relations.
    seed : int
        The seed of the isolated random generator that produced the sample.
    num_self_relations : int
        The exact number of self-relations in the structure.
    sum_phi_self_relations : float
        The exact Σφ_r over the self-relations.
    """

    def __init__(
        self,
        relations,
        normalization,
        seed,
        num_self_relations,
        sum_phi_self_relations,
    ):
        self.relations = tuple(relations)
        self.normalization = normalization
        self.seed = seed
        self.num_self_relations = num_self_relations
        self.sum_phi_self_relations = sum_phi_self_relations

    def __len__(self):
        return len(self.relations)

    def __iter__(self):
        return iter(self.relations)

    def __repr__(self):
        return (
            f"{type(self).__name__}(n={len(self.relations)}, "
            f"normalization={self.normalization}, seed={self.seed})"
        )

    def estimate(self, f) -> tuple[float, float]:
        """Return an unbiased estimate and standard error of ``Σ f(S)`` over
        all non-self relations.

        Parameters
        ----------
        f : Callable[[Relation], float]
            The per-relation summand.
        """
        if not self.relations:
            return 0.0, 0.0
        values = [
            self.normalization * float(f(relation)) / len(relation.purview)
            for relation in self.relations
        ]
        mean = math.fsum(values) / len(values)
        stderr = (
            statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else float("nan")
        )
        return mean, stderr

    def num_relations(self) -> tuple[float, float]:
        """Return an estimate and standard error of the total relation
        count, including the exact self-relation count."""
        estimate, stderr = self.estimate(lambda _: 1.0)
        return estimate + self.num_self_relations, stderr

    def sum_phi(self) -> tuple[float, float]:
        """Return an estimate and standard error of Σφ_r over all
        relations, including the exact self-relation total."""
        estimate, stderr = self.estimate(lambda relation: float(relation.phi))
        return estimate + self.sum_phi_self_relations, stderr


class AnalyticalRelations(Relations):
    """A closed-form summary of the relations among a set of distinctions.

    Every query (``sum_phi``, ``num_relations``, degree spectra, ...) is a
    pure function of ``distinctions``, so two instances are equal exactly
    when their ``distinctions`` are equal — regardless of whether either was
    freshly computed, deserialized, or produced by a separate call. A
    :class:`ConcreteRelations` built from the same distinctions is never
    equal to one: the two are distinct representations, and comparing them
    would require materializing one into the other's form.
    """

    def __init__(self, distinctions):
        self.distinctions = distinctions
        super().__init__()

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.distinctions == other.distinctions

    def __hash__(self) -> int:
        return hash((type(self), self.distinctions))

    def _not_enumerable(self, verb: str) -> NoReturn:
        raise TypeError(
            f"AnalyticalRelations is a closed-form summary and cannot be {verb}: "
            "it does not enumerate the relation set. Use .strongest(k) for the "
            "top-k relations by φ_r, .materialize() to enumerate all of them "
            "(may be expensive on large structures), or set "
            "pyphi.config.relation_computation = 'CONCRETE' to compute "
            "the full concrete relation set."
        )

    def __iter__(self) -> NoReturn:
        self._not_enumerable("iterated")

    def __getitem__(self, index: object) -> NoReturn:
        self._not_enumerable("indexed")

    @cached_property
    def self_relations(self):
        return tuple(_self_relations(self.distinctions))

    @cached_property
    def _atom_index(self):
        """Map each atom (a state-tagged unit) to the distinctions whose
        purview-union contains it.

        This incidence, together with each distinction's φ density, generates
        the entire relational structure (Albantakis et al. 2023, S3
        Appendix); every closed-form query below is computed from it. Groups
        are deterministically ordered by mechanism.
        """
        index = {}
        for purview, group in self.distinctions.purview_inclusion(max_order=1):
            (atom,) = purview
            index[atom] = tuple(sorted(group, key=lambda d: tuple(d.mechanism)))
        return index

    @staticmethod
    def _density(distinction) -> float:
        """The distinction's φ per unique purview unit."""
        return float(distinction.phi) / len(distinction.purview_union)

    def _facet_context(self) -> tuple[Iterable, frozenset | None]:
        return (self.distinctions, None)

    def sample(self, n: int, *, seed: int) -> RelationSample:
        """Draw ``n`` non-self relations, coverage-weighted, i.i.d.

        Sampling walks the atom incidence: an atom is drawn with probability
        proportional to the number of relations inside its distinction
        group (``2**m − m − 1`` for a group of ``m``), then a subset of size
        ≥ 2 of that group is drawn uniformly. The resulting relation is
        drawn with probability proportional to its overlap size ``|O(S)|``,
        which is known per sample, so the returned
        :class:`RelationSample` yields unbiased estimates with standard
        errors for any per-relation sum. No burn-in; exact normalization.

        Parameters
        ----------
        n : int
            The number of draws (with replacement).
        seed : int
            Seed for the isolated random generator. Required.
        """
        rng = random.Random(seed)
        index = self._atom_index
        atoms = sorted(index)
        weights = [2 ** len(index[a]) - len(index[a]) - 1 for a in atoms]
        normalization = sum(weights)
        sampled = []
        if normalization > 0:
            for _ in range(n):
                atom = rng.choices(atoms, weights=weights)[0]
                group = index[atom]
                while True:
                    mask = rng.getrandbits(len(group))
                    if mask.bit_count() >= 2:
                        break
                sampled.append(
                    Relation(
                        distinction
                        for i, distinction in enumerate(group)
                        if mask >> i & 1
                    )
                )
        return RelationSample(
            relations=sampled,
            normalization=normalization,
            seed=seed,
            num_self_relations=len(self.self_relations),
            sum_phi_self_relations=math.fsum(
                float(relation.phi) for relation in self.self_relations
            ),
        )

    def sum_phi_moment(self, k: int = 2) -> float:
        """Return Σφ_r^k over all relations, in closed form.

        Since ``φ_r = |O(S)| · min q`` and ``|O(S)|^k`` counts the ordered
        k-tuples of atoms covering ``S``, the k-th moment decomposes over
        atom k-tuples, each contributing a sum-of-minimum of ``q^k`` over the
        distinctions shared by the tuple. Cost is ``O(N^k)`` inner sums for
        ``N`` atoms.
        """
        if k < 1:
            raise ValueError(f"moment order must be a positive integer: {k}")
        index = self._atom_index
        atoms = sorted(index)
        total = 0.0
        for combo in itertools.product(atoms, repeat=k):
            group = set(index[combo[0]])
            for atom in combo[1:]:
                group &= set(index[atom])
            if len(group) >= 2:
                total += combinatorics.sum_of_minimum_among_subsets(
                    [self._density(d) ** k for d in group]
                )
        total += math.fsum(float(relation.phi) ** k for relation in self.self_relations)
        return total

    def num_relations_of_degree(self, degree: int) -> int:
        """Return the number of relations with exactly ``degree`` relata,
        by inclusion-exclusion over shared purview subsets."""
        if degree < 1:
            return 0
        if degree == 1:
            return len(self.self_relations)
        count = 0
        for purview, group in self.distinctions.purview_inclusion(max_order=None):
            count += (-1) ** (len(purview) - 1) * math.comb(len(group), degree)
        return count

    def sum_phi_of_degree(self, degree: int) -> float:
        """Return Σφ_r over relations with exactly ``degree`` relata, as a
        per-atom sorted dot product with binomial coefficients."""
        if degree == 1:
            return math.fsum(float(r.phi) for r in self.self_relations)
        return math.fsum(
            combinatorics.sum_of_minimum_of_size_among_subsets(
                [self._density(d) for d in group], degree
            )
            for group in self._atom_index.values()
        )

    def degree_spectrum(self) -> dict[int, tuple[int, float]]:
        """Return ``{degree: (count, Σφ_r)}``, in closed form per degree."""
        num_distinctions = sum(1 for _ in self.distinctions)
        spectrum = {}
        for degree in range(1, num_distinctions + 1):
            count = self.num_relations_of_degree(degree)
            if count:
                spectrum[degree] = (count, self.sum_phi_of_degree(degree))
        return spectrum

    def sum_phi_by_distinction(self, distinctions) -> tuple[float, ...]:
        """Return each distinction's incident Σφ_r in closed form.

        A relation either contains a given distinction or does not, so its
        incident Σφ_r is ``total − Σφ_r(relations avoiding it)``: the full
        total differenced against the total over the remaining distinctions.
        No relations are enumerated. The result is parallel to ``distinctions``.
        """
        from pyphi.models.distinctions import ResolvedDistinctions

        total = self.sum_phi()
        result = []
        for distinction in distinctions:
            mechanism = tuple(distinction.mechanism)
            others = ResolvedDistinctions(
                d for d in self.distinctions if tuple(d.mechanism) != mechanism
            )
            result.append(total - AnalyticalRelations(others).sum_phi())
        return tuple(result)

    def max_phi(self) -> float:
        """Return the maximum φ_r, scanning only pairs and self-relations.

        Notes
        -----
        The maximum over relations of degree ≥ 2 is always attained at
        degree 2: for any relation ``S`` with minimum-density member ``d*``
        and any other member ``d'``, the pair ``{d*, d'}`` has overlap
        containing ``O(S)`` and the same minimum density, so its φ_r is at
        least ``φ_r(S)``. The scan is ``O(D^2)``.
        """
        ds = list(self.distinctions)
        unions = [frozenset(d.purview_union) for d in ds]
        densities = [self._density(d) for d in ds]
        # numerics: exact — seeds a running max; callers compare tolerantly.
        best = max(
            (float(relation.phi) for relation in self.self_relations),
            default=0.0,
        )
        for i, j in itertools.combinations(range(len(ds)), 2):
            overlap = unions[i] & unions[j]
            if overlap:
                # numerics: exact — running max; callers compare tolerantly.
                best = max(best, len(overlap) * min(densities[i], densities[j]))
        return best

    def strongest(
        self,
        k: int | None = None,
        min_phi: float | None = None,
        max_degree: int | None = None,
    ) -> Iterator[Relation]:
        """Yield relations in descending φ_r order, lazily.

        Best-first search over the subset lattice: φ_r never increases when
        a relatum is added (the overlap shrinks and the minimum density can
        only fall), so seeding a max-heap with all valid pairs and the
        self-relations, and expanding each popped combination by
        larger-index distinctions only, yields relations in exact descending
        order. The first ``K`` yields cost ``O(|D|²)`` seeding plus
        ``O(K·|D|)`` heap pushes, independent of the total relation count.

        Ties in φ_r yield in an unspecified but deterministic order. The
        heap can grow to ``O(yielded · |D|)`` entries when the stream is
        consumed deeply; full enumeration is better served by
        :meth:`materialize`.

        Parameters
        ----------
        k : int, optional
            Yield at most this many relations. If None, yield all.
        min_phi : float, optional
            Stop once φ_r falls below this threshold (compared tolerantly
            at the configured precision). Sound as an early exit because
            the stream is globally descending.
        max_degree : int, optional
            Do not yield or expand relations with more than this many
            relata.
        """
        if k is not None and k <= 0:
            return
        ds = list(self.distinctions)
        unions = [frozenset(d.purview_union) for d in ds]
        densities = [self._density(d) for d in ds]

        def phi_of(indices):
            overlap = frozenset.intersection(*(unions[i] for i in indices))
            if not overlap:
                return None
            return len(overlap) * min(densities[i] for i in indices)

        heap: list = []
        counter = itertools.count()

        def push(phi, payload):
            # numerics: exact — heap ordering is a total order over floats;
            # the min_phi threshold at yield time is tolerant.
            heapq.heappush(heap, (-phi, next(counter), payload))

        if max_degree is None or max_degree >= 1:
            for relation in self.self_relations:
                push(float(relation.phi), relation)
        if max_degree is None or max_degree >= 2:
            for i, j in itertools.combinations(range(len(ds)), 2):
                phi = phi_of((i, j))
                if phi is not None:
                    push(phi, (i, j))

        yielded = 0
        while heap:
            negative_phi, _, payload = heapq.heappop(heap)
            phi = -negative_phi
            if min_phi is not None and not (phi > min_phi or numerics.eq(phi, min_phi)):
                return
            if isinstance(payload, Relation):
                relation = payload
            else:
                relation = Relation(ds[i] for i in payload)
                if max_degree is None or len(payload) < max_degree:
                    for nxt in range(payload[-1] + 1, len(ds)):
                        extended = (*payload, nxt)
                        extended_phi = phi_of(extended)
                        if extended_phi is not None:
                            push(extended_phi, extended)
            yield relation
            yielded += 1
            if k is not None and yielded >= k:
                return

    def materialize(
        self, max_degree: int | None = None, min_phi: float | None = None
    ) -> ConcreteRelations:
        """Enumerate the relations as an explicit
        :class:`ConcreteRelations`.

        The one deliberately loud way to obtain relation objects from this
        backend — the output is exponential in the number of distinctions,
        so ``max_degree`` and ``min_phi`` (tolerant ``≥``) exist to bound
        it. Self-relations are always included (they have degree 1 and
        there are at most ``|D|`` of them).
        """
        return ConcreteRelations(
            relation
            for relation in all_relations(self.distinctions, max_degree=max_degree)
            if _passes(relation, max_degree, min_phi)
        )

    def binding_matrix(self) -> pd.DataFrame:
        """Return the atom-pair binding matrix, in closed form.

        Each entry is one sum-of-minimum over the distinctions shared by the
        atom pair — ``O(A²)`` sorted dot products for ``A`` atoms, never
        touching a relation.
        """
        index = self._atom_index
        atoms = sorted(a for a in index if len(index[a]) >= 2)
        matrix = pd.DataFrame(0.0, index=pd.Index(atoms), columns=pd.Index(atoms))
        for a in atoms:
            members = set(index[a])
            for b in atoms:
                group = [d for d in index[b] if d in members]
                if len(group) >= 2:
                    matrix.loc[a, b] = combinatorics.sum_of_minimum_among_subsets(
                        [self._density(d) for d in group]
                    )
        return matrix

    def phi_histogram(self) -> dict[float, int]:
        """Return ``{φ_r: count}`` over all relations, in closed form.

        φ_r takes at most ``A × D`` distinct values, for ``A`` purview atoms
        and ``D`` distinctions (overlap size times minimum density). The
        histogram is computed by sweeping density thresholds from high to
        low: at each threshold, relations among the distinctions at or above
        it are counted by exact overlap size via Möbius inversion over the
        intersection closure of their purview-unions; differencing
        consecutive sweeps assigns counts to ``overlap × density`` buckets.
        Distinctions are grouped by density at the configured precision
        (:func:`pyphi.numerics.round_to_precision`), so mathematically equal
        densities that differ by float noise share a threshold, and each
        bucket key is rounded to that same precision.

        Notes
        -----
        The intersection closure is bounded by ``2 ** A`` for ``A`` purview
        atoms but is small for structured systems; if it grows
        pathologically, materialization or sampling are the fallbacks.
        """
        histogram: Counter[float] = Counter()
        groups: defaultdict[float, list] = defaultdict(list)
        for distinction in self.distinctions:
            groups[numerics.round_to_precision(self._density(distinction))].append(
                distinction
            )
        cumulative: list = []
        previous: Counter[int] = Counter()
        # numerics: exact — iteration over precision-rounded representatives.
        for threshold in sorted(groups, reverse=True):
            cumulative.extend(groups[threshold])
            density = min(self._density(d) for d in groups[threshold])
            counts: Counter[int] = Counter()
            exact = combinatorics.exact_intersection_counts(
                [frozenset(d.purview_union) for d in cumulative]
            )
            for overlap, count in exact.items():
                counts[len(overlap)] += count
            for size in counts.keys() | previous.keys():
                delta = counts[size] - previous[size]
                if delta:
                    histogram[numerics.round_to_precision(size * density)] += delta
            previous = counts
        for relation in self.self_relations:
            histogram[numerics.round_to_precision(float(relation.phi))] += 1
        return dict(histogram)

    def _sum_phi(self):
        sum_phi = 0
        # Sum of phi excluding self-relations
        for _, overlapping_distinctions in self.distinctions.purview_inclusion(
            max_order=1
        ):
            sum_phi += combinatorics.sum_of_minimum_among_subsets(
                [
                    distinction.phi / len(distinction.purview_union)
                    for distinction in overlapping_distinctions
                ]
            )
        # Count self-relations
        sum_phi += sum(relation.phi for relation in self.self_relations)
        return sum_phi

    def _apportioned_sum_phi(self):
        apportioned = 0
        # Apportioned sum (Σ φ_r / |r|) excluding self-relations
        for _, overlapping_distinctions in self.distinctions.purview_inclusion(
            max_order=1
        ):
            apportioned += combinatorics.sum_of_minimum_over_size_among_subsets(
                [
                    distinction.phi / len(distinction.purview_union)
                    for distinction in overlapping_distinctions
                ]
            )
        # Self-relations have |r| = 1, so they enter at full phi
        apportioned += sum(relation.phi for relation in self.self_relations)
        return apportioned

    def _num_relations(self):
        count = 0
        # Compute number of relations excluding self-relations
        for purview, overlapping_distinctions in self.distinctions.purview_inclusion(
            max_order=None
        ):
            inclusion_exclusion_term = (-1) ** (len(purview) - 1)
            overlap_size_term = (
                2 ** len(overlapping_distinctions) - len(overlapping_distinctions) - 1
            )
            count += inclusion_exclusion_term * overlap_size_term
        # Count self-relations
        count += len(self.self_relations)
        return count

    def num_faces(self) -> int:
        """Return the total number of faces across all relations, in closed
        form.

        A face is a set of two or more causes/effects (one per direction
        choice per relatum) with nonempty state-tagged overlap, so the total
        face count is the same subfamily count that
        :meth:`num_relations` computes over distinctions, run instead over
        the individual causes and effects — Möbius inversion over the
        intersection closure of the per-side purviews. Faces of
        self-relations (a distinction's cause paired with its own effect)
        are included, matching enumeration.
        """
        mice_purviews = [
            frozenset(side.purview_units)
            for distinction in self.distinctions
            for side in (distinction.cause, distinction.effect)
        ]
        return sum(combinatorics.exact_intersection_counts(mice_purviews).values())

    def __len__(self):
        return self.num_relations()


class AnalyticalFoldRelations(AnalyticalRelations):
    """Closed-form sums over the relations incident to a set of seed
    distinctions within a parent structure.

    Every analytical quantity is a sum over relations, and a relation either
    touches the seed set ``F`` or it does not, so the incident total is
    ``total(D) - total(D\\F)`` over two plain :class:`AnalyticalRelations`.
    Self-relations of ``D\\F`` cancel in the difference; self-relations of the
    seeds survive. Enumeration (iteration, faces) is not supported -- use
    concrete relations for that.

    Equality additionally requires matching ``_seeds``: a fold summary with
    the same parent distinctions but a different seed set describes a
    different (incident-only) relation set, so it is a distinct value, not
    just a distinct view.
    """

    def __init__(self, parent_distinctions, seeds):
        super().__init__(parent_distinctions)
        self._full = AnalyticalRelations(parent_distinctions)
        self._seeds = tuple(seeds)
        self._share_weighted_cached = None
        seed_mechanisms = {tuple(d.mechanism) for d in seeds}
        from pyphi.models.distinctions import ResolvedDistinctions

        complement = ResolvedDistinctions(
            d for d in parent_distinctions if tuple(d.mechanism) not in seed_mechanisms
        )
        self._complement = AnalyticalRelations(complement)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.distinctions == other.distinctions and self._seeds == other._seeds

    def __hash__(self) -> int:
        return hash((type(self), self.distinctions, self._seeds))

    def _facet_context(self) -> tuple[Iterable, frozenset | None]:
        """Every incident relation lies inside an incident Z(n) with n in a
        seed's purview union, so fold facets restrict the atoms to the
        seeds' purview unions."""
        atoms = frozenset().union(*(seed.purview_union for seed in self._seeds))
        return (self.distinctions, atoms)

    def _sum_phi(self):
        return self._full.sum_phi() - self._complement.sum_phi()

    def _num_relations(self):
        return self._full.num_relations() - self._complement.num_relations()

    def sum_phi_by_distinction(self, distinctions) -> tuple[float, ...]:
        """Each distinction's Σφ_r over the *incident* relations, in closed
        form.

        A fold relation either contains a given distinction or does not, so
        its incident Σφ_r is the fold total differenced against the fold of
        the remaining distinctions — with the seed set restricted
        accordingly, since a relation avoiding a seed distinction is
        incident to the fold exactly when it touches one of the *other*
        seeds. The result is parallel to ``distinctions``.
        """
        from pyphi.models.distinctions import ResolvedDistinctions

        total = self.sum_phi()
        result = []
        for distinction in distinctions:
            mechanism = tuple(distinction.mechanism)
            others = ResolvedDistinctions(
                d for d in self.distinctions if tuple(d.mechanism) != mechanism
            )
            remaining_seeds = [
                seed for seed in self._seeds if tuple(seed.mechanism) != mechanism
            ]
            if remaining_seeds:
                avoiding = AnalyticalFoldRelations(others, remaining_seeds).sum_phi()
            else:
                # Every fold relation contains the sole seed, so every fold
                # relation is incident to it.
                avoiding = 0.0
            result.append(total - avoiding)
        return tuple(result)

    def _apportioned_sum_phi(self):
        return self._full.apportioned_sum_phi() - self._complement.apportioned_sum_phi()

    def share_weighted_sum_phi(self):
        """Σ over incident relations of ``φ_r · |r ∩ F| / |r|``, where ``F``
        is the seed set.

        Computed without enumeration: for a single seed ``d``, the incident
        apportioned total is ``total(D) - total(D\\{d})`` over two closed-form
        :class:`AnalyticalRelations` sums, and the share-weighted total over
        ``F`` is the sum of these single-seed incident totals (a relation of
        degree ``|r|`` binding ``k`` seeds is counted ``k`` times at
        ``φ_r/|r|``).
        """
        if self._share_weighted_cached is None:
            from pyphi.models.distinctions import ResolvedDistinctions

            total = self._full.apportioned_sum_phi()
            result = 0
            for seed in self._seeds:
                seed_mechanism = tuple(seed.mechanism)
                others = ResolvedDistinctions(
                    d for d in self.distinctions if tuple(d.mechanism) != seed_mechanism
                )
                result += total - AnalyticalRelations(others).apportioned_sum_phi()
            self._share_weighted_cached = result
        return self._share_weighted_cached

    def _difference(self, query, *args, **kwargs):
        """Evaluate an additive query as full − complement.

        A relation either touches the seed set or it does not, so any
        quantity that is a sum over relations restricts to the incident set
        by differencing the parent total against the seed-free total.
        Self-relations of non-seed distinctions cancel; the seeds' survive.
        """
        return getattr(self._full, query)(*args, **kwargs) - getattr(
            self._complement, query
        )(*args, **kwargs)

    def sum_phi_moment(self, k: int = 2) -> float:
        """Return Σφ_r^k over the incident relations."""
        return self._difference("sum_phi_moment", k)

    def num_relations_of_degree(self, degree: int) -> int:
        """Return the number of incident relations with exactly ``degree``
        relata."""
        return self._difference("num_relations_of_degree", degree)

    def sum_phi_of_degree(self, degree: int) -> float:
        """Return Σφ_r over incident relations with exactly ``degree``
        relata."""
        return self._difference("sum_phi_of_degree", degree)

    def num_faces(self) -> int:
        """Return the total face count over the incident relations."""
        return self._difference("num_faces")

    def phi_histogram(self) -> dict[float, int]:
        """Return ``{φ_r: count}`` over the incident relations.

        Bucket-wise difference of the parent and seed-free histograms. Each
        bucket key is ``overlap size × density`` rounded to the configured
        precision, where the density is a raw representative of a
        precision-rounded group of distinctions. The parent and seed-free
        histograms share bucket keys when these representatives are
        seed-independent, so the difference is well defined. A representative
        that is a seed can shift a key between the two histograms; that
        misalignment surfaces as a negative count and is raised rather than
        returned.
        """
        histogram = Counter(self._full.phi_histogram())
        histogram.subtract(self._complement.phi_histogram())
        if any(count < 0 for count in histogram.values()):
            raise ValueError(
                "bucket keys failed to align between the parent and seed-free "
                "histograms; materialize the fold to enumerate exact values"
            )
        return {phi: count for phi, count in histogram.items() if count}

    def binding_matrix(self) -> pd.DataFrame:
        """Return the atom-pair binding matrix of the incident relations.

        Entry-wise difference of the parent and seed-free matrices, on the
        parent's atom index (rows for atoms bound only by seed-free
        relations go to zero).
        """
        full = self._full.binding_matrix()
        complement = self._complement.binding_matrix()
        aligned = complement.reindex(
            index=full.index, columns=full.columns, fill_value=0.0
        )
        return full - aligned

    def max_phi(self) -> float:
        """Return the maximum φ_r over the incident relations.

        Notes
        -----
        The incident maximum is attained at an incident pair or a seed's
        self-relation: for any incident relation ``S``, its
        minimum-density member ``d*`` paired with any seed in ``S`` is an
        incident pair with overlap containing ``O(S)`` and the same minimum
        density.
        """
        seed_set = set(self._seeds)
        ds = list(self.distinctions)
        unions = [frozenset(d.purview_union) for d in ds]
        densities = [self._density(d) for d in ds]
        # numerics: exact — seeds a running max; callers compare tolerantly.
        best = max(
            (
                float(relation.phi)
                for relation in self.self_relations
                if not seed_set.isdisjoint(relation)
            ),
            default=0.0,
        )
        for i, j in itertools.combinations(range(len(ds)), 2):
            if ds[i] not in seed_set and ds[j] not in seed_set:
                continue
            overlap = unions[i] & unions[j]
            if overlap:
                # numerics: exact — running max; callers compare tolerantly.
                best = max(best, len(overlap) * min(densities[i], densities[j]))
        return best

    def strongest(
        self,
        k: int | None = None,
        min_phi: float | None = None,
        max_degree: int | None = None,
    ) -> Iterator[Relation]:
        """Yield the incident relations in descending φ_r order.

        Filters the parent's descending stream by seed incidence, so the
        order is exact; non-incident relations are popped and discarded, so
        the cost tracks the parent stream's, not the incident count.
        """
        if k is not None and k <= 0:
            return
        seed_set = set(self._seeds)
        yielded = 0
        for relation in self._full.strongest(
            k=None, min_phi=min_phi, max_degree=max_degree
        ):
            if seed_set.isdisjoint(relation):
                continue
            yield relation
            yielded += 1
            if k is not None and yielded >= k:
                return

    def materialize(
        self, max_degree: int | None = None, min_phi: float | None = None
    ) -> ConcreteRelations:
        """Return the incident relations as an explicit
        :class:`ConcreteRelations`."""
        seed_set = set(self._seeds)
        return ConcreteRelations(
            relation
            for relation in self._full.materialize(max_degree, min_phi)
            if not seed_set.isdisjoint(relation)
        )

    def sample(self, n: int, *, seed: int) -> RelationSample:
        """Not supported on folds: sample the parent structure and restrict
        the summand to incident relations instead."""
        raise NotImplementedError(
            "sampling a fold is not supported; sample the parent "
            "AnalyticalRelations and restrict the estimated summand to "
            "relations touching the seeds"
        )


def relations(
    distinctions: ResolvedDistinctions,
    relation_computation: str | None = None,
    **kwargs: Any,
) -> Relations:
    """Return causal relations among a set of distinctions.

    Requires :class:`~pyphi.models.distinctions.ResolvedDistinctions`:
    relations between distinctions whose tied specified states haven't
    been disambiguated by a SIA system_state can include phantom faces
    that wouldn't exist after resolution. Pass the result of
    :meth:`~pyphi.models.distinctions.Distinctions.resolve_congruence`
    or use :func:`pyphi.formalism.iit4.ces` to obtain a
    consistent structure.
    """
    return relation_computations[
        fallback(relation_computation, config.formalism.iit.relation_computation)  # type: ignore[index]  # config.Option descriptor
    ](distinctions, **kwargs)


class RelationComputationsRegistry(Registry):
    """Storage for functions for computing relations.

    Users can define custom schemes and use them by setting
    ``pyphi.config.relation_computation = 'NONE'``.

    Examples
    --------
    >>> @relation_computations.register('NONE')  # doctest: +SKIP
    ... def no_relations(system, ces):
    ...    return Relations([])
    """

    desc = "methods for computing relations"


relation_computations = RelationComputationsRegistry()


@relation_computations.register("CONCRETE")
def concrete_relations(
    distinctions: Iterable[Distinction], **kwargs: Any
) -> ConcreteRelations:
    return ConcreteRelations(all_relations(distinctions, **kwargs))


@relation_computations.register("ANALYTICAL")
def analytical_relations(
    distinctions: Iterable[Distinction], **kwargs: Any
) -> AnalyticalRelations:
    if kwargs:
        raise TypeError(
            f"analytical relation computation does not support keyword "
            f"arguments {sorted(kwargs)}: the analytical backend summarizes "
            f"all degrees in closed form. Use "
            f"relation_computation='CONCRETE' for degree caps or parallel "
            f"controls."
        )
    return AnalyticalRelations(distinctions)


# Functional alias
def relation(distinctions: Iterable[Distinction]) -> Relation:
    return Relation(distinctions)
