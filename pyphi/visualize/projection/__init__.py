"""Pure projection of IIT result objects into plot-ready data.

This package is the only part of :mod:`pyphi.visualize` that touches
result-object internals (:class:`Distinction`, :class:`Relation`). It imports
no plotting libraries; renderers consume the dataclasses defined here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

from pyphi.labels import NodeLabels

__all__ = [
    "CESProjection",
    "DistinctionNode",
    "EndpointNode",
    "InclusionOrder",
    "RelationEdge",
    "RelationFaceEdge",
    "project_ces",
]


@dataclass(frozen=True)
class DistinctionNode:
    """Plot-ready data for one distinction."""

    id: int
    mechanism: tuple[int, ...]
    label: str
    cause_purview: tuple[int, ...]
    effect_purview: tuple[int, ...]
    mechanism_state: tuple[int, ...]
    phi: float
    sum_phi_relations: float
    includes: bool
    included: bool


@dataclass(frozen=True)
class RelationEdge:
    """Plot-ready data for one relation."""

    relata: tuple[int, ...]
    degree: int
    phi: float
    overlap: tuple[int, ...]


@dataclass(frozen=True)
class EndpointNode:
    """Plot-ready data for one side (cause or effect) of a distinction."""

    id: int
    distinction_id: int
    direction: str
    purview: tuple[int, ...]
    purview_state: tuple[int, ...]
    phi: float
    label: str


@dataclass(frozen=True)
class RelationFaceEdge:
    """Plot-ready data for one relation face (any degree)."""

    endpoints: tuple[int, ...]
    degree: int
    phi: float
    overlap: tuple[int, ...]


@dataclass(frozen=True)
class InclusionOrder:
    """An inclusion partial order over distinctions.

    ``covers[i]`` lists the node ids that node ``i`` directly down-includes
    (the transitive reduction); ``rank[i]`` is the length of the longest
    down-chain below ``i`` (minimal elements have rank 0, the "whole"
    distinction the maximum), so it is monotonic in the partial order and
    suitable as a vertical layout coordinate. ``size[i]`` is the cardinality
    of the underlying unit set, an alternative vertical coordinate that
    leaves gaps at sizes with no distinctions.
    """

    covers: tuple[tuple[int, ...], ...]
    rank: tuple[int, ...]
    size: tuple[int, ...]


@dataclass(frozen=True)
class CESProjection:
    """Everything a renderer needs to draw a cause-effect structure.

    Two inclusion orders are carried: ``mechanism_inclusion`` orders
    distinctions by strict subset relation on their mechanisms (the
    region/location order of Haun & Tononi 2019, Fig 9), and
    ``purview_union_inclusion`` by strict subset relation on the unions of
    their cause and effect purviews.

    ``endpoints`` carries one node per distinction side, interleaved so
    that ``endpoints[2 * d + 0]`` is distinction ``d``'s cause and
    ``endpoints[2 * d + 1]`` its effect. ``faces`` carries the relation
    faces at every degree present, referencing endpoints by id; renderers
    choose how to draw each degree.
    """

    nodes: tuple[DistinctionNode, ...]
    edges: tuple[RelationEdge, ...]
    mechanism_inclusion: InclusionOrder
    purview_union_inclusion: InclusionOrder
    node_labels: NodeLabels
    endpoints: tuple[EndpointNode, ...] = ()
    faces: tuple[RelationFaceEdge, ...] = ()
    degree_spectrum: dict[int, tuple[int, float]] = field(default_factory=dict)

    def inclusion(self, order: str) -> InclusionOrder:
        """The inclusion order named by ``order``.

        Parameters
        ----------
        order : str
            ``"mechanism"`` or ``"purview_union"``.

        Returns
        -------
        InclusionOrder

        Raises
        ------
        ValueError
            If ``order`` is neither ``"mechanism"`` nor ``"purview_union"``.
        """
        if order == "mechanism":
            return self.mechanism_inclusion
        if order == "purview_union":
            return self.purview_union_inclusion
        raise ValueError(f"unknown order {order!r}")


def _inclusion_order(unit_sets: Sequence[frozenset]) -> InclusionOrder:
    """Partial order by strict subset relation on unit sets."""
    n = len(unit_sets)
    below: list[set[int]] = [set() for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if a != b and unit_sets[b] < unit_sets[a]:
                below[a].add(b)
    covers = tuple(
        tuple(
            sorted(
                b for b in below[a] if not any(b in below[c] for c in below[a] if c != b)
            )
        )
        for a in range(n)
    )
    memo: dict[int, int] = {}

    def longest_chain(a: int) -> int:
        if a not in memo:
            memo[a] = 1 + max(longest_chain(b) for b in below[a]) if below[a] else 0
        return memo[a]

    rank = tuple(longest_chain(a) for a in range(n))
    size = tuple(len(s) for s in unit_sets)
    return InclusionOrder(covers=covers, rank=rank, size=size)


def _unit_indices(units) -> tuple[int, ...]:
    """Sorted integer indices from an iterable of units (or bare ints)."""
    return tuple(sorted(getattr(u, "index", u) for u in units))


def _state_cased_label(purview, purview_state, node_labels) -> str:
    """Purview label with case set by state (upper = ON, lower = OFF)."""
    return "".join(
        node_labels.set_case_by_state(node_labels.indices2labels(purview), purview_state)
    )


def _endpoints(distinctions, node_labels) -> tuple[EndpointNode, ...]:
    endpoints = []
    for i, d in enumerate(distinctions):
        for j, (direction, mice) in enumerate(
            (("cause", d.cause), ("effect", d.effect))
        ):
            purview = tuple(mice.purview)
            state = tuple(mice.purview_state)
            endpoints.append(
                EndpointNode(
                    id=2 * i + j,
                    distinction_id=i,
                    direction=direction,
                    purview=purview,
                    purview_state=state,
                    phi=float(mice.phi),
                    label=_state_cased_label(purview, state, node_labels),
                )
            )
    return tuple(endpoints)


def _faces(relations, mechanism_to_id) -> tuple[RelationFaceEdge, ...]:
    faces = []
    for relation in relations:
        for face in relation.faces:
            endpoint_ids = tuple(
                sorted(
                    2 * mechanism_to_id[tuple(relatum.mechanism)]
                    + (0 if relatum.direction.name == "CAUSE" else 1)
                    for relatum in face
                )
            )
            faces.append(
                RelationFaceEdge(
                    endpoints=endpoint_ids,
                    degree=len(face),
                    phi=float(face.phi),
                    overlap=_unit_indices(face.overlap),
                )
            )
    faces.sort(key=lambda f: (f.degree, f.endpoints, f.phi))
    return tuple(faces)


def project_ces(ces, node_labels=None, max_relations=None) -> CESProjection:
    """Project a :class:`~pyphi.models.ces.CauseEffectStructure` into plot-ready data.

    Parameters
    ----------
    ces : CauseEffectStructure
        The cause-effect structure to project. Must be relation-closed.
    node_labels : NodeLabels, optional
        Labels for substrate units. Defaults to the labels carried by the first
        distinction.
    max_relations : int, optional
        Render only the ``max_relations`` strongest relations (and their faces),
        in descending φ_r order. If None, render every relation; a relation set
        that cannot be enumerated (the analytical backend) then raises, since
        "every relation" is unbounded. Node marker sizes and the degree spectrum
        are always computed over the full structure, independent of this cap.

    Returns
    -------
    CESProjection

    Raises
    ------
    TypeError
        If ``ces`` is not relation-closed (e.g. a :class:`PhiFold`, whose
        relations may reference distinctions outside it).
    ValueError
        If ``max_relations`` is None and ``ces.relations`` is not enumerable.
    """
    if not getattr(ces, "relation_closed", True):
        raise TypeError(
            "cannot project a view that is not relation-closed (e.g. a PhiFold, "
            "whose relations may reference distinctions outside it); project "
            "the parent structure or an induced substructure, or use "
            "highlight_phi_fold to visualize a fold"
        )
    distinctions = list(ces.distinctions)
    if node_labels is None:
        node_labels = distinctions[0].node_labels
    mechanism_to_id = {tuple(d.mechanism): i for i, d in enumerate(distinctions)}
    if max_relations is None:
        try:
            iter(ces.relations)
        except TypeError:
            raise ValueError(
                "relations are not enumerable (analytical backend); pass "
                "max_relations=N to render the strongest N relations by φ_r"
            ) from None
    top = list(ces.relations.strongest(k=max_relations))
    edges = tuple(
        RelationEdge(
            relata=tuple(sorted(mechanism_to_id[tuple(m)] for m in relation.mechanisms)),
            degree=len(relation),
            phi=float(relation.phi),
            overlap=_unit_indices(relation.purview),
        )
        for relation in top
    )
    mechanism_inclusion = _inclusion_order(
        tuple(frozenset(d.mechanism) for d in distinctions)
    )
    unions = tuple(
        frozenset(getattr(u, "index", u) for u in d.purview_union) for d in distinctions
    )
    purview_union_inclusion = _inclusion_order(unions)
    sums = ces.relations.sum_phi_by_distinction(distinctions)
    nodes = tuple(
        DistinctionNode(
            id=i,
            mechanism=tuple(d.mechanism),
            label=str(d.mechanism_label),
            cause_purview=tuple(d.cause_purview),
            effect_purview=tuple(d.effect_purview),
            mechanism_state=tuple(d.mechanism_state),
            phi=float(d.phi),
            sum_phi_relations=sums[i],
            includes=bool(purview_union_inclusion.covers[i]),
            included=any(i in c for c in purview_union_inclusion.covers),
        )
        for i, d in enumerate(distinctions)
    )
    return CESProjection(
        nodes=nodes,
        edges=edges,
        mechanism_inclusion=mechanism_inclusion,
        purview_union_inclusion=purview_union_inclusion,
        node_labels=node_labels,
        endpoints=_endpoints(distinctions, node_labels),
        faces=_faces(top, mechanism_to_id),
        degree_spectrum=ces.relations.degree_spectrum(),
    )
