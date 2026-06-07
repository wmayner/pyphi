"""Pure projection of IIT result objects into plot-ready data.

This package is the only part of :mod:`pyphi.visualize` that touches
result-object internals (:class:`Distinction`, :class:`Relation`). It imports
no plotting libraries; renderers consume the dataclasses defined here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
    """Plot-ready data for one degree-2 or degree-3 relation face."""

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
    ``endpoints[2 * d + 1]`` its effect. ``faces`` carries the degree-2 and
    degree-3 relation faces (the drawable simplices), referencing
    endpoints by id.
    """

    nodes: tuple[DistinctionNode, ...]
    edges: tuple[RelationEdge, ...]
    mechanism_inclusion: InclusionOrder
    purview_union_inclusion: InclusionOrder
    node_labels: NodeLabels
    endpoints: tuple[EndpointNode, ...] = ()
    faces: tuple[RelationFaceEdge, ...] = ()

    def inclusion(self, order: str) -> InclusionOrder:
        """The inclusion order named by ``order``.

        Args:
            order (str): ``"mechanism"`` or ``"purview_union"``.
        """
        if order == "mechanism":
            return self.mechanism_inclusion
        if order == "purview_union":
            return self.purview_union_inclusion
        raise ValueError(f"unknown order {order!r}")


def _sum_phi_relations(n_nodes: int, edges: Sequence[RelationEdge]) -> tuple[float, ...]:
    """Per-node sum of relation phi over the edges involving each node."""
    sums = [0.0] * n_nodes
    for edge in edges:
        for i in edge.relata:
            sums[i] += edge.phi
    return tuple(sums)


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
    by_degree = relations.faces_by_degree
    faces = []
    for degree in (2, 3):
        for face in by_degree.get(degree, ()):
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
                    degree=degree,
                    phi=float(face.phi),
                    overlap=_unit_indices(face.overlap),
                )
            )
    faces.sort(key=lambda f: (f.degree, f.endpoints, f.phi))
    return tuple(faces)


def project_ces(ces, node_labels=None) -> CESProjection:
    """Project a |CauseEffectStructure| into plot-ready data."""
    from pyphi.models.ces import PhiFold

    if isinstance(ces, PhiFold):
        raise TypeError(
            "cannot project a PhiFold (its relations may reference distinctions "
            "outside the fold); use highlight_phi_fold to visualize a fold"
        )
    distinctions = list(ces.distinctions)
    if node_labels is None:
        node_labels = distinctions[0].node_labels
    mechanism_to_id = {tuple(d.mechanism): i for i, d in enumerate(distinctions)}
    edges = tuple(
        RelationEdge(
            relata=tuple(sorted(mechanism_to_id[tuple(m)] for m in relation.mechanisms)),
            degree=len(relation),
            phi=float(relation.phi),
            overlap=_unit_indices(relation.purview),
        )
        for relation in ces.relations
    )
    mechanism_inclusion = _inclusion_order(
        tuple(frozenset(d.mechanism) for d in distinctions)
    )
    unions = tuple(
        frozenset(getattr(u, "index", u) for u in d.purview_union) for d in distinctions
    )
    purview_union_inclusion = _inclusion_order(unions)
    sums = _sum_phi_relations(len(distinctions), edges)
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
        faces=_faces(ces.relations, mechanism_to_id),
    )
