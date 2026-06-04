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
    "DistinctionNode",
    "InclusionOrder",
    "PhiStructureProjection",
    "RelationEdge",
    "project_phi_structure",
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
class InclusionOrder:
    """The purview-inclusion partial order over distinctions.

    ``covers[i]`` lists the node ids that node ``i`` directly down-includes
    (the transitive reduction); ``rank[i]`` is the length of the longest
    down-chain below ``i`` (single-unit "points" have rank 0, the "whole"
    distinction the maximum), so it is monotonic in the partial order and
    suitable as a vertical layout coordinate.
    """

    covers: tuple[tuple[int, ...], ...]
    rank: tuple[int, ...]


@dataclass(frozen=True)
class PhiStructureProjection:
    """Everything a renderer needs to draw a phi-structure."""

    nodes: tuple[DistinctionNode, ...]
    edges: tuple[RelationEdge, ...]
    inclusion: InclusionOrder
    node_labels: NodeLabels


def _sum_phi_relations(n_nodes: int, edges: Sequence[RelationEdge]) -> tuple[float, ...]:
    """Per-node sum of relation phi over the edges involving each node."""
    sums = [0.0] * n_nodes
    for edge in edges:
        for i in edge.relata:
            sums[i] += edge.phi
    return tuple(sums)


def _inclusion_order(purview_unions: Sequence[frozenset]) -> InclusionOrder:
    """Partial order by strict subset relation on purview-unit sets."""
    n = len(purview_unions)
    below: list[set[int]] = [set() for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if a != b and purview_unions[b] < purview_unions[a]:
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
    return InclusionOrder(covers=covers, rank=rank)


def _unit_indices(units) -> tuple[int, ...]:
    """Sorted integer indices from an iterable of units (or bare ints)."""
    return tuple(sorted(getattr(u, "index", u) for u in units))


def project_phi_structure(ces, node_labels=None) -> PhiStructureProjection:
    """Project a |CauseEffectStructure| into plot-ready data."""
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
    unions = tuple(
        frozenset(getattr(u, "index", u) for u in d.purview_union) for d in distinctions
    )
    inclusion = _inclusion_order(unions)
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
            includes=bool(inclusion.covers[i]),
            included=any(i in c for c in inclusion.covers),
        )
        for i, d in enumerate(distinctions)
    )
    return PhiStructureProjection(
        nodes=nodes, edges=edges, inclusion=inclusion, node_labels=node_labels
    )
