# relabel.py
"""Relabel result objects through a node-index bijection.

Relabeling rewrites every node index in a result object through a
bijective mapping, reordering position-aligned state tuples and
transposing purview-shaped arrays to match the re-sorted index order.
No φ value changes: relabeling is an isomorphism of the frame, not a
recomputation.

Contract:

- Only complete IIT 4.0 structures are supported. Structure views
  (folds, induced substructures) raise — relabel the parent and
  re-derive the view. IIT 3.0 SIAs raise ``NotImplementedError``.
- Tie back-references are dropped: each relabeled object records only
  itself as its tie. Tie resolution is unaffected (the resolved member
  is what gets relabeled).
- ``node_labels`` for the target coordinates may be passed; when
  omitted, the structure's existing labels are reused, so the label of
  index ``j`` after relabeling is whatever the original labeling assigns
  to ``j``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pyphi.models.ces import CauseEffectStructure
from pyphi.models.ces import StructureView
from pyphi.models.distinction import Distinction
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import Part
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.models.state_specification import StateSpecification
from pyphi.models.state_specification import SystemStateSpecification
from pyphi.relations import AnalyticalRelations
from pyphi.relations import ConcreteRelations
from pyphi.relations import NullRelations
from pyphi.relations import Relation


def _argsorted_map(indices, mapping):
    """Map an index tuple, returning it re-sorted ascending together with
    the positional order that re-sorts any aligned tuple identically."""
    mapped = [mapping[i] for i in indices]
    order = sorted(range(len(mapped)), key=lambda k: mapped[k])
    return tuple(mapped[k] for k in order), order


def _reorder(aligned, order):
    if aligned is None:
        return None
    return tuple(aligned[k] for k in order)


def _transpose(array, purview_order, mapping):
    """Reorder the axes of a repertoire array to the re-sorted index order.

    Handles both array conventions: one axis per purview node
    (``ndim == len(purview_order)``) and one axis per system node in
    ascending index order (``ndim == len(mapping)``; off-purview axes have
    size 1). Scalar (0-d) arrays and ``None`` pass through. ``mapping``
    must cover exactly the system's node indices.
    """
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim == 0:
        return array
    if array.ndim == len(purview_order):
        return np.transpose(array, axes=purview_order)
    if array.ndim == len(mapping):
        old_indices = sorted(mapping)
        axes = sorted(range(len(old_indices)), key=lambda k: mapping[old_indices[k]])
        return np.transpose(array, axes=axes)
    raise NotImplementedError(
        f"cannot relabel an array of ndim {array.ndim} over a purview of "
        f"size {len(purview_order)} in a system of {len(mapping)} units"
    )


def relabel_state_specification(spec, mapping):
    purview, order = _argsorted_map(spec.purview, mapping)
    return StateSpecification(
        direction=spec.direction,
        purview=purview,
        state=_reorder(spec.state, order),  # pyright: ignore[reportArgumentType]  # None only if input state is None
        intrinsic_information=spec.intrinsic_information,
        repertoire=_transpose(spec.repertoire, order, mapping),  # pyright: ignore[reportArgumentType]  # None only if input is None
        unconstrained_repertoire=_transpose(  # pyright: ignore[reportArgumentType]  # None only if input is None
            spec.unconstrained_repertoire, order, mapping
        ),
        runner_up_state=_reorder(spec.runner_up_state, order),
        runner_up_intrinsic_information=spec.runner_up_intrinsic_information,
    )


def relabel_system_state(system_state, mapping):
    return SystemStateSpecification(
        cause=relabel_state_specification(system_state.cause, mapping),
        effect=relabel_state_specification(system_state.effect, mapping),
    )


def relabel_joint_partition(partition, mapping, node_labels=None):
    if partition is None:
        return None
    parts = tuple(
        Part(
            mechanism=tuple(sorted(mapping[i] for i in part.mechanism)),
            purview=tuple(sorted(mapping[i] for i in part.purview)),
            node_labels=node_labels,
        )
        for part in partition
    )
    return type(partition)(*parts, node_labels=node_labels)


def _relabel_partition(partition, mapping, node_labels=None):
    """Relabel either partition kind: mechanism-level ``JointPartition``
    (a sequence of ``Part`` blocks) or a system-level edge cut."""
    from pyphi.models.partitions import JointPartition

    if partition is None:
        return None
    if isinstance(partition, JointPartition):
        return relabel_joint_partition(partition, mapping, node_labels)
    return _relabel_system_partition(partition, mapping, node_labels)


def relabel_ria(ria, mapping, node_labels=None):
    mechanism, mechanism_order = _argsorted_map(ria.mechanism, mapping)
    purview, purview_order = _argsorted_map(ria.purview, mapping)
    specified_state = ria.specified_state
    return RepertoireIrreducibilityAnalysis(
        phi=ria.signed_phi,
        direction=ria.direction,
        mechanism=mechanism,
        purview=purview,
        partition=_relabel_partition(ria.partition, mapping, node_labels),  # pyright: ignore[reportArgumentType]  # preserves the input partition's kind
        repertoire=_transpose(ria.repertoire, purview_order, mapping),
        partitioned_repertoire=_transpose(
            ria.partitioned_repertoire, purview_order, mapping
        ),
        specified_state=(
            None
            if specified_state is None
            else relabel_state_specification(specified_state, mapping)
        ),
        mechanism_state=_reorder(ria.mechanism_state, mechanism_order),
        purview_state=_reorder(ria.purview_state, purview_order),
        node_labels=node_labels,
        selectivity=ria.selectivity,
        reasons=ria.reasons,
        signed_phi=ria.signed_phi,
    )


def relabel_mice(mice, mapping, node_labels=None):
    return type(mice)(relabel_ria(mice._ria, mapping, node_labels))


def relabel_distinction(distinction, mapping, node_labels=None):
    return Distinction(
        mechanism=tuple(sorted(mapping[i] for i in distinction.mechanism)),
        cause=relabel_mice(distinction.cause, mapping, node_labels),
        effect=relabel_mice(distinction.effect, mapping, node_labels),
    )


def _relabel_relations(relations, new_by_old):
    if isinstance(relations, NullRelations):
        return relations
    if isinstance(relations, ConcreteRelations):
        return ConcreteRelations(
            Relation(new_by_old[d] for d in relation) for relation in relations
        )
    if isinstance(relations, AnalyticalRelations):
        return AnalyticalRelations(ResolvedDistinctions(new_by_old.values()))
    raise TypeError(f"cannot relabel relations of type {type(relations).__name__}")


def _relabel_system_partition(partition, mapping, node_labels=None):
    if partition is None:
        return None
    if isinstance(partition, NullCut):
        return NullCut(tuple(sorted(mapping[i] for i in partition.indices)), node_labels)
    if hasattr(partition, "relabel"):
        return partition.relabel(
            tuple(mapping[i] for i in partition.node_indices), node_labels
        )
    raise NotImplementedError(
        f"cannot relabel a system partition of type {type(partition).__name__}"
    )


def relabel_sia(sia, mapping, node_labels=None):
    from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis

    if not isinstance(sia, SystemIrreducibilityAnalysis):
        raise NotImplementedError(
            "relabel supports IIT 4.0 system irreducibility analyses; got "
            f"{type(sia).__name__}"
        )
    node_indices, order = _argsorted_map(sia.node_indices, mapping)
    return dataclasses.replace(
        sia,
        phi=sia.signed_phi,
        normalized_phi=sia.signed_normalized_phi,
        signed_phi=sia.signed_phi,
        signed_normalized_phi=sia.signed_normalized_phi,
        partition=_relabel_system_partition(sia.partition, mapping, node_labels),
        cause=(
            None if sia.cause is None else relabel_ria(sia.cause, mapping, node_labels)
        ),
        effect=(
            None if sia.effect is None else relabel_ria(sia.effect, mapping, node_labels)
        ),
        system_state=(
            None
            if sia.system_state is None
            else relabel_system_state(sia.system_state, mapping)
        ),
        current_state=_reorder(sia.current_state, order),
        node_indices=node_indices,
        node_labels=node_labels,
    )


def relabel_ces(ces, mapping, node_labels=None) -> CauseEffectStructure:
    """Return ``ces`` rewritten through the node-index bijection ``mapping``.

    ``mapping`` must be injective and cover the structure's node indices.
    All φ values are preserved exactly; see the module docstring for the
    contract on ties, views, and labels.
    """
    from pyphi.condensation import _sia_node_indices

    if isinstance(ces, StructureView):
        raise ValueError(
            "cannot relabel a structure view; relabel the parent structure "
            "and re-derive the view"
        )
    if node_labels is None:
        node_labels = getattr(ces.sia, "node_labels", None)
    mapping = dict(mapping)
    indices = _sia_node_indices(ces.sia)
    if indices is None:
        raise ValueError("structure's SIA carries no node indices")
    if not set(mapping) >= set(indices):
        raise ValueError(f"mapping must cover all node indices {tuple(indices)}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("mapping must be injective")
    # Restrict to the system's own indices so array-axis order is
    # well-defined for system-shaped repertoires.
    mapping = {i: mapping[i] for i in indices}
    new_by_old = {
        d: relabel_distinction(d, mapping, node_labels) for d in ces.distinctions
    }
    return CauseEffectStructure(
        sia=relabel_sia(ces.sia, mapping, node_labels),
        distinctions=ResolvedDistinctions(new_by_old.values()),
        relations=_relabel_relations(ces.relations, new_by_old),
        config=ces.config,
    )
