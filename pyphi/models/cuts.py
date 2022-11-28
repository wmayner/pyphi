#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/cuts.py

"""Objects that represent partitions of sets of nodes."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np

from .. import connectivity, utils
from ..labels import NodeLabels
from . import cmp, fmt


class _CutBase:
    """Base class for all unidirectional system cuts.

    Concrete cut classes must implement a ``cut_matrix`` method and an
    ``indices`` property. See ``Cut`` for a concrete example.
    """

    @property
    def indices(self):
        """Indices of this cut."""
        raise NotImplementedError

    def cut_matrix(self, n):
        """Return the cut matrix for this cut.

        The cut matrix is a square matrix representing  connections severed
        by the cut: if the connection from node `a` to node `b` is cut,
        `cut_matrix[a, b]` is `1`; otherwise it is `0`.

        Args:
           n (int): The size of the network.
        """
        raise NotImplementedError

    @property
    def is_null(self):
        """Is this cut a null cut?

        All concrete cuts should return ``False``.
        """
        return False

    def apply_cut(self, cm):
        """Return a modified connectivity matrix with all connections that are
        severed by this cut removed.

        Args:
            cm (np.ndarray): A connectivity matrix.
        """
        # Invert the cut matrix, creating a matrix of preserved connections
        inverse = np.logical_not(self.cut_matrix(cm.shape[0])).astype(int)
        return cm * inverse

    def cuts_connections(self, a, b):
        """Check if this cut severs any connections from ``a`` to ``b``.

        Args:
            a (tuple[int]): A set of nodes.
            b (tuple[int]): A set of nodes.
        """
        n = max(self.indices + a + b) + 1
        return self.cut_matrix(n)[np.ix_(a, b)].any()

    def splits_mechanism(self, mechanism):
        """Check if this cut splits a mechanism.

        Args:
            mechanism (tuple[int]): The mechanism in question.

        Returns:
            bool: ``True`` if `mechanism` has elements on both sides of the
            cut; ``False`` otherwise.
        """
        return self.cuts_connections(mechanism, mechanism)

    def all_cut_mechanisms(self):
        """Return all mechanisms with elements on both sides of this cut.

        Yields:
            tuple[int]: The next cut mechanism.
        """
        for mechanism in utils.powerset(self.indices, nonempty=True):
            if self.splits_mechanism(mechanism):
                yield mechanism


class NullCut(_CutBase):
    """The cut that does nothing."""

    def __init__(self, indices, node_labels=None):
        self._indices = indices
        self.node_labels = node_labels

    @property
    def is_null(self):
        """This is the only cut where ``is_null == True``."""
        return True

    @property
    def indices(self):
        """Indices of the cut."""
        return self._indices

    def cut_matrix(self, n):
        """Return a matrix of zeros."""
        return np.zeros((n, n))

    def to_json(self):
        return {"indices": self.indices}

    def __repr__(self):
        return fmt.make_repr(self, ["indices"])

    def __str__(self):
        return "NullCut({})".format(self.indices)

    @cmp.sametype
    def __eq__(self, other):
        return self.indices == other.indices

    def __hash__(self):
        return hash(self.indices)


class Cut(_CutBase):
    """Represents a unidirectional cut.

    Attributes:
        from_nodes (tuple[int]): Connections from this group of nodes to those
            in ``to_nodes`` are from_nodes.
        to_nodes (tuple[int]): Connections to this group of nodes from those in
            ``from_nodes`` are from_nodes.
    """

    # Don't construct an attribute dictionary; see
    # https://docs.python.org/3.3/reference/datamodel.html#notes-on-using-slots
    __slots__ = ("from_nodes", "to_nodes", "node_labels")

    def __init__(self, from_nodes, to_nodes, node_labels=None):
        self.from_nodes = from_nodes
        self.to_nodes = to_nodes
        self.node_labels = node_labels

    @property
    def indices(self):
        """Indices of this cut."""
        return tuple(sorted(set(self.from_nodes + self.to_nodes)))

    def cut_matrix(self, n):
        """Compute the cut matrix for this cut.

        The cut matrix is a square matrix which represents connections severed
        by the cut.

        Args:
           n (int): The size of the network.

        Example:
            >>> cut = Cut((1,), (2,))
            >>> cut.cut_matrix(3)
            array([[0., 0., 0.],
                   [0., 0., 1.],
                   [0., 0., 0.]])
        """
        return connectivity.relevant_connections(n, self.from_nodes, self.to_nodes)

    @cmp.sametype
    def __eq__(self, other):
        return self.from_nodes == other.from_nodes and self.to_nodes == other.to_nodes

    def __hash__(self):
        return hash((self.from_nodes, self.to_nodes))

    def __repr__(self):
        return fmt.make_repr(self, ["from_nodes", "to_nodes"])

    def __str__(self):
        return fmt.fmt_cut(self)

    def __len__(self):
        """The number of parts in the Cut."""
        # TODO(4.0) generalize this when/if general Partition object is used
        return 2

    def format(self, node_labels=None):
        return fmt.fmt_part(self, node_labels=node_labels)

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {"from_nodes": self.from_nodes, "to_nodes": self.to_nodes}

    @classmethod
    def from_json(cls, data):
        """Return a Cut object from a JSON-serializable representation."""
        return cls(data["from_nodes"], data["to_nodes"])


class SystemPartition(Cut):
    """A system partition.

    Same as a IIT 3.0 unidirectional partition, but with a Direction.
    """

    def __init__(self, direction, *args, **kwargs):
        self.direction = direction
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return fmt.fmt_cut(self, direction=self.direction)

    def to_json(self):
        return {
            "direction": self.direction,
            **super().to_json(),
        }

    @classmethod
    def from_json(cls, data):
        """Return a SystemPartition object from a JSON-serializable representation."""
        return cls(data["direction"], data["from_nodes"], data["to_nodes"])


class CompleteSystemPartition:
    """Represents the SystemPartition that destroys all distinctions & relations."""

    def __repr__(self):
        return "Complete"


class KCut(_CutBase):
    """A cut that severs all connections between parts of a K-partition."""

    def __init__(self, direction, partition, node_labels=None):
        self.direction = direction
        self.partition = partition
        self.node_labels = node_labels

    @property
    def indices(self):
        assert set(self.partition.mechanism) == set(self.partition.purview)
        return self.partition.mechanism

    def cut_matrix(self, n):
        """The matrix of connections that are severed by this cut."""
        cm = np.zeros((n, n))

        for part in self.partition:
            from_, to = self.direction.order(part.mechanism, part.purview)
            # All indices external to this part
            external = tuple(set(self.indices) - set(to))
            cm[np.ix_(from_, external)] = 1

        return cm

    @cmp.sametype
    def __eq__(self, other):
        return self.partition == other.partition and self.direction == other.direction

    def __hash__(self):
        return hash((self.direction, self.partition))

    def __repr__(self):
        return fmt.make_repr(self, ["direction", "partition"])

    # TODO: improve
    def __str__(self):
        return fmt.fmt_kcut(self)

    def to_json(self):
        return {"direction": self.direction, "partition": self.partition}


class ActualCut(KCut):
    """Represents an cut for a |Transition|."""

    @property
    def indices(self):
        return tuple(sorted(set(self.partition.mechanism + self.partition.purview)))


class GeneralKCut(_CutBase):
    """A cut defined by a matrix of cut connections."""

    def __init__(self, node_indices, cut_matrix, node_labels=None):
        self.node_indices = node_indices
        self._cut_matrix = cut_matrix
        self.node_labels = node_labels

    def normalization_factor(self):
        """The normalization factor for this cut."""
        return 1 / np.sum(self._cut_matrix)

    @property
    def indices(self):
        return self.node_indices

    def cut_matrix(self, n):
        """The matrix of connections that are severed by this cut."""
        cm = np.zeros([n, n])
        cm[np.ix_(self.node_indices, self.node_indices)] = self._cut_matrix
        return cm

    @cmp.sametype
    def __eq__(self, other):
        return self.node_indices == other.node_indices and np.array_equal(
            self._cut_matrix, other._cut_matrix
        )

    def __hash__(self):
        return hash((self.node_indices, utils.np_hash(self._cut_matrix)))

    def __repr__(self):
        return fmt.make_repr(self, ["node_indices", "_cut_matrix"])

    def __str__(self):
        # TODO: improve
        return str(self._cut_matrix)

    def to_json(self):
        raise NotImplementedError


class CompleteGeneralKCut(GeneralKCut):
    def __init__(self, node_indices, node_labels=None):
        self.node_indices = node_indices
        self.node_labels = node_labels
        self._cut_matrix = np.ones([len(node_indices), len(node_indices)], dtype=int)

    def normalization_factor(self):
        """The normalization factor for this cut."""
        return 1 / len(self.node_indices)


class GeneralSetPartition(GeneralKCut):
    def __init__(self, *args, set_partition=None, **kwargs):
        self.set_partition = set_partition
        super().__init__(*args, **kwargs)
        self.parts = [
            [self.node_indices[i] for i in part] for part in self.set_partition
        ]

    @property
    def num_parts(self):
        return len(self.set_partition)

    def __str__(self):
        if self.node_labels is not None:
            parts = map(self.node_labels.coerce_to_labels, self.parts)
        else:
            parts = map(str, self.parts)
        return (
            f"{self.num_parts} parts: "
            + "{"
            + ",".join("".join(part) for part in parts)
            + "}\n"
            + super().__str__()
        )

    def to_json(self):
        return self.set_partition

    # TODO(4.0) add to other classes after consolidating partitions
    def relabel(self, node_indices, node_labels=None):
        if node_labels is None:
            node_labels = self.node_labels
        if not len(node_indices) == len(self.node_indices):
            raise ValueError("New node indices must have same length as the old.")
        return GeneralSetPartition(
            node_indices,
            self._cut_matrix,
            set_partition=self.set_partition,
            node_labels=node_labels,
        )


class CompleteGeneralSetPartition(CompleteGeneralKCut):
    def __str__(self):
        return "Complete\n" + super().__str__()


@dataclass(order=True)
class Part:
    """Represents one part of a |Bipartition|.

    Attributes:
        mechanism (tuple[int]): The nodes in the mechanism for this part.
        purview (tuple[int]): The nodes in the mechanism for this part.

    Example:
        When calculating |small_phi| of a 3-node subsystem, we partition the
        system in the following way::

            mechanism:  A,C    B
                        ─── ✕ ───
              purview:   B    A,C

        This class represents one term in the above product.
    """

    mechanism: tuple
    purview: tuple
    node_labels: NodeLabels = None

    def __hash__(self):
        return hash((self.mechanism, self.purview))

    def __eq__(self, other):
        return (self.mechanism == other.mechanism) and (self.purview == other.purview)

    def __repr__(self):
        return fmt.fmt_part(self, node_labels=self.node_labels)

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {"mechanism": self.mechanism, "purview": self.purview}


class KPartition(Sequence, _CutBase):
    """A partition with an arbitrary number of parts."""

    __slots__ = ["parts", "node_labels", "_mechanism", "_purview"]

    def __init__(self, *parts, node_labels=None):
        self.parts = parts
        self.node_labels = node_labels
        self._mechanism = None
        self._purview = None

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, index):
        return self.parts[index]

    def __eq__(self, other):
        if not isinstance(other, KPartition):
            return NotImplemented
        return self.parts == other.parts

    def __hash__(self):
        return hash(self.parts)

    def __str__(self):
        return fmt.fmt_partition(self)

    def __repr__(self):
        return fmt.make_repr(self, ["parts", "node_labels"])

    @property
    def mechanism(self):
        """tuple[int]: The nodes of the mechanism in the partition."""
        # TODO(4.0) do we need to sort here? slow
        if self._mechanism is None:
            self._mechanism = tuple(
                chain.from_iterable(part.mechanism for part in self)
            )
        return self._mechanism

    @property
    def purview(self):
        """tuple[int]: The nodes of the purview in the partition."""
        # TODO(4.0) do we need to sort here? slow
        if self._purview is None:
            self._purview = tuple(chain.from_iterable(part.purview for part in self))
        return self._purview

    @property
    def indices(self):
        return tuple(sorted(set(self.mechanism + self.purview)))

    def normalize(self):
        """Normalize the order of parts in the partition."""
        return type(self)(*sorted(self), node_labels=self.node_labels)

    def num_connections_cut(self):
        """The number of connections cut by this partition."""
        n = 0
        purview_lengths = [len(part.purview) for part in self.parts]
        for i, part in enumerate(self.parts):
            n += len(part.mechanism) * (
                sum(purview_lengths[:i]) + sum(purview_lengths[i + 1 :])
            )
        return n

    # TODO(4.0) consolidate cut classes
    def cut_matrix(self, n):
        """The matrix of connections that are severed by this cut."""
        cm = np.zeros((n, n))

        for part in self.parts:
            # Indices of all other part's purviews
            outside_part = tuple(set(self.purview) - set(part.purview))
            cm[np.ix_(part.mechanism, outside_part)] = 1

        return cm

    def to_json(self):
        return {"parts": list(self)}

    @classmethod
    def from_json(cls, dct):
        return cls(*dct["parts"])


class Bipartition(KPartition):
    """A bipartition of a mechanism and purview.

    Attributes:
        part0 (Part): The first part of the partition.
        part1 (Part): The second part of the partition.
    """

    __slots__ = KPartition.__slots__

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {"part0": self[0], "part1": self[1]}

    @classmethod
    def from_json(cls, dct):
        return cls(dct["part0"], dct["part1"])


class Tripartition(KPartition):
    """A partition with three parts."""

    __slots__ = KPartition.__slots__


class RelationPart(Part):
    """A part of a relation-style partition."""

    def __init__(self, mechanism, purview, relata, node_labels=None):
        self.relata = relata
        super().__init__(mechanism, purview, node_labels=node_labels)

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            "mechanism": self.mechanism,
            "purview": self.purview,
            "relata": self.relata,
        }

    def to_indirect_json(self):
        """Return an indirect representation of the Part.

        This uses the integer indices of distinctions in the given CES rather
        than the objects themselves, which is more efficient for storage on
        disk.
        """
        return [self.mechanism, self.purview]

    @classmethod
    def from_indirect_json(cls, relata, data, node_labels=None):
        mechanism, purview = data
        return cls(mechanism, purview, relata, node_labels=node_labels)

    def __repr__(self):
        numer = (
            ", ".join(
                fmt.fmt_relatum(relatum, node_labels=self.node_labels)
                for relatum in [self.relata[i] for i in self.mechanism]
            )
            if self.mechanism
            else fmt.EMPTY_SET
        )
        denom = fmt.fmt_nodes(self.purview, node_labels=self.node_labels)
        return fmt.fmt_fraction(numer=numer, denom=denom)


class RelationPartition(Tripartition):
    def __init__(self, relata, *parts, node_labels=None):
        self.relata = relata
        super().__init__(*parts, node_labels=node_labels)
        self._purview = None

    @property
    def purview(self):
        if self._purview is None:
            self._purview = set(super().purview)
        return self._purview

    # TODO(4.0) refactor to use relatum objects themselves, rather than indices
    # - avoids need for O(1) integer indexing of OrderedSet; can replace with
    #   simpler implementation, maybe just a >3.7 dict
    def for_relatum(self, i):
        """Return the implied `Tripartition` with respect to just a single mechanism.

        Arguments:
            i (int): The index of the relatum in the relata object.
        """
        relatum = self.relata[i]
        nonoverlap_purview_elements = tuple(set(relatum.purview) - set(self.purview))
        return Tripartition(
            *[
                Part(
                    mechanism=(relatum.mechanism if i in part.mechanism else ()),
                    purview=tuple(
                        part.purview
                        # Non-overlapping purview elements are included only
                        # once, in the part corresponding to this mechanism
                        + (nonoverlap_purview_elements if i in part.mechanism else ())
                    ),
                    node_labels=part.node_labels,
                )
                for part in self.parts
            ]
        )

    def to_indirect_json(self):
        """Return an indirect representation of the Partition.

        This uses the integer indices of distinctions in the given CES rather
        than the objects themselves, which is more efficient for storage on
        disk.
        """
        return [part.to_indirect_json() for part in self]

    @classmethod
    def from_indirect_json(cls, relata, data, node_labels=None):
        return cls(
            relata,
            *[
                RelationPart.from_indirect_json(relata, part, node_labels=node_labels)
                for part in data
            ],
            node_labels=node_labels,
        )


class CompleteRelationPartition(RelationPartition):
    """A partition of a relation that cuts a singel relatum away from its entire
    mechanism and leaves the others untouched.
    """

    def __init__(self, relata, candidate_joint_purview, i, node_labels=None):
        if i >= len(relata):
            raise ValueError(f"i must be < len(relata) = {len(relata)}; got {i}")
        self.relata = relata
        self.i = i
        self._purview = candidate_joint_purview
        self.node_labels = node_labels

    def __repr__(self):
        return f"CompleteRelationPartition[{self.i}]"

    # TODO(4.0) refactor to use relatum objects themselves, rather than indices
    # - avoids need for O(1) integer indexing of OrderedSet; can replace with
    #   simpler implementation, maybe just a >3.7 dict
    def for_relatum(self, i):
        """Return the implied `Tripartition` with respect to just a single mechanism.

        Arguments:
            i (int): The index of the relatum in the relata object.
        """
        relatum = self.relata[i]
        if i == self.i:
            return Tripartition(
                Part(
                    mechanism=relatum.mechanism,
                    purview=(),
                    node_labels=self.node_labels,
                ),
                Part(
                    mechanism=(),
                    purview=tuple(relatum.purview),
                    node_labels=self.node_labels,
                ),
            )
        return Tripartition(
            Part(
                mechanism=tuple(relatum.mechanism),
                purview=tuple(relatum.purview),
                node_labels=self.node_labels,
            ),
        )
