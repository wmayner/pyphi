"""Partition and edge-cut value types.

Two distinct mathematical concepts share this module's
:class:`_PartitionBase` interface:

**Vertex partitions** (IIT 4.0 paper terminology):

- :class:`DirectedBipartition` — a directed bipartition of an index set
  (Θ(S), Eq. 14-18). Stores ``(direction, from_nodes, to_nodes)``.
- :class:`JointPartition` — a sequence of :class:`Part` blocks, each a
  ``(mechanism, purview)`` pair (Π(M,Z), Eq. 5-7). Subclasses
  :class:`JointBipartition` (k=2) and :class:`JointTripartition` (k=3,
  wedge constraint).
- :class:`DirectedJointPartition` — a :class:`JointPartition` together
  with a :class:`Direction`, used for AC (Ψ, Eq. 7 of Albantakis et al.
  2019) and for disintegrating partitions (Θ(M,Z), Eq. 38 of IIT 4.0).
- :class:`DirectedSetPartition` — a k-way set partition with per-part
  direction.

**Edge cuts** (graph theory terminology):

- :class:`EdgeCut` — an explicit n by n binary severance matrix.
- :class:`CompleteEdgeCut` — all edges severed (boundary).
- :class:`NullCut` — no edges severed (identity).

A vertex partition *induces* an edge cut: every concrete
:class:`_PartitionBase` exposes :meth:`cut_matrix(n)` returning the
binary matrix of severed connections. :meth:`apply_cut(cm)` applies
the induced cut to a connectivity matrix.

The boundary classes (:class:`CompleteJointPartition`,
:class:`AtomicJointPartition`) live in :mod:`pyphi.partition` alongside
the partition generators.

**Use-case mapping:**

- IIT 3.0 / IIT 4.0 SIA partitioning → :class:`DirectedBipartition` and
  :class:`DirectedSetPartition` (via the ``DIRECTED_BI``, ``SET_UNI/BI``,
  etc. scheme registries).
- IIT 4.0 system SIA general scheme → :class:`EdgeCut` (matrix-based).
- IIT 4.0 mechanism MIP search → :class:`JointPartition` and subclasses
  (via the ``BI``, ``TRI``, ``ALL`` mechanism partition schemes).
- Actual causation distinction finding → :class:`DirectedJointPartition`.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any
from typing import Self

import numpy as np
from numpy.typing import NDArray

from pyphi import connectivity
from pyphi import utils
from pyphi.direction import Direction
from pyphi.labels import NodeLabels

from . import cmp
from . import fmt


class _PartitionBase:
    """Base class for partitions and edge cuts.

    Concrete subclasses implement :meth:`cut_matrix` and the
    :attr:`indices` property. :meth:`apply_cut`, :meth:`cuts_connections`,
    :meth:`splits_mechanism`, and :meth:`all_cut_mechanisms` are derived.
    """

    @property
    def indices(self) -> tuple[int, ...]:
        """Indices of the partitioned nodes."""
        raise NotImplementedError

    def cut_matrix(self, n: int) -> NDArray[np.int_]:
        """Return the binary edge-cut matrix induced by this partition.

        ``cut_matrix[a, b] == 1`` iff the directed connection a→b is
        severed.

        Args:
            n (int): The size of the substrate.
        """
        raise NotImplementedError

    @property
    def is_null(self) -> bool:
        """``True`` if this partition severs no connections."""
        return False

    def apply_cut(self, cm: NDArray[np.int_]) -> NDArray[np.int_]:
        """Return ``cm`` with the partition's induced edge cut removed.

        Args:
            cm (np.ndarray): A connectivity matrix.
        """
        inverse = np.logical_not(self.cut_matrix(cm.shape[0])).astype(int)
        return cm * inverse

    def cuts_connections(self, a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        """Whether this partition severs any connection from ``a`` to ``b``."""
        n = max(self.indices + a + b) + 1
        return bool(self.cut_matrix(n)[np.ix_(a, b)].any())

    def splits_mechanism(self, mechanism: tuple[int, ...]) -> bool:
        """Whether this partition splits ``mechanism`` across its parts."""
        return self.cuts_connections(mechanism, mechanism)

    def all_cut_mechanisms(self) -> Iterator[tuple[int, ...]]:
        """Yield all mechanisms with elements split by this partition."""
        for mechanism in utils.powerset(self.indices, nonempty=True):
            if self.splits_mechanism(mechanism):
                yield mechanism


class NullCut(_PartitionBase):
    """The empty edge cut: no connections severed."""

    def __init__(
        self, indices: tuple[int, ...], node_labels: NodeLabels | None = None
    ) -> None:
        self._indices = indices
        self.node_labels = node_labels

    @property
    def is_null(self) -> bool:
        return True

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    def cut_matrix(self, n: int) -> NDArray[np.int_]:
        return np.zeros((n, n), dtype=int)

    def to_json(self) -> dict[str, Any]:
        return {"indices": self.indices}

    def __repr__(self) -> str:
        return fmt.make_repr(self, ["indices"])

    def __str__(self) -> str:
        return f"NullCut({self.indices})"

    @cmp.sametype
    def __eq__(self, other: object) -> bool:
        return self.indices == other.indices  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(self.indices)


class DirectedBipartition(_PartitionBase):
    """A directed bipartition of an index set.

    Severs connections from ``from_nodes`` to ``to_nodes`` in a causal
    ``direction`` (CAUSE or EFFECT). Corresponds to θ ∈ Θ(S) in IIT 4.0
    Eq. 14-18 in the bipartite case.

    Attributes:
        direction: The causal direction of the cut.
        from_nodes: Source side; connections from these to ``to_nodes`` are severed.
        to_nodes: Target side; connections from ``from_nodes`` to these are severed.
        node_labels: Optional labels for pretty-printing.
    """

    __slots__ = ("direction", "from_nodes", "node_labels", "to_nodes")

    direction: Direction
    from_nodes: tuple[int, ...]
    to_nodes: tuple[int, ...]
    node_labels: NodeLabels | None

    def __init__(
        self,
        direction: Direction,
        from_nodes: tuple[int, ...],
        to_nodes: tuple[int, ...],
        node_labels: NodeLabels | None = None,
    ) -> None:
        self.direction = direction
        self.from_nodes = from_nodes
        self.to_nodes = to_nodes
        self.node_labels = node_labels

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.from_nodes + self.to_nodes)))

    def cut_matrix(self, n: int) -> NDArray[np.int_]:
        """Connections from ``from_nodes`` to ``to_nodes`` are severed.

        Example:
            >>> from pyphi.direction import Direction
            >>> sp = DirectedBipartition(Direction.EFFECT, (1,), (2,))
            >>> sp.cut_matrix(3)
            array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 0, 0]])
        """
        return connectivity.relevant_connections(
            n, self.from_nodes, self.to_nodes
        ).astype(np.int_)

    @cmp.sametype
    def __eq__(self, other: object) -> bool:
        return (
            self.direction == other.direction  # type: ignore[attr-defined]
            and self.from_nodes == other.from_nodes  # type: ignore[attr-defined]
            and self.to_nodes == other.to_nodes  # type: ignore[attr-defined]
        )

    def __hash__(self) -> int:
        return hash((self.direction, self.from_nodes, self.to_nodes))

    def __repr__(self) -> str:
        return fmt.fmt_partition_arrow(self, direction=self.direction)

    def __str__(self) -> str:
        return fmt.fmt_partition_arrow(self, direction=self.direction)

    def __len__(self) -> int:
        return 2

    def format(self, node_labels: NodeLabels | None = None) -> str:
        return fmt.fmt_part(self, node_labels=node_labels)

    def to_json(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "from_nodes": self.from_nodes,
            "to_nodes": self.to_nodes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DirectedBipartition:
        return cls(data["direction"], data["from_nodes"], data["to_nodes"])


class DirectedJointPartition(_PartitionBase):
    """A joint partition with a causal direction.

    Wraps a :class:`JointPartition` with a :class:`Direction`. Corresponds
    to disintegrating partitions Θ(M,Z) in IIT 4.0 Eq. 38 and to AC
    partitions ψ in Albantakis et al. 2019 Eq. 7.

    Attributes:
        direction: Causal direction of the induced edge cut.
        partition: The joint partition (sequence of (mechanism, purview) parts).
        node_labels: Optional labels for pretty-printing.
    """

    direction: Direction
    partition: JointPartition
    node_labels: NodeLabels | None

    def __init__(
        self,
        direction: Direction,
        partition: JointPartition,
        node_labels: NodeLabels | None = None,
    ) -> None:
        self.direction = direction
        self.partition = partition
        self.node_labels = node_labels

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.partition.mechanism + self.partition.purview)))

    def cut_matrix(self, n: int) -> NDArray[np.int_]:
        cm = np.zeros((n, n), dtype=int)
        for part in self.partition:
            from_, to = self.direction.order(part.mechanism, part.purview)
            external = tuple(set(self.indices) - set(to))
            cm[np.ix_(from_, external)] = 1
        return cm

    @cmp.sametype
    def __eq__(self, other: object) -> bool:
        return self.partition == other.partition and self.direction == other.direction  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((self.direction, self.partition))

    def __repr__(self) -> str:
        return fmt.make_repr(self, ["direction", "partition"])

    def __str__(self) -> str:
        return fmt.fmt_directed_joint_partition(self)

    def to_json(self) -> dict[str, Any]:
        return {"direction": self.direction, "partition": self.partition}


class EdgeCut(_PartitionBase):
    """An edge cut specified by an explicit binary severance matrix.

    Stores ``(node_indices, _cut_matrix)`` where ``_cut_matrix[i, j] == 1``
    indicates that the connection from node ``node_indices[i]`` to
    ``node_indices[j]`` is severed. The full n by n cut matrix is produced
    by embedding ``_cut_matrix`` at the rows/cols corresponding to
    ``node_indices``.
    """

    node_indices: tuple[int, ...]
    _cut_matrix: NDArray[np.int_]
    node_labels: NodeLabels | None

    def __init__(
        self,
        node_indices: tuple[int, ...],
        cut_matrix: NDArray[np.int_],
        node_labels: NodeLabels | None = None,
    ) -> None:
        self.node_indices = node_indices
        self._cut_matrix = cut_matrix
        self.node_labels = node_labels

    def normalization_factor(self) -> float:
        """Normalization factor: 1 / number of severed connections."""
        return float(1 / np.sum(self._cut_matrix))

    @property
    def indices(self) -> tuple[int, ...]:
        return self.node_indices

    def cut_matrix(self, n: int) -> NDArray[np.int_]:
        cm = np.zeros([n, n], dtype=int)
        cm[np.ix_(self.node_indices, self.node_indices)] = self._cut_matrix
        return cm

    @cmp.sametype
    def __eq__(self, other: object) -> bool:
        return (
            self.node_indices == other.node_indices  # type: ignore[attr-defined]
            and np.array_equal(
                self._cut_matrix,
                other._cut_matrix,  # type: ignore[attr-defined]
            )
        )

    def __hash__(self) -> int:
        return hash((self.node_indices, utils.np_hash(self._cut_matrix)))

    def __repr__(self) -> str:
        return fmt.make_repr(self, ["node_indices", "_cut_matrix"])

    def __str__(self) -> str:
        return str(self._cut_matrix)

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EdgeCut:
        return cls(
            node_indices=data["node_indices"],
            cut_matrix=data["_cut_matrix"],
            node_labels=data["node_labels"],
        )


class CompleteEdgeCut(EdgeCut):
    """Edge cut that severs every connection (all-ones matrix).

    Used as the boundary case in partition enumeration and as the
    "complete" cut against which an SIA's partition is compared.
    """

    def __init__(
        self, node_indices: tuple[int, ...], node_labels: NodeLabels | None = None
    ) -> None:
        self.node_indices = node_indices
        self.node_labels = node_labels
        self._cut_matrix = np.ones([len(node_indices), len(node_indices)], dtype=int)

    def normalization_factor(self) -> float:
        return 1 / len(self.node_indices)

    def __repr__(self) -> str:
        return "Complete"


class DirectedSetPartition(EdgeCut):
    """A k-way set partition of nodes with per-part directional cuts.

    Stores both the explicit severance matrix (inherited from
    :class:`EdgeCut`) and the semantic set-partition structure
    (``set_partition``: which node indices group together; ``parts``:
    the corresponding actual node indices).
    """

    set_partition: list[list[int]]
    parts: list[list[int]]

    def __init__(
        self,
        node_indices: tuple[int, ...],
        cut_matrix: NDArray[np.int_],
        set_partition: list[list[int]],
        node_labels: NodeLabels | None = None,
    ) -> None:
        self.set_partition = set_partition
        super().__init__(node_indices, cut_matrix, node_labels)
        self.parts = [
            [self.node_indices[i] for i in part] for part in self.set_partition
        ]

    @property
    def num_parts(self) -> int:
        return len(self.set_partition)

    def __str__(self) -> str:
        if self.node_labels is not None:
            parts = map(self.node_labels.coerce_to_labels, self.parts)
        else:
            parts = map(str, self.parts)  # type: ignore[arg-type]
        return (
            f"{self.num_parts} parts: "
            + "{"
            + ",".join("".join(str(x) for x in part) for part in parts)
            + "}\n"
            + super().__str__()
        )

    def to_json(self) -> dict[str, Any]:
        dct = self.__dict__.copy()
        del dct["parts"]
        return dct

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DirectedSetPartition:
        data["cut_matrix"] = np.array(data.pop("_cut_matrix"))
        return cls(**data)

    def relabel(
        self,
        node_indices: tuple[int, ...],
        node_labels: NodeLabels | None = None,
    ) -> DirectedSetPartition:
        if node_labels is None:
            node_labels = self.node_labels
        if not len(node_indices) == len(self.node_indices):
            raise ValueError("New node indices must have same length as the old.")
        return DirectedSetPartition(
            node_indices,
            self._cut_matrix,
            set_partition=self.set_partition,
            node_labels=node_labels,
        )


@dataclass(order=True, frozen=True)
class Part:
    """One block of a :class:`JointPartition`.

    Attributes:
        mechanism: Nodes on the mechanism side of this block.
        purview: Nodes on the purview side of this block.

    Example:
        For a |small_phi| computation on a 3-node system, a 2-block
        partition could be::

            mechanism:  A,C    B
                        ─── ✕ ───
              purview:   B    A,C
    """

    mechanism: tuple[int, ...]
    purview: tuple[int, ...]
    node_labels: NodeLabels | None = None

    def __hash__(self) -> int:
        return hash((self.mechanism, self.purview))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Part):
            return NotImplemented
        return (self.mechanism == other.mechanism) and (self.purview == other.purview)

    def __repr__(self) -> str:
        return fmt.fmt_part(self, node_labels=self.node_labels)

    def to_json(self) -> dict[str, Any]:
        return {"mechanism": self.mechanism, "purview": self.purview}


class JointPartition(Sequence[Part], _PartitionBase):
    """A joint partition of a (mechanism, purview) pair into k matched parts.

    Stores a sequence of :class:`Part` blocks. Each Part pairs a
    mechanism subset with a purview subset; the mechanism subsets across
    all Parts form a partition of the union mechanism, and likewise for
    the purview subsets. The two side-partitions are matched index-by-index.

    Corresponds to Π(M,Z) in IIT 4.0 Eq. 5-7. Subclasses
    :class:`JointBipartition` (k=2) and :class:`JointTripartition` (k=3)
    add semantic markers.
    """

    __slots__ = ["_mechanism", "_purview", "node_labels", "parts"]

    parts: tuple[Part, ...]
    node_labels: NodeLabels | None
    _mechanism: tuple[int, ...] | None
    _purview: tuple[int, ...] | None

    def __init__(self, *parts: Part, node_labels: NodeLabels | None = None) -> None:
        self.parts = parts
        self.node_labels = node_labels
        self._mechanism = None
        self._purview = None

    def __len__(self) -> int:
        return len(self.parts)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, index: int) -> Part:  # type: ignore[override]
        return self.parts[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointPartition):
            return NotImplemented
        return self.parts == other.parts

    def __hash__(self) -> int:
        return hash(self.parts)

    def __str__(self) -> str:
        return fmt.fmt_partition(self)

    def __repr__(self) -> str:
        return fmt.make_repr(self, ["parts", "node_labels"])

    @property
    def mechanism(self) -> tuple[int, ...]:
        if self._mechanism is None:
            self._mechanism = tuple(chain.from_iterable(part.mechanism for part in self))
        return self._mechanism

    @property
    def purview(self) -> tuple[int, ...]:
        if self._purview is None:
            # Sort because downstream callers index by sorted purview
            # (e.g., System.partitioned_repertoire pairs state with purview
            # in order); states are positional tuples, not mappings.
            self._purview = tuple(
                sorted(chain.from_iterable(part.purview for part in self))
            )
        return self._purview

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.mechanism + self.purview)))

    def normalize(self) -> Self:
        """Return a copy with parts sorted into a canonical order."""
        return type(self)(*sorted(self), node_labels=self.node_labels)

    def num_connections_cut(self) -> int:
        """Number of connections severed by the induced edge cut (IIT 4.0 Eq. 24)."""
        n = 0
        purview_lengths = [len(part.purview) for part in self.parts]
        for i, part in enumerate(self.parts):
            n += len(part.mechanism) * (
                sum(purview_lengths[:i]) + sum(purview_lengths[i + 1 :])
            )
        return n

    def cut_matrix(self, n: int) -> NDArray[np.int_]:
        cm = np.zeros((n, n), dtype=int)
        for part in self.parts:
            outside_part = tuple(set(self.purview) - set(part.purview))
            cm[np.ix_(part.mechanism, outside_part)] = 1
        return cm

    def to_json(self) -> dict[str, Any]:
        return {"parts": list(self)}

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> JointPartition:
        return cls(*dct["parts"])


class JointBipartition(JointPartition):
    """A :class:`JointPartition` with exactly two parts."""

    __slots__ = JointPartition.__slots__

    def to_json(self) -> dict[str, Any]:
        return {"part0": self[0], "part1": self[1]}

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> JointBipartition:
        return cls(dct["part0"], dct["part1"])


class JointTripartition(JointPartition):
    """A :class:`JointPartition` with exactly three parts.

    Typically the "wedge" partition where the mechanism is strictly split
    across the first two parts.
    """

    __slots__ = JointPartition.__slots__
