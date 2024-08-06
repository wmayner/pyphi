# parallel/tree.py
"""Classes for specifying distributed computations."""

from dataclasses import dataclass
from typing import Optional

from ..conf import fallback
from ..utils import enforce_integer, enforce_integer_or_none


@dataclass(frozen=True)
class TreeSpec:
    depth: int
    size: int
    leaves: int
    leaf_size: int


class TreeConstraints:
    """Constraints on how a MapReduce tree should grow."""

    def __init__(
        self,
        total: Optional[int] = None,
        chunksize: Optional[int] = None,
        sequential_threshold: int = 1,
        max_depth: Optional[int] = None,
        max_size: Optional[int] = None,
        max_leaves: Optional[int] = None,
        branch_factor: int = 2,
    ) -> None:
        self.total = enforce_integer_or_none(total, name="total", min=0)
        self.chunksize = enforce_integer_or_none(chunksize, name="chunksize", min=1)
        self.current_chunksize = self.chunksize
        self.sequential_threshold = enforce_integer(
            sequential_threshold, name="sequential_threshold", min=1
        )
        self.max_depth = fallback(
            enforce_integer_or_none(max_depth, name="max_depth", min=1), float("inf")
        )
        self.max_size = fallback(
            enforce_integer_or_none(max_size, name="max_size", min=1), float("inf")
        )
        self.max_leaves = fallback(
            enforce_integer_or_none(max_leaves, name="max_leaves", min=1),
            float("inf"),
        )
        self.branch_factor = enforce_integer(branch_factor, name="branch_factor", min=2)

    def __repr__(self):
        attrs = ", ".join(
            f"{attr}={getattr(self, attr)}"
            for attr in [
                "total",
                "chunksize",
                "sequential_threshold",
                "max_depth",
                "max_size",
                "max_leaves",
                "branch_factor",
            ]
        )
        return f"{self.__class__.__name__}({attrs})"

    def validate(self, spec: TreeSpec) -> bool:
        return (
            spec.depth <= self.max_depth
            and spec.size <= self.max_size
            and spec.leaves <= self.max_leaves
            and self.sequential_threshold <= spec.leaf_size
        )

    def branch(self, spec: TreeSpec):
        raise NotImplementedError

    def simulate(self) -> TreeSpec:
        tree = TreeSpec(depth=1, size=1, leaves=1, leaf_size=self.total)
        while True:
            branched = self.branch(tree)
            if not self.validate(branched):
                return tree
            tree = branched

    def get_initial_chunksize(self) -> int:
        raise NotImplementedError


class TreeConstraintsChunksize(TreeConstraints):
    """Chunksize constraints on how a MapReduce tree should grow."""

    def branch(self, spec: TreeSpec) -> TreeSpec:
        depth = spec.depth + 1
        leaves = spec.leaves * (spec.leaf_size // max(self.current_chunksize, 1))
        size = spec.size + leaves
        leaf_size = self.current_chunksize
        self.current_chunksize = self.current_chunksize // self.branch_factor
        return TreeSpec(depth=depth, size=size, leaves=leaves, leaf_size=leaf_size)

    def simulate(self) -> TreeSpec:
        if self.total is None:
            # Chunksize with no total; size unknown
            return TreeSpec(
                depth=self.max_depth,
                size=self.max_size,
                leaves=self.max_leaves,
                leaf_size=None,
            )
        return super().simulate()

    def get_initial_chunksize(self):
        return self.chunksize


class TreeConstraintsSize(TreeConstraints):
    """Tree size constraints on how a MapReduce tree should grow."""

    def branch(self, spec: TreeSpec) -> TreeSpec:
        depth = spec.depth + 1
        leaves = spec.leaves * self.branch_factor
        size = spec.size + leaves
        leaf_size = self.total // leaves
        return TreeSpec(depth=depth, size=size, leaves=leaves, leaf_size=leaf_size)

    def get_initial_chunksize(self):
        return self.total // self.branch_factor


def get_constraints(
    total: Optional[int] = None,
    chunksize: Optional[int] = None,
    sequential_threshold: int = 1,
    max_depth: Optional[int] = None,
    max_size: Optional[int] = None,
    max_leaves: Optional[int] = None,
    branch_factor: int = 2,
) -> TreeConstraints:
    cls = TreeConstraintsSize
    if total is None:
        if chunksize is None:
            # No chunksize and no total; cannot determine tree size
            raise ValueError("if no total is given, chunksize must be provided")
        if not all(arg is None for arg in [max_size, max_leaves]):
            # Cannot enforce max_size or max_leaves with chunksize constraints if
            # total is not given
            raise ValueError(
                "if no total is given, only max_depth can be enforced; "
                f"got max_size={max_size}, max_leaves={max_leaves}"
            )
        cls = TreeConstraintsChunksize
    return cls(
        total=total,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        max_depth=max_depth,
        max_size=max_size,
        max_leaves=max_leaves,
        branch_factor=branch_factor,
    )
