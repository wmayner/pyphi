"""Tests for the parallel tree constraint module."""

from dataclasses import FrozenInstanceError

import pytest

from pyphi.parallel.tree import TreeConstraintsChunksize
from pyphi.parallel.tree import TreeConstraintsSize
from pyphi.parallel.tree import TreeSpec
from pyphi.parallel.tree import get_constraints


class TestTreeSpec:
    """Tests for TreeSpec dataclass."""

    def test_creation_and_attributes(self):
        """TreeSpec correctly stores all attributes."""
        spec = TreeSpec(depth=3, size=7, leaves=4, leaf_size=25)
        assert spec.depth == 3
        assert spec.size == 7
        assert spec.leaves == 4
        assert spec.leaf_size == 25

    def test_frozen_immutability(self):
        """TreeSpec is frozen and cannot be modified."""
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        with pytest.raises(FrozenInstanceError):
            spec.depth = 2  # type: ignore[misc]

    def test_equality(self):
        """Two TreeSpecs with same values are equal."""
        spec1 = TreeSpec(depth=2, size=3, leaves=2, leaf_size=50)
        spec2 = TreeSpec(depth=2, size=3, leaves=2, leaf_size=50)
        assert spec1 == spec2

        spec3 = TreeSpec(depth=2, size=3, leaves=2, leaf_size=51)
        assert spec1 != spec3


class TestTreeConstraints:
    """Tests for TreeConstraints base class."""

    def test_default_construction(self):
        """TreeConstraintsSize can be constructed with minimal arguments."""
        # Use concrete subclass since base class has abstract methods
        constraints = TreeConstraintsSize(total=100)
        assert constraints.total == 100
        assert constraints.chunksize is None
        assert constraints.sequential_threshold == 1
        assert constraints.branch_factor == 2

    def test_total_must_be_non_negative(self):
        """total=-1 raises ValueError."""
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=-1)

    def test_total_zero_allowed(self):
        """total=0 is allowed."""
        constraints = TreeConstraintsSize(total=0)
        assert constraints.total == 0

    def test_chunksize_must_be_positive(self):
        """chunksize=0 or negative raises ValueError."""
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=100, chunksize=0)
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=100, chunksize=-1)

    def test_sequential_threshold_must_be_positive(self):
        """sequential_threshold must be >= 1."""
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=100, sequential_threshold=0)
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=100, sequential_threshold=-1)

    def test_branch_factor_must_be_at_least_2(self):
        """branch_factor < 2 raises ValueError."""
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=100, branch_factor=1)
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=100, branch_factor=0)

    def test_max_depth_none_becomes_infinity(self):
        """max_depth=None converts to float('inf')."""
        constraints = TreeConstraintsSize(total=100, max_depth=None)
        assert constraints.max_depth == float("inf")

    def test_max_size_none_becomes_infinity(self):
        """max_size=None converts to float('inf')."""
        constraints = TreeConstraintsSize(total=100, max_size=None)
        assert constraints.max_size == float("inf")

    def test_max_leaves_none_becomes_infinity(self):
        """max_leaves=None converts to float('inf')."""
        constraints = TreeConstraintsSize(total=100, max_leaves=None)
        assert constraints.max_leaves == float("inf")

    def test_max_depth_validates(self):
        """max_depth must be >= 1 if provided."""
        with pytest.raises(ValueError):
            TreeConstraintsSize(total=100, max_depth=0)

    def test_validate_respects_max_depth(self):
        """validate() returns False when spec.depth > max_depth."""
        constraints = TreeConstraintsSize(total=100, max_depth=2)
        spec_ok = TreeSpec(depth=2, size=3, leaves=2, leaf_size=50)
        spec_bad = TreeSpec(depth=3, size=7, leaves=4, leaf_size=25)
        assert constraints.validate(spec_ok) is True
        assert constraints.validate(spec_bad) is False

    def test_validate_respects_max_size(self):
        """validate() returns False when spec.size > max_size."""
        constraints = TreeConstraintsSize(total=100, max_size=5)
        spec_ok = TreeSpec(depth=2, size=5, leaves=4, leaf_size=25)
        spec_bad = TreeSpec(depth=2, size=6, leaves=4, leaf_size=25)
        assert constraints.validate(spec_ok) is True
        assert constraints.validate(spec_bad) is False

    def test_validate_respects_max_leaves(self):
        """validate() returns False when spec.leaves > max_leaves."""
        constraints = TreeConstraintsSize(total=100, max_leaves=4)
        spec_ok = TreeSpec(depth=2, size=5, leaves=4, leaf_size=25)
        spec_bad = TreeSpec(depth=3, size=7, leaves=8, leaf_size=12)
        assert constraints.validate(spec_ok) is True
        assert constraints.validate(spec_bad) is False

    def test_validate_respects_sequential_threshold(self):
        """validate() returns False when leaf_size < sequential_threshold."""
        constraints = TreeConstraintsSize(total=100, sequential_threshold=30)
        spec_ok = TreeSpec(depth=2, size=3, leaves=2, leaf_size=50)
        spec_bad = TreeSpec(depth=3, size=7, leaves=4, leaf_size=25)
        assert constraints.validate(spec_ok) is True
        assert constraints.validate(spec_bad) is False

    def test_repr(self):
        """__repr__ includes all constraint parameters."""
        constraints = TreeConstraintsSize(
            total=100,
            chunksize=10,
            sequential_threshold=5,
            max_depth=3,
            max_size=10,
            max_leaves=8,
            branch_factor=2,
        )
        repr_str = repr(constraints)
        assert "TreeConstraintsSize" in repr_str
        assert "total=100" in repr_str
        assert "chunksize=10" in repr_str
        assert "sequential_threshold=5" in repr_str
        assert "max_depth=3" in repr_str
        assert "max_size=10" in repr_str
        assert "max_leaves=8" in repr_str
        assert "branch_factor=2" in repr_str


class TestTreeConstraintsSize:
    """Tests for TreeConstraintsSize (branches based on total elements)."""

    def test_branch_increments_depth(self):
        """branch() increments depth by 1."""
        constraints = TreeConstraintsSize(total=100)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        branched = constraints.branch(spec)
        assert branched.depth == 2

    def test_branch_multiplies_leaves(self):
        """Leaves grow by branch_factor on each branch."""
        constraints = TreeConstraintsSize(total=100, branch_factor=2)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        branched = constraints.branch(spec)
        assert branched.leaves == 2

        constraints3 = TreeConstraintsSize(total=100, branch_factor=3)
        branched3 = constraints3.branch(spec)
        assert branched3.leaves == 3

    def test_branch_calculates_leaf_size(self):
        """leaf_size = total // leaves."""
        constraints = TreeConstraintsSize(total=100, branch_factor=2)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        branched = constraints.branch(spec)
        assert branched.leaf_size == 50  # 100 // 2

        branched2 = constraints.branch(branched)
        assert branched2.leaf_size == 25  # 100 // 4

    def test_branch_updates_size(self):
        """Size = old_size + new_leaves."""
        constraints = TreeConstraintsSize(total=100, branch_factor=2)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        branched = constraints.branch(spec)
        assert branched.size == 3  # 1 + 2

        branched2 = constraints.branch(branched)
        assert branched2.size == 7  # 3 + 4

    def test_branch_with_total_none(self):
        """When total is None, leaf_size becomes 0."""
        constraints = TreeConstraintsSize(total=None)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=0)
        branched = constraints.branch(spec)
        assert branched.leaf_size == 0

    def test_simulate_stops_at_constraints(self):
        """simulate() branches until validate() returns False."""
        # With max_depth=2, should stop at depth 2
        constraints = TreeConstraintsSize(total=100, max_depth=2)
        tree = constraints.simulate()
        assert tree.depth == 2
        assert tree.leaves == 2
        assert tree.leaf_size == 50

    def test_simulate_respects_sequential_threshold(self):
        """Tree stops when leaf_size would drop below threshold."""
        constraints = TreeConstraintsSize(total=100, sequential_threshold=30)
        tree = constraints.simulate()
        # After 1 branch: leaf_size=50 (ok)
        # After 2 branches: leaf_size=25 (< 30, invalid)
        assert tree.depth == 2
        assert tree.leaf_size == 50

    def test_simulate_with_max_leaves(self):
        """Tree stops when leaves would exceed max_leaves."""
        constraints = TreeConstraintsSize(total=100, max_leaves=2)
        tree = constraints.simulate()
        assert tree.leaves == 2

    def test_get_initial_chunksize(self):
        """Returns total // branch_factor."""
        constraints = TreeConstraintsSize(total=100, branch_factor=2)
        assert constraints.get_initial_chunksize() == 50

        constraints4 = TreeConstraintsSize(total=100, branch_factor=4)
        assert constraints4.get_initial_chunksize() == 25

    def test_get_initial_chunksize_without_total(self):
        """Returns 0 when total is None."""
        constraints = TreeConstraintsSize(total=None)
        assert constraints.get_initial_chunksize() == 0


class TestTreeConstraintsChunksize:
    """Tests for TreeConstraintsChunksize (branches based on chunksize)."""

    def test_branch_halves_current_chunksize(self):
        """current_chunksize divided by branch_factor each branch."""
        constraints = TreeConstraintsChunksize(chunksize=100, branch_factor=2)
        assert constraints.current_chunksize == 100

        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        constraints.branch(spec)
        assert constraints.current_chunksize == 50

        constraints.branch(spec)
        assert constraints.current_chunksize == 25

    def test_branch_calculates_leaves_from_chunksize(self):
        """leaves = old_leaves * (leaf_size // current_chunksize)."""
        constraints = TreeConstraintsChunksize(chunksize=50, branch_factor=2)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        branched = constraints.branch(spec)
        # leaves = 1 * (100 // 50) = 2
        assert branched.leaves == 2

    def test_branch_sets_leaf_size_to_current_chunksize(self):
        """leaf_size becomes current_chunksize after branch."""
        constraints = TreeConstraintsChunksize(chunksize=50, branch_factor=2)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)
        branched = constraints.branch(spec)
        assert branched.leaf_size == 50

    def test_branch_handles_zero_chunksize_safely(self):
        """Division by zero protected with max(current_chunksize, 1)."""
        constraints = TreeConstraintsChunksize(chunksize=1, branch_factor=2)
        spec = TreeSpec(depth=1, size=1, leaves=1, leaf_size=100)

        # After several branches, current_chunksize could become 0
        # due to integer division, but max(..., 1) prevents division by zero
        constraints.branch(spec)  # chunksize: 1 -> 0
        branched = constraints.branch(spec)  # Should not raise
        assert branched is not None

    def test_simulate_without_total_returns_max_constraints(self):
        """When total is None, returns TreeSpec with max constraints."""
        constraints = TreeConstraintsChunksize(
            chunksize=50, max_depth=5, max_size=100, max_leaves=32
        )
        tree = constraints.simulate()
        assert tree.depth == 5
        assert tree.size == 100
        assert tree.leaves == 32
        assert tree.leaf_size == 0

    def test_simulate_with_total_uses_parent(self):
        """When total is provided, delegates to parent simulate()."""
        constraints = TreeConstraintsChunksize(total=100, chunksize=50, max_depth=3)
        tree = constraints.simulate()
        # Should behave like normal simulation, not return max values
        assert tree.depth <= 3
        assert tree.leaf_size > 0 or tree.depth == 1

    def test_get_initial_chunksize(self):
        """Returns the configured chunksize."""
        constraints = TreeConstraintsChunksize(chunksize=50)
        assert constraints.get_initial_chunksize() == 50

    def test_get_initial_chunksize_returns_zero_if_none(self):
        """Returns 0 if chunksize is None."""
        constraints = TreeConstraintsChunksize(total=100)
        assert constraints.get_initial_chunksize() == 0


class TestGetConstraints:
    """Tests for get_constraints() factory function."""

    def test_returns_size_constraints_with_total(self):
        """With total, returns TreeConstraintsSize."""
        constraints = get_constraints(total=100)
        assert isinstance(constraints, TreeConstraintsSize)

    def test_returns_chunksize_constraints_without_total(self):
        """Without total but with chunksize, returns TreeConstraintsChunksize."""
        constraints = get_constraints(chunksize=50)
        assert isinstance(constraints, TreeConstraintsChunksize)

    def test_error_no_total_no_chunksize(self):
        """Raises ValueError when neither total nor chunksize given."""
        with pytest.raises(ValueError, match="chunksize must be provided"):
            get_constraints()

    def test_error_no_total_with_max_size(self):
        """Raises ValueError when no total but max_size specified."""
        with pytest.raises(ValueError, match="only max_depth can be enforced"):
            get_constraints(chunksize=50, max_size=100)

    def test_error_no_total_with_max_leaves(self):
        """Raises ValueError when no total but max_leaves specified."""
        with pytest.raises(ValueError, match="only max_depth can be enforced"):
            get_constraints(chunksize=50, max_leaves=8)

    def test_allows_max_depth_without_total(self):
        """max_depth allowed without total (only depth can be enforced)."""
        constraints = get_constraints(chunksize=50, max_depth=5)
        assert isinstance(constraints, TreeConstraintsChunksize)
        assert constraints.max_depth == 5

    def test_passes_all_parameters(self):
        """All parameters correctly passed to constructed class."""
        constraints = get_constraints(
            total=100,
            chunksize=25,
            sequential_threshold=10,
            max_depth=4,
            max_size=50,
            max_leaves=16,
            branch_factor=3,
        )
        assert constraints.total == 100
        assert constraints.chunksize == 25
        assert constraints.sequential_threshold == 10
        assert constraints.max_depth == 4
        assert constraints.max_size == 50
        assert constraints.max_leaves == 16
        assert constraints.branch_factor == 3
