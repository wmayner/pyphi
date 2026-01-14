"""Tests for the adaptive chunking module."""

from pyphi.parallel.chunking import adaptive_chunk
from pyphi.parallel.chunking import calculate_target_work
from pyphi.parallel.chunking import estimate_total_work
from pyphi.parallel.chunking import estimate_work_size


class TestEstimateWorkSize:
    """Tests for work size estimation."""

    def test_no_context_returns_uniform(self):
        """Without context, all elements have weight 1."""
        assert estimate_work_size("anything", context=None) == 1.0
        assert estimate_work_size([1, 2, 3], context=None) == 1.0

    def test_mechanism_context(self):
        """Mechanism context scales with size * 2^size."""
        # Size 1: 1 * 2^1 = 2
        assert estimate_work_size((1,), context="mechanism") == 2
        # Size 2: 2 * 2^2 = 8
        assert estimate_work_size((1, 2), context="mechanism") == 8
        # Size 3: 3 * 2^3 = 24
        assert estimate_work_size((1, 2, 3), context="mechanism") == 24

    def test_purview_context(self):
        """Purview context scales linearly with size."""
        assert estimate_work_size((1,), context="purview") == 1
        assert estimate_work_size((1, 2), context="purview") == 2
        assert estimate_work_size((1, 2, 3), context="purview") == 3

    def test_partition_context(self):
        """Partition context scales quadratically."""
        assert estimate_work_size((1,), context="partition") == 1
        assert estimate_work_size((1, 2), context="partition") == 4
        assert estimate_work_size((1, 2, 3), context="partition") == 9

    def test_cut_context(self):
        """Cut context scales exponentially."""
        assert estimate_work_size((1,), context="cut") == 2
        assert estimate_work_size((1, 2), context="cut") == 4
        assert estimate_work_size((1, 2, 3), context="cut") == 8

    def test_unknown_context_returns_uniform(self):
        """Unknown context returns uniform weight."""
        assert estimate_work_size((1, 2, 3), context="unknown") == 1.0

    def test_handles_non_iterable(self):
        """Handles elements without len() gracefully."""
        # Should return 1.0 when len() fails
        assert estimate_work_size(42, context="mechanism") == 1.0


class TestAdaptiveChunk:
    """Tests for adaptive chunking."""

    def test_empty_iterable(self):
        """Empty iterable yields nothing."""
        result = list(adaptive_chunk([], target_work_per_chunk=10))
        assert result == []

    def test_single_element_below_target(self):
        """Single element below target goes in one chunk."""
        result = list(
            adaptive_chunk([(1,)], target_work_per_chunk=10, context="mechanism")
        )
        assert result == [[(1,)]]

    def test_uniform_chunking_without_context(self):
        """Without context, chunks by count (each element weight 1)."""
        items = list(range(10))
        result = list(adaptive_chunk(items, target_work_per_chunk=3))
        # Each item has weight 1, so ~3 items per chunk
        assert len(result) >= 3  # At least 3 chunks for 10 items / 3 per chunk

    def test_balances_by_work(self):
        """Adaptive chunking balances work, not count."""
        # Mix of small and large mechanisms
        items = [(1,), (1, 2, 3), (1,), (1, 2)]  # weights: 2, 24, 2, 8 = 36 total

        # With target of 18, should split into 2 chunks
        result = list(
            adaptive_chunk(items, target_work_per_chunk=18, context="mechanism")
        )

        # Verify chunks are balanced by work
        chunk_works = []
        for chunk in result:
            work = sum(estimate_work_size(item, "mechanism") for item in chunk)
            chunk_works.append(work)

        # All chunks should be roughly balanced
        for work in chunk_works:
            assert work <= 26  # Each chunk should have at most ~18 + one big item

    def test_max_chunks_limit(self):
        """max_chunks limits number of chunks."""
        items = list(range(100))
        result = list(adaptive_chunk(items, target_work_per_chunk=5, max_chunks=3))
        assert len(result) <= 3

    def test_custom_size_func(self):
        """Custom size function overrides context."""
        items = [1, 2, 3, 4, 5]

        # Custom: weight = value^2
        def custom_func(x):
            return x**2

        result = list(
            adaptive_chunk(items, target_work_per_chunk=10, size_func=custom_func)
        )
        # Weights: 1, 4, 9, 16, 25 = 55 total
        # With target 10: expect chunks like [1,4], [9], [16], [25]
        assert len(result) >= 4


class TestEstimateTotalWork:
    """Tests for total work estimation."""

    def test_empty_iterable(self):
        """Empty iterable has zero work."""
        total, items = estimate_total_work([])
        assert total == 0
        assert items == []

    def test_materializes_iterable(self):
        """Materializes generators."""
        _total, items = estimate_total_work(x for x in [1, 2, 3])
        assert items == [1, 2, 3]

    def test_sums_work(self):
        """Correctly sums work with context."""
        items = [(1,), (1, 2)]  # weights: 2, 8
        total, _ = estimate_total_work(items, context="mechanism")
        assert total == 10


class TestCalculateTargetWork:
    """Tests for target work calculation."""

    def test_basic_calculation(self):
        """Basic target calculation."""
        # 100 total work, 4 workers -> target ~100/(4*2) = 12.5
        target = calculate_target_work(100, num_workers=4)
        assert 10 <= target <= 30

    def test_respects_min_chunks(self):
        """Target respects minimum chunks when feasible.

        Note: min_chunks is balanced against max_chunks_per_worker.
        With num_workers=4 and max_chunks_per_worker=4, we can have
        up to 16 chunks total.
        """
        # With min_chunks=10, max target = total/10 = 10
        # With 4 workers and max_chunks_per_worker=4, min_target = 100/16 = 6.25
        # Expected: max(6.25, min(10, 12.5)) = max(6.25, 10) = 10
        target = calculate_target_work(100, num_workers=4, min_chunks=10)
        assert target <= 10

    def test_limits_chunks_per_worker(self):
        """Limits excessive chunks per worker."""
        # With max_chunks_per_worker=2, min target = total/(workers*2)
        target = calculate_target_work(100, num_workers=4, max_chunks_per_worker=2)
        assert target >= 100 / (4 * 2)
