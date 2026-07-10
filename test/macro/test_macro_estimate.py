"""Tests for pyphi.macro.estimate: the grain-search cost pre-flight."""

import pytest

from pyphi import config
from pyphi.conf import presets
from pyphi.macro.estimate import SearchEstimate
from pyphi.macro.estimate import estimate_search
from pyphi.macro.search import SearchBounds


class TestCountingWalk:
    def test_min_defaults_worst_case(self):
        # n=2 at default bounds: one candidate decomposition ({0},{1});
        # judgment evaluates it + its two singleton competitors; its 5
        # FAMILIES-mapped variants join the pool; the sweep adds them.
        est = estimate_search(SearchBounds(), 2)
        assert est.judgments_by_level == (1,)
        assert est.worst_case_pool_by_level == (7,)  # 2 micro + 5 variants
        assert est.assemblies_upper_bound == 8
        assert est.distinct_systems_upper_bound == 8
        assert est.systems_by_unit_count == {1: 7, 2: 1}
        assert est.construction_keys_upper_bound == 3
        assert est.is_exact is False
        assert est.truncated is False

    def test_min_exhaustive_worst_case(self):
        est = estimate_search(SearchBounds(mappings="EXHAUSTIVE"), 2)
        assert est.distinct_systems_upper_bound == 10  # 3 + 7 mappings

    def test_depth_zero_is_exact(self):
        est = estimate_search(SearchBounds(max_depth=0), 2)
        assert est.is_exact is True
        assert est.judgments_by_level == ()
        assert est.distinct_systems_upper_bound == 3  # {0}, {1}, {0,1}
        assert est.assemblies_upper_bound == 3
        assert est.systems_by_unit_count == {1: 2, 2: 1}

    def test_depth_zero_counts_subsets(self):
        est = estimate_search(SearchBounds(max_depth=0), 3)
        assert est.distinct_systems_upper_bound == 7  # nonempty subsets
        assert est.systems_by_unit_count == {1: 3, 2: 3, 3: 1}

    def test_truncation(self):
        est = estimate_search(SearchBounds(), 4, limit=5)
        assert est.truncated is True
        assert est.is_exact is False
        assert 0 < est.distinct_systems_upper_bound <= 5

    def test_exhaustive_above_cap_raises_without_running(self):
        bounds = SearchBounds(
            mappings="EXHAUSTIVE", max_constituents=4, exhaustive_cap=8
        )
        with pytest.raises(ValueError, match="exhaustive_cap"):
            estimate_search(bounds, 4)

    def test_monotone_in_depth(self):
        shallow = estimate_search(SearchBounds(max_depth=0), 3)
        deep = estimate_search(SearchBounds(max_depth=1), 3)
        assert deep.distinct_systems_upper_bound >= shallow.distinct_systems_upper_bound


class TestAgainstRealSweeps:
    def test_min_defaults_estimate_equals_records(self):
        from pyphi.macro.search import complexes
        from test.macro.test_macro_criteria import min_substrate

        substrate = min_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0), SearchBounds())
        est = SearchBounds().estimate(substrate)
        # Worst case achieved: min's only candidate decomposition passes.
        assert est.distinct_systems_upper_bound == len(result.records) == 8

    def test_min_exhaustive_estimate_equals_records(self):
        from pyphi.macro.search import complexes
        from test.macro.test_macro_criteria import min_substrate

        substrate = min_substrate()
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0), bounds)
        est = bounds.estimate(substrate)
        assert est.distinct_systems_upper_bound == len(result.records) == 10

    def test_bu_depth_zero_unreachable_gap(self):
        from pyphi.macro.search import complexes
        from test.macro.test_macro_criteria import bu_substrate

        substrate = bu_substrate()
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0), bounds)
        est = bounds.estimate(substrate)
        # One candidate's state is unreachable under its own TPM and is
        # discarded at run time; the enumeration cannot predict that.
        assert est.distinct_systems_upper_bound == 7
        assert len(result.records) == 6

    def test_bu_defaults_upper_bound(self):
        from pyphi.macro.search import complexes
        from test.macro.test_macro_criteria import bu_substrate

        substrate = bu_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0), SearchBounds())
        est = SearchBounds().estimate(substrate)
        assert est.distinct_systems_upper_bound >= len(result.records)

    def test_min_depth_zero_exact_against_records(self):
        from pyphi.macro.search import complexes
        from test.macro.test_macro_criteria import min_substrate

        substrate = min_substrate()
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0), bounds)
        est = bounds.estimate(substrate)
        assert est.is_exact is True
        assert est.distinct_systems_upper_bound == len(result.records) == 3

    def test_chain_depth_zero_upper_bound(self):
        from pyphi.macro.search import complexes
        from test.macro.test_macro_search import decaying_chain_substrate

        substrate = decaying_chain_substrate()
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0, 0), bounds)
        est = bounds.estimate(substrate)
        # 15 nonempty subsets enumerated; unreachable-state candidates
        # (if any) are discarded at run time, so records may be fewer.
        assert est.distinct_systems_upper_bound == 15
        assert est.distinct_systems_upper_bound >= len(result.records)


class TestPartitionWeights:
    def test_partition_counts_pinned(self):
        # Measured values under DIRECTED_SET_PARTITION.
        with config.override(**presets.iit4_2023):
            est = estimate_search(SearchBounds(max_depth=0), 5)
        assert est.partitions_by_unit_count == {
            1: 1,
            2: 3,
            3: 22,
            4: 150,
            5: 1061,
        }

    def test_partition_sweeps_are_weighted_sum(self):
        with config.override(**presets.iit4_2023):
            est = estimate_search(SearchBounds(max_depth=0), 3)
        expected = sum(
            est.systems_by_unit_count[m] * est.partitions_by_unit_count[m]
            for m in est.systems_by_unit_count
        )
        assert est.partition_sweeps_upper_bound == expected
        # n=3 subsets: 3 singletons + 3 pairs + 1 triple.
        assert expected == 3 * 1 + 3 * 3 + 1 * 22


class TestSurfaces:
    def test_display_card_headline_rows(self):
        est = estimate_search(SearchBounds(), 2)
        desc = est._describe(verbosity=2)
        labels = [row.label for section in desc.sections for row in section.rows]
        assert "Candidate systems" in labels
        assert "Partition sweeps" in labels
        assert "m = 1" in labels  # bucket section

    def test_qualifier_tracks_exactness(self):
        exact = estimate_search(SearchBounds(max_depth=0), 2)
        bound = estimate_search(SearchBounds(), 2)
        assert "= 3" in exact._describe(2).compact
        assert "≤ 8" in bound._describe(2).compact

    def test_pandas_record_scalars(self):
        est = estimate_search(SearchBounds(), 2)
        record = est._pandas_record()
        assert record["distinct_systems_upper_bound"] == 8
        assert record["is_exact"] is False

    def test_package_exports(self):
        import pyphi.macro

        assert pyphi.macro.SearchEstimate is SearchEstimate
        assert pyphi.macro.estimate_search is estimate_search
