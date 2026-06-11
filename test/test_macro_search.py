"""Tests for pyphi.macro.search: bounded intrinsic-unit search (Eqs 15-19)."""

import pytest

from pyphi.macro.search import SearchBounds
from pyphi.macro.search import candidate_mappings


class TestSearchBounds:
    def test_defaults(self):
        bounds = SearchBounds()
        assert bounds.max_constituents == 4
        assert bounds.max_update_grain == 1
        assert bounds.max_depth == 1
        assert bounds.mappings == "FAMILIES"
        assert bounds.exhaustive_cap == 8
        assert bounds.apportionment == "NONE"
        assert bounds.max_background == 0

    def test_frozen(self):
        bounds = SearchBounds()
        with pytest.raises(AttributeError):
            bounds.max_depth = 2

    def test_max_micro_grain_composes(self):
        assert SearchBounds().max_micro_grain == 1
        assert SearchBounds(max_update_grain=2, max_depth=2).max_micro_grain == 4

    def test_max_constituents_below_one_rejected(self):
        with pytest.raises(ValueError, match="max_constituents"):
            SearchBounds(max_constituents=0)

    def test_max_update_grain_below_one_rejected(self):
        with pytest.raises(ValueError, match="max_update_grain"):
            SearchBounds(max_update_grain=0)

    def test_negative_max_depth_rejected(self):
        with pytest.raises(ValueError, match="max_depth"):
            SearchBounds(max_depth=-1)

    def test_unknown_mappings_policy_rejected(self):
        with pytest.raises(ValueError, match="mappings"):
            SearchBounds(mappings="ALL")

    def test_unknown_apportionment_policy_rejected(self):
        with pytest.raises(ValueError, match="apportionment"):
            SearchBounds(apportionment="ALWAYS")

    def test_enumerate_requires_max_background(self):
        with pytest.raises(ValueError, match="max_background"):
            SearchBounds(apportionment="ENUMERATE")
        assert (
            SearchBounds(apportionment="ENUMERATE", max_background=1).max_background
            == 1
        )


class TestCandidateMappings:
    def test_families_two_constituents_grain_one(self):
        tables = candidate_mappings(2, 1, SearchBounds())
        # Coarse-grainings (canonicalized: complement when the all-OFF
        # state maps to ON), then black-boxings, first-seen order:
        # on_counts {0} -> complement of (1,0,0,0) = at-least-one-ON;
        # {1} -> exactly-one-ON; {2} -> both-ON; {0,1}, {0,2}, {1,2} ->
        # duplicates of the first three; blackbox {0} -> constituent-0;
        # {1} -> constituent-1; {0,1} -> duplicate of both-ON.
        assert tables == (
            (0, 1, 1, 1),
            (0, 1, 1, 0),
            (0, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        )

    def test_families_count_three_constituents(self):
        assert len(candidate_mappings(3, 1, SearchBounds())) == 13

    def test_families_higher_grain_blackbox_only(self):
        # Coarse-graining is defined at update grain 1 only.
        tables = candidate_mappings(1, 2, SearchBounds(max_update_grain=2))
        assert tables == ((0, 0, 1, 1),)

    def test_exhaustive_min_shape(self):
        tables = candidate_mappings(2, 1, SearchBounds(mappings="EXHAUSTIVE"))
        # 2**(4-1) - 1 = 7 canonical surjective tables.
        assert len(tables) == 7
        assert len(set(tables)) == 7
        for table in tables:
            assert table[0] == 0  # canonical: all-OFF maps to OFF
            assert 1 in table  # surjective
        assert (0, 0, 0, 1) in tables

    def test_exhaustive_cap_exceeded(self):
        with pytest.raises(ValueError, match="exhaustive_cap"):
            candidate_mappings(
                2,
                2,
                SearchBounds(mappings="EXHAUSTIVE", max_update_grain=2),
            )

    def test_all_tables_canonical_and_unique(self):
        for policy in ("FAMILIES", "EXHAUSTIVE"):
            tables = candidate_mappings(2, 1, SearchBounds(mappings=policy))
            assert len(set(tables)) == len(tables)
            assert all(t[0] == 0 for t in tables)
