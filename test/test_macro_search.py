"""Tests for pyphi.macro.search: bounded intrinsic-unit search (Eqs 15-19)."""

import pytest

from pyphi.macro.search import SearchBounds


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
