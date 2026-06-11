"""Tests for pyphi.macro.units: MacroUnit value objects and mappings."""

import pytest

from pyphi.macro.units import MacroUnit
from pyphi.macro.units import _mixed_radix_digits
from pyphi.macro.units import _mixed_radix_index


class TestMixedRadix:
    def test_roundtrip_binary(self):
        radices = (2, 2, 2)
        for i in range(8):
            digits = _mixed_radix_digits(i, radices)
            assert _mixed_radix_index(digits, radices) == i

    def test_first_digit_varies_fastest(self):
        # little-endian: index 1 flips the FIRST digit
        assert _mixed_radix_digits(1, (2, 3, 2)) == (1, 0, 0)
        assert _mixed_radix_digits(2, (2, 3, 2)) == (0, 1, 0)

    def test_heterogeneous_radices(self):
        radices = (2, 3, 4)
        seen = set()
        for i in range(24):
            digits = _mixed_radix_digits(i, radices)
            assert _mixed_radix_index(digits, radices) == i
            seen.add(digits)
        assert len(seen) == 24

    def test_index_rejects_out_of_range_digit(self):
        with pytest.raises(ValueError):
            _mixed_radix_index((2, 0), (2, 2))


class TestMacroUnitConstruction:
    def test_minimal_identity_unit(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        assert unit.micro_constituents == (0,)
        assert unit.micro_grain == 1
        assert unit.alphabet_size == 2

    def test_micro_constituents_sorted_union(self):
        unit = MacroUnit(constituents=(3, 1), update_grain=1, mapping=(0, 0, 0, 1))
        assert unit.micro_constituents == (1, 3)

    def test_micro_grain_is_product_down_hierarchy(self):
        inner = MacroUnit(constituents=(0, 1), update_grain=2, mapping=(0,) * 15 + (1,))
        outer = MacroUnit(
            constituents=(inner,), update_grain=3, mapping=(0, 1, 1, 0, 0, 1, 1, 0)
        )
        assert inner.micro_grain == 2
        assert outer.micro_grain == 6
        assert outer.micro_constituents == (0, 1)

    def test_empty_constituents_rejected(self):
        with pytest.raises(ValueError, match="constituent"):
            MacroUnit(constituents=(), update_grain=1, mapping=(0, 1))

    def test_update_grain_below_one_rejected(self):
        with pytest.raises(ValueError, match="grain"):
            MacroUnit(constituents=(0,), update_grain=0, mapping=(0, 1))

    def test_wrong_mapping_length_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            MacroUnit(constituents=(0, 1), update_grain=1, mapping=(0, 1))

    def test_nonbinary_mapping_entry_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 2))

    def test_nonsurjective_mapping_rejected(self):
        with pytest.raises(ValueError, match="both"):
            MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 0))

    def test_overlapping_constituents_rejected(self):
        inner = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        with pytest.raises(ValueError, match="overlap"):
            MacroUnit(constituents=(inner, 0), update_grain=1, mapping=(0, 0, 0, 1))

    def test_mismatched_constituent_grains_rejected(self):
        deep = MacroUnit(constituents=(0,), update_grain=2, mapping=(0, 0, 0, 1))
        with pytest.raises(ValueError, match="grain"):
            MacroUnit(constituents=(deep, 2), update_grain=1, mapping=(0, 0, 0, 1))

    def test_apportionment_overlapping_constituents_rejected(self):
        with pytest.raises(ValueError, match="apportionment"):
            MacroUnit(
                constituents=(0, 1),
                update_grain=1,
                mapping=(0, 0, 0, 1),
                background_apportionment=(1,),
            )

    def test_frozen(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        with pytest.raises(AttributeError):
            unit.update_grain = 2
