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


class TestStateFrom:
    def test_identity_unit(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        assert unit.state_from(((0,),)) == 0
        assert unit.state_from(((1,),)) == 1

    def test_history_length_validated(self):
        unit = MacroUnit(constituents=(0,), update_grain=2, mapping=(0, 0, 0, 1))
        with pytest.raises(ValueError, match="history"):
            unit.state_from(((0,),))

    def test_entry_shape_validated(self):
        unit = MacroUnit(constituents=(0, 1), update_grain=1, mapping=(0, 0, 0, 1))
        with pytest.raises(ValueError, match="state"):
            unit.state_from(((0,),))

    def test_constituent_order_pins_truth_table_digits(self):
        # Mapping is over constituents in GIVEN order (3 first), while
        # state_from input columns follow sorted U^J = (1, 3).
        # mapping index = digits (state(3), state(1)) little-endian:
        # ON iff constituent 3 is ON and constituent 1 is OFF -> index 1.
        unit = MacroUnit(constituents=(3, 1), update_grain=1, mapping=(0, 1, 0, 0))
        # columns ordered by micro index: (state(1), state(3))
        assert unit.state_from(((0, 1),)) == 1
        assert unit.state_from(((1, 0),)) == 0
        assert unit.state_from(((1, 1),)) == 0

    def test_updates_oldest_first_newest_slowest(self):
        # tau = 2 over one constituent; digits = (oldest, newest).
        # mapping index 1 = (1, 0): ON in the OLD update only.
        unit = MacroUnit(constituents=(5,), update_grain=2, mapping=(0, 1, 0, 0))
        assert unit.state_from(((1,), (0,))) == 1
        assert unit.state_from(((0,), (1,))) == 0

    def test_meso_composition_hand_checked(self):
        # inner: over micro (0, 1), grain 1, ON iff both ON
        inner = MacroUnit(constituents=(0, 1), update_grain=1, mapping=(0, 0, 0, 1))
        # outer: over (inner,), grain 2, ON iff inner ON at the NEWER
        # of its two updates: digits (old, new) -> indices 2, 3 are ON
        outer = MacroUnit(constituents=(inner,), update_grain=2, mapping=(0, 0, 1, 1))
        # micro history: two updates of (u0, u1), oldest first
        assert outer.state_from(((0, 0), (1, 1))) == 1
        assert outer.state_from(((1, 1), (0, 1))) == 0
        assert outer.state_from(((1, 1), (1, 1))) == 1

    def test_micro_mapping_identity(self):
        unit = MacroUnit(constituents=(0,), update_grain=1, mapping=(0, 1))
        assert unit.micro_mapping == (0, 1)

    def test_micro_mapping_matches_state_from(self):
        inner = MacroUnit(constituents=(2, 0), update_grain=1, mapping=(0, 1, 1, 0))
        outer = MacroUnit(constituents=(inner,), update_grain=2, mapping=(0, 1, 1, 0))
        n = len(outer.micro_constituents)
        tau = outer.micro_grain
        for index in range(2 ** (n * tau)):
            digits = _mixed_radix_digits(index, (2,) * (n * tau))
            history = tuple(digits[t * n : (t + 1) * n] for t in range(tau))
            assert outer.micro_mapping[index] == outer.state_from(history)
