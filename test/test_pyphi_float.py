"""Tests for ``pyphi.data_structures.PyPhiFloat``.

Pins two specific behaviors that prior versions got wrong:

1. ``__eq__`` / ``__ne__`` return ``NotImplemented`` (not ``False`` / ``True``)
   for non-numeric arguments, so Python's reflected-comparison protocol
   kicks in and lets the other operand decide equality.

2. ``__hash__`` snapshots ``config.PRECISION`` at construction. A
   ``PyPhiFloat`` placed in a set or dict keeps a stable hash even if
   ``config.PRECISION`` is later changed; without this, set/dict
   membership lookups silently return wrong answers after a precision
   change.
"""

from __future__ import annotations

from pyphi.conf import config
from pyphi.data_structures import PyPhiFloat


class TestPyPhiFloatComparison:
    """Comparison protocol behavior."""

    def test_eq_with_int_returns_bool(self):
        assert (PyPhiFloat(1.0) == 1) is True
        assert (PyPhiFloat(1.0) == 2) is False

    def test_eq_with_float_returns_bool(self):
        assert (PyPhiFloat(1.0) == 1.0) is True
        assert (PyPhiFloat(1.0) == 2.0) is False

    def test_eq_with_pyphifloat_returns_bool(self):
        assert (PyPhiFloat(1.0) == PyPhiFloat(1.0)) is True
        assert (PyPhiFloat(1.0) == PyPhiFloat(2.0)) is False

    def test_eq_with_non_numeric_returns_notimplemented(self):
        # Python's standard "==" never raises; it falls back to identity
        # comparison if both sides return NotImplemented. The test is that
        # the comparison doesn't claim equality with a string or list.
        assert (PyPhiFloat(1.0) == "1.0") is False
        assert (PyPhiFloat(1.0) == [1.0]) is False
        assert (PyPhiFloat(1.0) == None) is False  # noqa: E711
        assert (PyPhiFloat(1.0) == object()) is False

    def test_eq_returns_notimplemented_directly(self):
        """The dunder method itself returns NotImplemented for foreign types
        (verified by direct call), so reflected ``__eq__`` on the other
        operand gets a chance to handle the comparison."""
        result = PyPhiFloat(1.0).__eq__("1.0")
        assert result is NotImplemented

    def test_ne_returns_notimplemented_directly(self):
        result = PyPhiFloat(1.0).__ne__("1.0")
        assert result is NotImplemented

    def test_precision_aware_equality(self):
        """Values within ``config.PRECISION`` compare equal."""
        # Default PRECISION is 13; values differing at 1e-14 should be equal
        a = PyPhiFloat(1.0)
        b = PyPhiFloat(1.0 + 1e-14)
        assert a == b

    def test_precision_aware_inequality(self):
        """Values outside ``config.PRECISION`` compare unequal."""
        a = PyPhiFloat(1.0)
        b = PyPhiFloat(1.5)
        assert a != b


class TestPyPhiFloatHash:
    """Hash snapshots ``config.PRECISION`` at construction."""

    def test_hash_consistent_with_eq(self):
        """Equal values hash equal."""
        a = PyPhiFloat(1.0)
        b = PyPhiFloat(1.0 + 1e-14)
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_stable_across_precision_change(self):
        """A value's hash doesn't change when ``config.PRECISION`` changes
        after construction."""
        with config.override(PRECISION=13):
            value = PyPhiFloat(0.123456789)
            original_hash = hash(value)

        with config.override(PRECISION=6):
            # The hash uses the snapshot from construction, not the new
            # PRECISION. Critically, this means a PyPhiFloat placed in a
            # set under PRECISION=13 is still findable under PRECISION=6.
            assert hash(value) == original_hash

    def test_set_membership_stable_across_precision_change(self):
        """A PyPhiFloat in a set stays findable across precision changes."""
        with config.override(PRECISION=13):
            value = PyPhiFloat(0.5)
            container = {value}

        with config.override(PRECISION=6):
            # Look up the same value object — must still be in the set.
            assert value in container

    def test_precision_used_at_construction_not_at_hash_time(self):
        """Two PyPhiFloats constructed under different precisions hash
        according to *their own* construction-time precision."""
        with config.override(PRECISION=13):
            high_precision = PyPhiFloat(0.123456789012)

        with config.override(PRECISION=3):
            low_precision = PyPhiFloat(0.123456789012)

        # high_precision rounds to 13 digits; low_precision rounds to 3.
        # They generally hash differently because their snapshots differ.
        assert high_precision._precision == 13
        assert low_precision._precision == 3
